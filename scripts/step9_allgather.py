#!/usr/bin/env python3
"""Step 9 item 4 (design only): all-gather of A before the GEMM, the other
half of a tensor-parallel layer, on the engines. Queue model, no runs.

Each GPU holds M/p rows of A (full K) and computes C = A x B_local over all
M rows (256 x 256 x 256, the f~0.5 GEMM per GPU). Its own rows are ready at
the launch; every peer's engine sends its rows to every GPU as messages of
one row of tiles (32 rows) x one K chunk, in row order, chunk 0 first, on
per-peer queues (one link per pair; at 2 GPUs the single link). A message
costs the engine overhead, its transfer at 0.8 of the link and the wire
latency, as in step8_model.Comm.

GEMM: the Step 8 processor-sharing schedule (4 CUs, 8 resident, cold first
round), one launch per row of tiles per K chunk, launched local rows first
and then in arrival order, each held at the CP until its message arrives
(ordered dispatch, so an unreleased launch holds the later ones). A split
row (s K chunks) needs only its chunk's columns of A, so its first chunk
can start before the rest of the row has arrived; its s partials are added
locally afterwards (one more pass over the row, not modelled).

Knobs: f = 0.5 by ratio_model.derive_knobs, with the bytes per link being one
GPU's A shard (M/p x K x 4, the same as C/p: the reduce-scatter rule), and
the capped GEMM's measured time. Ring order: a ring all-gather on the same
fabric uses one link per direction, so the shard from distance d arrives
after d shard times.

Prints the schedules for 2 and 8 GPUs to stdout and
results/step9_baselines/allgather.txt.
"""
import heapq
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step9_baselines")
ROWS, COLS, K = 8, 8, 256
ROW_BYTES = 32 * K * 4  # one row of tiles of A


def knobs(p, f=0.5):
    pt = rm.capped_points(4096)[1]
    pt = rm.Point(pt.label, rm.RealLayer(pt.layer.model, pt.layer.gemm, "reduce-scatter",
                                         pt.layer.p, pt.layer.tokens),
                  pt.m, pt.n, pt.kg, pt.cus)
    k = rm.derive_knobs(pt, f, measured=rm.capped_measured(), num_gpu=p)
    return k["link_bw_gbps"] * 1e9, k["link_latency_ns"], k["msg_overhead_ns"]


def arrivals(p, plan, ring=False, eff=0.8, calib=1.07):
    """Arrival time (us) of every (row, chunk) at GPU 0. GPU 0 owns rows
    [0, R/p); peer q's rows go over link q in row order."""
    bw, lat, ovh = knobs(p)
    per = ROWS // p
    wire = calib * (ovh + lat) / 1e3 - ovh / 1e3
    out = {}
    for r in range(per):
        for c in range(plan[r]):
            out[(r, c)] = 0.0
    for q in range(1, p):
        t_lane = t_link = 0.0
        for r in range(q * per, (q + 1) * per):
            s = plan[r]
            x = (ROW_BYTES / s + 16) / (bw * eff) * 1e6
            for c in range(s):
                t_lane += ovh / 1e3
                t_link = max(t_link, t_lane) + x
                out[(r, c)] = t_link + wire
    if ring:
        # Distance d = q here; the shard is forwarded d - 1 times, each hop
        # behind the previous one on the next GPU's link.
        shard = sum((ROW_BYTES + 16 * plan[r]) / (bw * eff) * 1e6 for r in range(per))
        for (r, c), t in list(out.items()):
            q = r // per
            if q:
                out[(r, c)] = t + (q - 1) * shard
    return out


def gemm(plan, arrive, warm=False):
    """GEMM end (us after the launch) with launches held until arrival."""
    steady, first = m.SIM_STEADY, None if warm else m.SIM_FIRST
    order = sorted({(arrive[(r, c)], r, c) for r in range(ROWS) for c in range(plan[r])})
    jobs = [(t, plan[r]) for t, r, c in order for _ in range(COLS)]
    active = [[] for _ in range(4)]
    t, j = 0.0, 0
    while j < len(jobs) or any(active):
        while j < len(jobs) and jobs[j][0] <= t + 1e-9:
            cu = min(range(4), key=lambda c: len(active[c]))
            if len(active[cu]) >= 8:
                break
            s = jobs[j][1]
            active[cu].append(first[s] if first and t == 0 else steady[s])
            j += 1
        steps = [w * len(act) / 8 for act in active for w in act]
        if j < len(jobs) and jobs[j][0] > t + 1e-9:
            steps.append(jobs[j][0] - t)
        dt = min(steps)
        for act in active:
            n = len(act)
            for i in range(n):
                act[i] -= dt * 8 / n
        t += dt
        for i, act in enumerate(active):
            active[i] = [w for w in act if w > 1e-9]
    return m.SIM_LAUNCH + t


def best(p, ring=False):
    per = ROWS // p
    remote = ROWS - per
    res = []
    for ss in itertools.product((1, 2, 4), repeat=remote):
        plan = (1,) * per + ss
        res.append((gemm(plan, arrivals(p, plan, ring)), plan))
    res.sort()
    return res[0]


def main():
    lines = ["All-gather of A before the GEMM, f = 0.5 (queue model, cold first round)"]
    alone = gemm((1,) * ROWS, {(r, 0): 0.0 for r in range(ROWS)})
    lines.append(f"GEMM alone: {alone:.1f} us")
    for p in (2, 8):
        bw, lat, ovh = knobs(p)
        ones = (1,) * ROWS
        arr = arrivals(p, ones)
        last = max(arr.values())
        lines.append(f"\n{p} GPUs: -link-bw {bw / 1e9:.4f} GB/s, -link-latency {lat:.0f} ns, "
                     f"-msg-overhead {ovh:.0f} ns; last row arrives at {last:.1f} us")
        no = last + alone
        s1 = gemm(ones, arr)
        wp, plan = best(p)
        lines.append(f"  no overlap (gather, then GEMM): {no:.1f} us")
        lines.append(f"  per-row overlap, s = 1, local rows first: {s1:.1f} us "
                     f"({s1 / no - 1:+.1%} vs no overlap)")
        lines.append(f"  + best split plan {m.name(plan)}: {wp:.1f} us ({wp / s1 - 1:+.1%} vs s = 1)")
        if p > 2:
            r1 = gemm(ones, arrivals(p, ones, ring=True))
            wr, pr = best(p, ring=True)
            lines.append(f"  ring order (one link per direction): s = 1 {r1:.1f} us, "
                         f"best plan {m.name(pr)} {wr:.1f} us")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "allgather.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
