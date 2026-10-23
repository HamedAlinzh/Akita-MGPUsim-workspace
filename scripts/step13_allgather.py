#!/usr/bin/env python3
"""Step 13: K-chunk splits of the remote rows in the all-gather before the
GEMM (input side of the split-K plan). Same config as Step 12 (256 x 256 x
256 per GPU, 4 CUs, tileSize 8, cap 8, f 0.5 knobs).

A remote row split s ways arrives as s messages (32 rows x K/s of A), each
raising gpuN/a/rowR/chunkC, and runs as s launches held on their chunk; the
engine then adds the row's s - 1 partial buffers into C (one more pass over
the row, started when the tracker reports all its tiles complete).

Model: scripts/step9_allgather.py (the design note's queue model), with
per-row GEMM end times, plus the partial-sum pass on the engine, one row
after another: reading the row and its s - 1 partials and writing the sum
cost (s + 1) x 0.512 us (Step 12: the engine wrote one 32 KB row into L2
every 0.512 us) and the adds (s - 1) x 512 lines x 1 cycle at 1 GHz. The
window is the later of the GEMM end and the last sum. s = 8 work-group
times (not measured) are extrapolated with the linear work-group time
5.4 + 0.514 x K us.

Runs (results/step13_allgather_split/runs/): g8_s4 (1,4,4,4,4,4,4,4),
g8_s2, g8_s4_nosum (sum switched off: does not verify, only for the sum's
cost), g2_last4 (1,1,1,1,1,1,1,4), g2_last2, then g8_s8 only if g8_s4 beats
g8_s2 by more than 3%; defaults_433. Step 12 runs are reused for s = 1,
gather-then-GEMM and the floor.

Writes runs.txt (before running; --s8 adds g8_s8) and summary.txt.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step9_allgather as ag  # noqa: E402
import step12_allgather as s12  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step13_allgather_split")
WRITE_US = 0.512  # engine: one 32 KB row into L2 (Step 12)
ADD_US = 512 * 1e-3  # 512 lines x 1 cycle at 1 GHz, per partial added

# s = 8: K chunk 32, work-group time 5.4 + 0.514 x 32 us, first round scaled as s = 4's
STEADY = {**m.SIM_STEADY, 8: 5.4 + 0.514 * 32}
FIRST = {**m.SIM_FIRST, 8: m.SIM_FIRST[4] * (5.4 + 0.514 * 32) / m.SIM_STEADY[4]}

PLANS = {"g8_s4": (8, (1,) + (4,) * 7), "g8_s2": (8, (1,) + (2,) * 7),
         "g8_s4_nosum": (8, (1,) + (4,) * 7), "g2_last4": (2, (1,) * 7 + (4,)),
         "g2_last2": (2, (1,) * 7 + (2,)), "g8_s8": (8, (1,) + (8,) * 7)}


def gemm_rows(plan, arrive):
    """step9_allgather.gemm, also returning each row's end (us)."""
    order = sorted({(arrive[(r, c)], r, c) for r in range(ag.ROWS) for c in range(plan[r])})
    jobs = [(t, plan[r], r) for t, r, c in order for _ in range(ag.COLS)]
    active = [[] for _ in range(4)]
    t, j, end = 0.0, 0, {}
    while j < len(jobs) or any(active):
        while j < len(jobs) and jobs[j][0] <= t + 1e-9:
            cu = min(range(4), key=lambda c: len(active[c]))
            if len(active[cu]) >= 8:
                break
            s = jobs[j][1]
            active[cu].append([FIRST[s] if t == 0 else STEADY[s], jobs[j][2]])
            j += 1
        steps = [w[0] * len(act) / 8 for act in active for w in act]
        if j < len(jobs) and jobs[j][0] > t + 1e-9:
            steps.append(jobs[j][0] - t)
        dt = min(steps)
        for act in active:
            n = len(act)
            for w in act:
                w[0] -= dt * 8 / n
        t += dt
        for i, act in enumerate(active):
            for w in act:
                if w[0] <= 1e-9:
                    end[w[1]] = m.SIM_LAUNCH + t
            active[i] = [w for w in act if w[0] > 1e-9]
    return m.SIM_LAUNCH + t, end


def sum_cost(s):
    return (s + 1) * WRITE_US + (s - 1) * ADD_US


def model(p, plan, with_sum=True):
    """Window (us): GEMM end, or the last partial sum if later."""
    total, end = gemm_rows(plan, ag.arrivals(p, plan))
    if not with_sum:
        return total
    t = 0.0
    for r in sorted((r for r in range(ag.ROWS) if plan[r] > 1), key=lambda r: end[r]):
        t = max(t, end[r]) + sum_cost(plan[r])
    return max(total, t)


def name(plan):
    return ",".join(str(x) for x in plan)


def flags(run):
    p, plan = PLANS[run]
    f = s12.flags(p, "rows") + f" -allgather-plan={name(plan)}"
    return f + (" -allgather-no-sum" if run.endswith("nosum") else "")


def parse(run):
    path = os.path.join(OUT, "runs", run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    r = {"ok": "Passed!" in text}
    mm = re.search(r"allgather phases: mode=\w+ window=([\d.]+) gather=[\d.]+ "
                   r"gemm_wait=([\d.]+)", text) or re.search(r"allreducegemm phases: window=([\d.]+)", text)
    if mm:
        r["window"] = float(mm.group(1)) * 1e6
    w = re.search(r"wall_s=([\d.]+)", text)
    r["wall"] = float(w.group(1)) if w else None
    r["remote"] = [(int(g), int(row), int(s), *(float(x) * 1e6 for x in rest)) for g, row, s, *rest
                   in re.findall(r"allgather remote: gpu=(\d+) row=(\d+) s=(\d+) first_release=([\d.-]+) "
                                 r"last_in_l2=([\d.]+) sum_ready=([\d.]+) sum_done=([\d.]+)", text)]
    r["launches"] = [tuple(float(x) * 1e6 for x in g) for g in re.findall(
        r"allgather launch: name=\S+ held=([\d.]+) signal=([\d.]+) released=([\d.]+) "
        r"done=([\d.]+)", text)]
    r["dispatchers"] = [int(x) for x in re.findall(r"allgather dispatchers: gpu=\d+ max_busy=(\d+)", text)]
    p = re.search(r"allgather push: done=([\d.]+) last_arrival=([\d.]+)", text)
    if p:
        r["push_done"], r["last_arrival"] = float(p.group(1)) * 1e6, float(p.group(2)) * 1e6
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = ["g8_s4", "g8_s2", "g8_s4_nosum", "g2_last4", "g2_last2"]
    if "--s8" in sys.argv or os.path.exists(os.path.join(OUT, "runs", "g8_s8")):
        runs.append("g8_s8")
    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step13_allgather.py\n")
        for run in runs:
            fh.write(f"{run} {flags(run)}\n")
        fh.write("defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 "
                 "-algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 "
                 "-collective-threads=16384\n")

    L = ["Model (us): plan | without the partial sum (Step 9 model) | with it | moved"]
    for p in (2, 8):
        L.append(f"{p} GPUs s = 1: {model(p, (1,) * 8):.1f}")
    for run in runs:
        p, plan = PLANS[run]
        a, b = model(p, plan, False), model(p, plan)
        L.append(f"{run} {name(plan)} | {a:.1f} | {b:.1f} | {b - a:+.1f} ({b / a - 1:+.2%})")
    L.append("sum pass per row (model): s = 2 %.2f us, s = 4 %.2f us, s = 8 %.2f us"
             % (sum_cost(2), sum_cost(4), sum_cost(8)))

    ref = {p: {mode: s12.parse(f"g{p}_{mode}") for mode in ("rows", "bulk", "local")} for p in (2, 8)}
    L.append("\nrun | plan | window us | model (with sum) | error | Step 12 s = 1 / gather-then-GEMM / "
             "floor | vs s = 1 | held launches | max dispatchers busy | releases late (>0.1 us after "
             "max(held, signal)) | launch reached CP after its chunk | verified | wall s")
    res = {}
    for run in runs:
        p, plan = PLANS[run]
        r = parse(run)
        res[run] = r
        mp = model(p, plan, not run.endswith("nosum"))
        rf = ref[p]
        refs = " / ".join(f"{rf[k]['window']:.1f}" if rf[k] else "-" for k in ("rows", "bulk", "local"))
        if not r or "window" not in r:
            L.append(f"{run} | {name(plan)} | (not run) | {mp:.1f} | | {refs}")
            continue
        late = [rel - max(h, s) for h, s, rel, _ in r["launches"]]
        after = sum(1 for h, s, _, _ in r["launches"] if h > s)
        err = f"{r['window'] / mp - 1:+.1%}"
        vs = f"{r['window'] / rf['rows']['window'] - 1:+.1%}" if rf["rows"] else "-"
        L.append(f"{run} | {name(plan)} | {r['window']:.1f} | {mp:.1f} | {err} | {refs} | {vs} | "
                 f"{len(r['launches'])} | {max(r['dispatchers'], default=0)} | "
                 f"{sum(1 for d in late if d > 0.1)} (max {max(late, default=0) * 1e3:.0f} ns) | {after} | "
                 f"{r['ok'] if not run.endswith('nosum') else str(r['ok']) + ' (expected: no sum)'} | "
                 f"{r['wall']}")

    L.append("\nRemote rows, GPU 0 (us): row s | first launch released | last chunk in L2 | "
             "sum ready -> done (cost)")
    for run in runs:
        r = res[run]
        if not r or not r.get("remote"):
            continue
        cells = []
        for g, row, s, first, inl2, ready, done in r["remote"]:
            if g == 0:
                sm = f"{ready:.1f}->{done:.1f} ({done - ready:.2f})" if done > 0 else "-"
                cells.append(f"row {row} s{s}: {first:.1f} | {inl2:.1f} | {sm}")
        L.append(f"{run}: " + "; ".join(cells))
        costs = [done - ready for _, _, s, _, _, ready, done in r["remote"] if done > 0]
        if costs:
            last_ready = max(ready for *_, ready, done in r["remote"] if done > 0)
            last_done = max(done for *_, done in r["remote"])
            L.append(f"  partial sums: {len(costs)} rows, cost per row {min(costs):.2f}-{max(costs):.2f} "
                     f"us (mean {sum(costs) / len(costs):.2f}); last ready {last_ready:.1f}, last done "
                     f"{last_done:.1f} (tail {last_done - last_ready:.2f} us)")
    a, b = res.get("g8_s4"), res.get("g8_s4_nosum")
    if a and b and "window" in a and "window" in b:
        L.append(f"sum cost on the window (g8_s4 - g8_s4_nosum): {a['window'] - b['window']:+.1f} us")
    a, b = res.get("g8_s4"), res.get("g8_s2")
    if a and b and "window" in a and "window" in b:
        gain = a["window"] / b["window"] - 1
        L.append(f"g8_s4 vs g8_s2: {gain:+.1%} -> s = 8 run "
                 f"{'warranted (> 3%)' if gain < -0.03 else 'skipped (not > 3%)'}")
    d = parse("defaults_433")
    if d and "window" in d:
        L.append(f"defaults: window {d['window']:.3f} us, verified {d['ok']}")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
