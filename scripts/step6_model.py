#!/usr/bin/env python3
"""Step 6 model: work-group scheduling + per-link message queues, for the
simulated points (2 and 4 GPUs, all-reduce and reduce-scatter) and a real
MI300X projection.

GEMM: slots run work-groups in dispatch order (ordered dispatch); a split row
runs its K chunks one after another; a tile is ready when its last work-group
ends. Times per work-group: steady[s], and first[s] for those started at time
0 when the cold first round is kept.

Comm: tile t (one message per tile) is owned by GPU t mod p. Every non-owner
sends its partial to the owner; for an all-reduce the owner sends the sum to
every other GPU. Every message is ready when its tile is (the partials'
arrival before the sum is not modelled). Each engine spends its per-message
overhead serially (one engine per GPU in the simulator; one per link in the
real projection, i.e. a DMA queue per peer), then the message waits for its
link, which sends one message at a time. Window = the last link finish + a
tail: the Step 5 round trip 2 x 1.07 x (overhead + latency) + 2 x transfer
for an all-reduce, half of it for a reduce-scatter.
"""
import csv
import heapq
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step5_wgs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "step6_paper")
RUNS = os.path.join(OUT, "runs")

# Simulator work-group times (us), Step 5 traces at 256x256, K/GPU 256.
T = 137.0
SIM_STEADY = {1: T, 2: 1.04 * T / 2, 4: 1.15 * T / 4}
SIM_FIRST = {1: 172.0, 2: 72.5, 4: 48.8}
SIM_LAUNCH = 46.0


class Gemm:
    def __init__(self, rows, cols, slots, steady, first=None, launch=0.0):
        self.rows, self.cols, self.slots = rows, cols, slots
        self.steady, self.first, self.launch = steady, first, launch

    def run(self, plan):
        """Tile ready times and GEMM end, us after the launch."""
        heap = [0.0] * self.slots
        ready = [0.0] * (self.rows * self.cols)
        end = 0.0
        for row, s in enumerate(plan):
            for _ in range(s):
                for col in range(self.cols):
                    start = heapq.heappop(heap)
                    dur = self.first[s] if self.first and start == 0 else self.steady[s]
                    fin = start + dur
                    heapq.heappush(heap, fin)
                    t = row * self.cols + col
                    ready[t] = max(ready[t], fin)
                    end = max(end, fin)
        return [self.launch + r for r in ready], self.launch + end


class Comm:
    def __init__(self, p, collective, bw, lat_ns, ovh_ns, tile_bytes, header=16,
                 eff=0.8, calib=1.07, engine_per_link=False):
        self.p, self.collective = p, collective
        self.x = (tile_bytes + header) / (bw * eff) * 1e6
        self.ovh = ovh_ns / 1e3
        rt = 2 * calib * (ovh_ns + lat_ns) / 1e3 + 2 * self.x
        self.tail = rt if collective == "all-reduce" else rt / 2
        self.engine_per_link = engine_per_link

    def window(self, ready):
        p = self.p
        msgs = {a: [] for a in range(p)}  # (ready, tile, to)
        for t, r in enumerate(ready):
            o = t % p
            for a in range(p):
                if a != o:
                    msgs[a].append((r, t, 0, o))
                elif self.collective == "all-reduce":
                    for b in range(p):
                        if b != o:
                            msgs[a].append((r, t, 1, b))
        link_free = {}
        last = 0.0
        for a in range(p):
            engine = {}
            for r, _, _, b in sorted(msgs[a]):
                key = b if self.engine_per_link else 0
                handed = max(engine.get(key, 0.0), r) + self.ovh
                engine[key] = handed
                done = max(link_free.get((a, b), 0.0), handed) + self.x
                link_free[(a, b)] = done
                last = max(last, done)
        return last + self.tail


def rows_plans_full(rows):
    return itertools.product((1, 2, 4), repeat=rows)


def rows_plans_structured(rows):
    """[4]*a + [2]*b + [1]*m + [2]*c + [4]*d."""
    for a in range(rows + 1):
        for b in range(rows + 1 - a):
            for c in range(rows + 1 - a - b):
                for d in range(rows + 1 - a - b - c):
                    m = rows - a - b - c - d
                    yield (4,) * a + (2,) * b + (1,) * m + (2,) * c + (4,) * d


def search(gemm, comm, plans, top=3):
    res = []
    for plan in plans:
        ready, end = gemm.run(plan)
        res.append((comm.window(ready), end, tuple(plan)))
    res.sort()
    return res[:top]


def name(plan):
    return ",".join(str(s) for s in plan)


def parse(run):
    path = os.path.join(RUNS, run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(r"window=([\d.]+) gemm_wait=([\d.]+) \w+=([\d.]+)(?: first_ready=([\d.]+))?", text)
    if not m:
        return dict(ok=False)
    return dict(ok="Passed!" in text, window=float(m.group(1)) * 1e6,
                gemm=float(m.group(2)) * 1e6,
                first=float(m.group(4)) * 1e6 if m.group(4) else None)


def knobs():
    rows = {r["point"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "step0_ratio", "points_capped.csv")))}
    rows.update({r["point"]: r for r in csv.DictReader(
        open(os.path.join(ROOT, "results", "step0_ratio", "points_step6.csv")))})
    k = lambda r: (float(r["link_bw_gbps"]) * 1e9, float(r["link_latency_ns"]),
                   float(r["msg_overhead_ns"]))
    return {"f05": k(rows["f~0.5"]), "f08": k(rows["f~0.8"]),
            "f025": k(rows["f~0.25 capped"]), "f05_4gpu": k(rows["f~0.5 4 GPUs"]),
            "f05_8gpu": k(rows["f~0.5 8 GPUs"])}


def calibrate_rounds(run, slots=32):
    """Plain work-group times from an s = 1 run's trace (GPU 1): the first
    round's mean and the rest's."""
    path = os.path.join(RUNS, run, "wgs.csv")
    if not os.path.exists(path):
        return None
    rows = sorted((r for r in step5_wgs.load(path) if r["gpu"] == "1"),
                  key=lambda r: r["start"])
    first = rows[:slots]
    rest = rows[slots:]
    mean = lambda rs: sum(r["end"] - r["start"] for r in rs) / len(rs)
    return mean(first), mean(rest)


def main():
    os.makedirs(OUT, exist_ok=True)
    K = knobs()
    lines = []
    out = lines.append

    sim256 = Gemm(8, 8, 32, SIM_STEADY, SIM_FIRST, SIM_LAUNCH)
    sim256_warm = Gemm(8, 8, 32, SIM_STEADY, None, SIM_LAUNCH)
    plan5b = (4, 4, 4, 4, 1, 1, 2, 2)
    flat = (1,) * 8

    def comm(point, p=2, coll="all-reduce"):
        bw, lat, ovh = K[point]
        return Comm(p, coll, bw, lat, ovh, 4096)

    # f~0.25: 512x320, 16 x 10 tiles, reduce-scatter. Work-group times from
    # the s = 1 run's trace if it exists, else the 256x256 ones.
    cal = calibrate_rounds("f025_tile_s1")
    steady25, first25 = dict(SIM_STEADY), dict(SIM_FIRST)
    if cal:
        r1, r2 = cal
        steady25 = {s: v * r2 / T for s, v in SIM_STEADY.items()}
        first25 = dict(SIM_FIRST)
        first25[1] = r1
        out(f"f~0.25 recalibrated from f025_tile_s1: plain round 1 {r1:.1f} us, later "
            f"rounds {r2:.1f} us (256x256: 172, 137); split times scaled by {r2 / T:.3f}")
    sim512 = Gemm(16, 10, 32, steady25, first25, SIM_LAUNCH)

    out("\nSearches (simulator; first round kept = cold)")
    searches = {}
    for label, gemm, cm, plans in [
        ("f05 2 GPUs", sim256, comm("f05"), rows_plans_full(8)),
        ("f025 2 GPUs RS", sim512, comm("f025", 2, "reduce-scatter"), rows_plans_structured(16)),
        ("f05 4 GPUs", sim256, comm("f05_4gpu", 4), rows_plans_full(8)),
    ]:
        best = search(gemm, cm, plans)
        searches[label] = best[0][2]
        s1 = cm.window(gemm.run((1,) * gemm.rows)[0])
        out(f"{label}: s=1 {s1:.1f} us; best " + "; ".join(
            f"{name(p)} {w:.1f} us ({w / s1 - 1:+.1%}), gemm {e:.0f}" for w, e, p in best))
    p5b_4 = comm("f05_4gpu", 4).window(sim256.run(plan5b)[0])
    out(f"f05 4 GPUs: plan {name(plan5b)} {p5b_4:.1f} us")

    # Predictions for the Step 6 runs (a priori, from the model).
    out("\nModel check: run | measured window | model window | error | measured gemm | "
        "model gemm")
    preds = [
        ("warm_f05_s1", sim256_warm, comm("f05"), flat),
        ("warm_f05_plan", sim256_warm, comm("f05"), plan5b),
        ("f025_tile_s1", sim512, comm("f025", 2, "reduce-scatter"), (1,) * 16),
        ("f025_tile_plan", sim512, comm("f025", 2, "reduce-scatter"), searches["f025 2 GPUs RS"]),
        ("f05_4gpu_tile_s1", sim256, comm("f05_4gpu", 4), flat),
        ("f05_4gpu_tile_plan", sim256, comm("f05_4gpu", 4), searches["f05 4 GPUs"]),
    ]
    for run, gemm, cm, plan in preds:
        ready, end = gemm.run(plan)
        w = cm.window(ready)
        r = parse(run)
        if r and "window" in r:
            out(f"{run} | {r['window']:.1f} | {w:.1f} | {w / r['window'] - 1:+.1%} | "
                f"{r['gemm']:.1f} | {end:.1f}")
        else:
            out(f"{run} | - | {w:.1f} | - | - | {end:.1f}")
    r = parse("f025_bulk_s1")
    if r and "window" in r:
        _, end = sim512.run((1,) * 16)
        cm = comm("f025", 2, "reduce-scatter")
        w = r["gemm"] + cm.window([0.0] * 160)
        out(f"f025_bulk_s1 | {r['window']:.1f} | {w:.1f} (measured gemm + comm) | "
            f"{w / r['window'] - 1:+.1%} | {r['gemm']:.1f} | {end:.1f}")
        out(f"f025 achieved f (bulk: (window - gemm) / window) = "
            f"{(r['window'] - r['gemm']) / r['window']:.3f}")

    # Real MI300X projection.
    out("\nMI300X projection (model only; TP 8, 4096 tokens per GPU, 304 CUs, one 256x256 "
        "work-group per CU, BF16, 64 GB/s x 0.8 per link, 1 us per message = 172 ns "
        "wire + 828 ns engine, one DMA queue per peer link; split costs 1.04x (s=2) and "
        "1.15x (s=4) from the simulator's steady rounds, an assumption; no cold first round)")
    out("point | layer | tiles | waves | s=1 gemm | s=1 window | best plan | plan gemm | "
        "window | gain (us, model)")
    real_points = [(pt.label, pt.layer) for pt in rm.points(4096)]
    for label, layer in real_points:
        cols = layer.model.hidden // rm.REAL_TILE
        rows = 4096 // rm.REAL_TILE
        tiles = rows * cols
        t_wg = layer.t_gemm() * 1e6 * rm.REAL_CUS / tiles
        g = Gemm(rows, cols, rm.REAL_CUS,
                 {1: t_wg, 2: 1.04 * t_wg / 2, 4: 1.15 * t_wg / 4})
        cm = Comm(8, layer.collective, rm.REAL_LINK_BW, rm.REAL_WIRE_NS,
                  rm.REAL_MSG_LATENCY_NS - rm.REAL_WIRE_NS, rm.REAL_TILE ** 2 * rm.BF16,
                  header=0, eff=rm.REAL_LINK_EFF, calib=1.0, engine_per_link=True)
        r1, e1 = g.run((1,) * rows)
        w1 = cm.window(r1)
        best = search(g, cm, rows_plans_structured(rows), top=1)[0]
        out(f"{label} | {layer.model.name} {layer.gemm} {layer.collective} | {tiles} | "
            f"{tiles / rm.REAL_CUS:.2f} | {e1:.0f} | {w1:.0f} | {name(best[2])} | "
            f"{best[1]:.0f} | {best[0]:.0f} | {best[0] / w1 - 1:+.1%}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "model.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
