#!/usr/bin/env python3
"""Step 7 models (results/step7_baselines/model.txt).

1. Software-overlap baseline (-sw-overlap): every band's GEMM launched up
   front; the host waits for band b, adds its split-K partials (s-1 calls of
   mccl.Accumulate, PRESUM_US each) and all-reduces it with mccl (F + bytes /
   (eta x link bandwidth)), one band after another. F and eta are fitted to
   the capped mccl runs (step4b_capped f05_mccl, f08_mccl: the same 256 KB
   all-reduce at two link speeds, -collective-threads 16384). PRESUM_US is
   measured in TestSWOverlap (5 Accumulate calls, 247 us). The collectives'
   kernels share the CUs with the GEMM: its work-groups run 1/(1-phi) slower,
   phi = 0 or 0.25 (not measured before these runs). A split band runs its K
   chunks one after another over all its rows, as its launches do.
   Searches band layouts (contiguous, at least 2 bands) for s = 1 and for
   the plan 4,4,4,4,1,1,2,2 (uniform split per band).
2. 8 GPUs at f~0.5: per-row plan search with per-peer engine queues.
3. The plans picked for the rank test (from step6_model's full search).
"""
import itertools
import heapq
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402

ROOT = m.ROOT
OUT = os.path.join(ROOT, "results", "step7_baselines")
C_BYTES = 256 * 256 * 4
PRESUM_US = 247.0 / 5
PLAN = (4, 4, 4, 4, 1, 1, 2, 2)


def mccl_allreduce(run):
    text = open(os.path.join(ROOT, "results", "step4b_capped", "runs", run, "stdout.txt")).read()
    return float(re.search(r"allreduce=([\d.]+)", text).group(1)) * 1e6


def fit_sw():
    k = m.knobs()
    a5, a8 = mccl_allreduce("f05_mccl"), mccl_allreduce("f08_mccl")
    bw5, bw8 = k["f05"][0], k["f08"][0]
    eta = C_BYTES * (1 / bw8 - 1 / bw5) / ((a8 - a5) * 1e-6)
    F = a5 - C_BYTES / (eta * bw5) * 1e6
    return F, eta, a5, a8


def band_gemm(bands, phi, cold=True, rows=8, cols=8, slots=32):
    """Tile-row ready times for bands [(first row, n rows, s)], launched in
    order, each split band chunk by chunk over all its rows."""
    scale = 1 / (1 - phi)
    heap = [0.0] * slots
    ready = [0.0] * rows
    for r0, n, s in bands:
        for _ in range(s):
            for r in range(r0, r0 + n):
                for _ in range(cols):
                    start = heapq.heappop(heap)
                    dur = (m.SIM_FIRST[s] if cold and start == 0 else m.SIM_STEADY[s]) * scale
                    heapq.heappush(heap, start + dur)
                    ready[r] = max(ready[r], start + dur)
    return [m.SIM_LAUNCH + x for x in ready]


def sw_window(bands, phi, F, eta, bw):
    ready = band_gemm(bands, phi)
    t = 0.0
    for r0, n, s in bands:
        done = max(ready[r0:r0 + n])
        t = max(t, done) + (s - 1) * PRESUM_US + F + n * 32 * 256 * 4 / (eta * bw) * 1e6
    return t


def layouts(plan, min_bands=2):
    """Contiguous band layouts with a uniform split factor per band."""
    rows = len(plan)
    for cuts in itertools.product((0, 1), repeat=rows - 1):
        bands, start = [], 0
        for i, c in enumerate(cuts + (1,)):
            if c:
                bands.append((start, i + 1 - start, plan[start]))
                start = i + 1
        if len(bands) >= min_bands and all(len(set(plan[a:a + n])) == 1 for a, n, _ in bands):
            yield bands


def rows_str(bands):
    return ",".join(str(n * 32) for _, n, _ in bands)


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    out = lines.append
    K = m.knobs()
    F, eta, a5, a8 = fit_sw()
    bw = K["f05"][0]
    out(f"1. Software overlap (f~0.5, 2 GPUs). Fit: F = {F:.1f} us per all-reduce call, "
        f"eta = {eta:.3f} (mccl after the GEMM: {a5:.1f} us at f~0.5, {a8:.1f} at f~0.8); "
        f"presum {PRESUM_US:.1f} us per Accumulate call.")
    one = [(0, 8, 1)]
    for phi in (0.0, 0.25):
        out(f"  phi = {phi}: one band (no overlap) {sw_window(one, phi, F, eta, bw):.1f} us")
    chosen = {}
    for label, plan in (("s=1", (1,) * 8), ("plan 4,4,4,4,1,1,2,2", PLAN)):
        for phi in (0.0, 0.25):
            res = sorted((sw_window(b, phi, F, eta, bw), rows_str(b), b) for b in layouts(plan))
            out(f"  {label}, phi = {phi}: best layout (>= 2 bands) {res[0][1]} -> "
                f"{res[0][0]:.1f} us; worst {res[-1][1]} -> {res[-1][0]:.1f} us")
            if phi == 0.0:
                chosen[label] = res[0][2]
    for label, b in chosen.items():
        preds = [sw_window(b, phi, F, eta, bw) for phi in (0.0, 0.25)]
        out(f"  run {label}: bands {rows_str(b)}: predicted {preds[0]:.1f} us (phi 0), "
            f"{preds[1]:.1f} us (phi 0.25)")

    out("\n2. 8 GPUs at f~0.5 (per-pair mesh, per-peer engine queues)")
    g = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    cm8 = m.Comm(8, "all-reduce", *K["f05_8gpu"], 4096, engine_per_link=True)
    w1 = cm8.window(g.run((1,) * 8)[0])
    best = m.search(g, cm8, itertools.product((1, 2, 4), repeat=8), top=3)
    out(f"  s=1 {w1:.1f} us; best " + "; ".join(
        f"{m.name(p)} {w:.1f} us ({w / w1 - 1:+.1%})" for w, _, p in best))
    out(f"  plan 4,4,4,4,1,1,2,2: {cm8.window(g.run(PLAN)[0]):.1f} us")
    one_q = m.Comm(8, "all-reduce", *K["f05_8gpu"], 4096)
    out(f"  s=1 with one engine queue instead: {one_q.window(g.run((1,) * 8)[0]):.1f} us")

    out("\n3. Rank test plans (f~0.5, 2 GPUs, cold; 31 distinct predicted windows over "
        "6,561 plans)")
    cm2 = m.Comm(2, "all-reduce", *K["f05"], 4096)
    for label, plan in (("2nd best distinct", (4, 4, 4, 4, 4, 4, 4, 1)),
                        ("10th best distinct", (1, 2, 4, 1, 1, 1, 1, 1)),
                        ("median", (4, 1, 4, 4, 2, 1, 2, 4)),
                        ("worst", (1, 1, 1, 1, 4, 4, 4, 4))):
        out(f"  {label}: {m.name(plan)} {cm2.window(g.run(plan)[0]):.1f} us")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "model.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
