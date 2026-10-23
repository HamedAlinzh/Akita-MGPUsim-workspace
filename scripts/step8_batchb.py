#!/usr/bin/env python3
"""Step 8 items 3-4: shapes, K per GPU, an odd tile-row count, and
reduce-scatter, each at f ~ 0.555 (the f~0.5 point's f) or 0.814 (f~0.8), on
2 GPUs, cap 8, 4 CUs, tileSize 8.

Model: processor-sharing work-group schedule (step8_model.PSGemm) + the event
message model. Work-group time is linear in its K chunk, 5.4 + 0.514 x K us
(fitted to 137 us at K 256 and the steady split work-groups at 128 and 64);
a cold first round multiplies it by Step 5's factors (plain 172/137, s = 2
72.5/71.2, s = 4 48.8/39.4). Link knobs per case from
ratio_model.derive_knobs, with the GEMM time from this model (+ the 4.4 us tail
it misses, - the 22.75 us launch, as for the capped baseline).

Prints one line per case: shape, waves, knobs, model s=1, best plan, model
plan; RUN lines for step8_runs.sh; and, once runs exist, measured vs model and
the measured split cost per work-group (steady split work-groups x s / steady
plain ones, GPU 1) from the traces.
"""
import collections
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step5_summary as s5  # noqa: E402
import step5_wgs  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step8_edges")
RUNS = os.path.join(OUT, "runs")
FIRST_FACTOR = {1: 172 / 137, 2: 72.5 / 71.2, 4: 48.8 / 39.4}


def wg_times(kg):
    steady = {s: 5.4 + 0.514 * kg / s for s in (1, 2, 4)}
    return steady, {s: steady[s] * FIRST_FACTOR[s] for s in steady}


CASES = [  # name, M, N, K/GPU, collective, f
    ("w15", 192, 256, 256, "all-reduce", 0.555),
    ("w337", 384, 288, 256, "all-reduce", 0.555),
    ("r9", 288, 256, 256, "all-reduce", 0.555),
    ("k128", 256, 256, 128, "all-reduce", 0.555),
    ("k512", 256, 256, 512, "all-reduce", 0.555),
    ("rs05", 256, 256, 256, "reduce-scatter", 0.555),
    ("rs08", 256, 256, 256, "reduce-scatter", 0.814),
]


def setup(name, M, N, kg, coll, f):
    rows, cols = M // 32, N // 32
    steady, first = wg_times(kg)
    g = e.PSGemm(rows, cols, steady, first, m.SIM_LAUNCH)
    _, end = g.run((1,) * rows)
    t_g = end + 4.4 - rm.SIM_LAUNCH_US
    first_share = (m.SIM_LAUNCH + first[1] - rm.SIM_LAUNCH_US) / t_g
    layer = rm.RealLayer(rm.LLAMA70, "MLP down-proj", coll, 8, 4096)
    pt = rm.Point(name, layer, M, N, kg, 4)
    k = rm.derive_knobs(pt, f, measured=(t_g, first_share))
    knobs = (k["link_bw_gbps"], k["link_latency_ns"], k["msg_overhead_ns"])
    cm = e.Comm(2, coll, knobs[0] * 1e9, knobs[1], knobs[2], 4096)
    return g, cm, knobs, rows * cols / 32


def plans(rows):
    if rows <= 9:
        return itertools.product((1, 2, 4), repeat=rows)
    return m.rows_plans_structured(rows)


def split_cost(run, plan):
    """Steady split cost per work-group: mean duration of split work-groups
    not dispatched at time 0, times s, over the mean of plain ones, GPU 1."""
    path = os.path.join(RUNS, run, "wgs.csv")
    if not os.path.exists(path):
        return {}
    rows = [r for r in step5_wgs.load(path) if r["gpu"] == "1"]
    t0 = min(r["start"] for r in rows)
    grid = collections.Counter()
    by_kernel = collections.defaultdict(list)
    for r in rows:
        by_kernel[int(r["kernel_id"].split("@")[0])].append(r)
    # Launch order maps to the plan's bands, as in step5b_summary.plan_ready.
    bands = []
    for row, s in enumerate(plan):
        if s == 1 and bands and bands[-1] == 1:
            continue
        bands.extend([s] * s if s > 1 else [1])
    dur = collections.defaultdict(list)
    for i, kid in enumerate(sorted(by_kernel)):
        s = bands[i]
        for r in by_kernel[kid]:
            if r["start"] > t0 + 1:
                dur[s].append(r["end"] - r["start"])
        grid[s] += 1
    if 1 not in dur:
        return {}
    plain = sum(dur[1]) / len(dur[1])
    return {s: sum(v) / len(v) * s / plain for s, v in dur.items() if s > 1 and v}


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["case | shape | K/GPU | collective | f | waves | -link-bw GB/s | latency ns | "
             "overhead ns | model s=1 | best plan | model plan | model change"]
    runs = []
    for c in CASES:
        name, M, N, kg, coll, f = c
        g, cm, (bw, lat, ovh), waves = setup(*c)
        w1 = cm.window(g.run((1,) * g.rows)[0])
        w, _, plan = e.search(g, cm, plans(g.rows), top=1)[0]
        lines.append(f"{name} | {M}x{N} | {kg} | {coll} | {f} | {waves:.2f} | {bw:.4f} | {lat:.0f} | "
                     f"{ovh:.0f} | {w1:.1f} | {m.name(plan)} | {w:.1f} | {w / w1 - 1:+.1%}")
        print(lines[-1], flush=True)
        flags = (f"-m={M} -n={N} -k={2 * kg} -link-bw={bw:.4f} -link-latency={lat:.0f} "
                 f"-msg-overhead={ovh:.0f}" + (" -reduce-scatter" if coll != "all-reduce" else ""))
        print(f"RUN {name} {flags} | {m.name(plan)}")
        runs.append((name, w1, w, plan))
    lines.append("\nmeasured: case | s=1 measured / model / error | plan measured / model / error | "
                 "measured change | steady split cost per work-group (x s / plain)")
    for name, w1, w, plan in runs:
        a = s5.parse(os.path.join(RUNS, f"{name}_s1"))
        b = s5.parse(os.path.join(RUNS, f"{name}_plan"))
        if not (a and b and "window" in a and "window" in b):
            continue
        sc = split_cost(f"{name}_plan", plan)
        scs = ", ".join(f"s={s}: {v:.2f}" for s, v in sorted(sc.items())) or "-"
        lines.append(f"{name} | {a['window']:.1f} / {w1:.1f} / {w1 / a['window'] - 1:+.1%} | "
                     f"{b['window']:.1f} / {w:.1f} / {w / b['window'] - 1:+.1%} | "
                     f"{b['window'] / a['window'] - 1:+.1%} (verified {a['ok'] and b['ok']}) | {scs}")
    with open(os.path.join(OUT, "batchb.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[len(CASES) + 1:]))


if __name__ == "__main__":
    main()
