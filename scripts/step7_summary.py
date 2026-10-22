#!/usr/bin/env python3
"""Step 7 (batch 1) summary: software overlap, the direct software
collective on 4 and 8 GPUs, the 8-GPU engine runs, and the model's rank test,
each against the model's prediction (step6_model / step7_model). Reuses the
f~0.5 2-GPU engine runs of Steps 5-6 for the rank test and the s = 1 / plan
references."""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402
import step6_model as m  # noqa: E402
import step7_model as m7  # noqa: E402

R = os.path.join(s5.ROOT, "results")
RUNS = os.path.join(R, "step7_baselines", "runs")
PLAN = (4, 4, 4, 4, 1, 1, 2, 2)


def run(path):
    path = path if os.path.isabs(path) else os.path.join(R, path)
    text = open(os.path.join(path, "stdout.txt")).read()
    w = re.search(r"window=([\d.]+) gemm_wait=([\d.]+)", text)
    f = re.search(r"first_ready=([\d.]+)", text)
    return dict(ok="Passed!" in text, window=float(w.group(1)) * 1e6,
                gemm=float(w.group(2)) * 1e6, first=float(f.group(1)) * 1e6 if f else None)


def ranks(xs):
    """1-based ranks, ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb)


def main():
    K = m.knobs()
    g = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    hw_s1, hw_plan = run("step5_splitk/runs/f05_tile_s1"), \
        run("step5b_bounds/runs/c_f05_plan")
    print(f"References (2 GPUs, f~0.5, engine tile): s=1 {hw_s1['window']:.1f} us, plan "
          f"{hw_plan['window']:.1f} us; mccl ring after the GEMM (no overlap) "
          f"{run('step4b_capped/runs/f05_mccl')['window']:.1f} us; engine bulk "
          f"{run('step4b_capped/runs/f05_bulk')['window']:.1f} us")

    # 1. Software overlap.
    F, eta, _, _ = m7.fit_sw()
    bw = K["f05"][0]
    print("\n1. Software overlap (2 GPUs): run | bands | measured | model phi 0 | error | "
          "model phi 0.25 | error")
    sw = {}
    for name, plan, bands in (("sw_s1", (1,) * 8, [(0, 4, 1), (4, 4, 1)]),
                              ("sw_plan", PLAN, [(0, 4, 4), (4, 2, 1), (6, 2, 2)])):
        r = run(os.path.join(RUNS, name))
        p0, p25 = (m7.sw_window(bands, phi, F, eta, bw) for phi in (0.0, 0.25))
        sw[name] = (r["window"], p0)
        print(f"{name} | {m7.rows_str(bands)} | {r['window']:.1f} ({'ok' if r['ok'] else 'FAIL'}) | "
              f"{p0:.1f} | {p0 / r['window'] - 1:+.1%} | {p25:.1f} | {p25 / r['window'] - 1:+.1%}")
    hw_gain = hw_plan["window"] - hw_s1["window"]
    for label, i in (("measured", 0), ("modelled (phi 0)", 1)):
        d = sw["sw_plan"][i] - sw["sw_s1"][i]
        print(f"plan vs s=1 in software, {label}: {d:+.1f} us "
              f"({sw['sw_plan'][i] / sw['sw_s1'][i] - 1:+.1%}); engine: {hw_gain:+.1f} us "
              f"({hw_plan['window'] / hw_s1['window'] - 1:+.1%}); share of the engine's gain "
              f"kept: {d / hw_gain:+.0%}")

    # 2-3. 4 and 8 GPUs.
    print("\n2-3. More GPUs: run | measured | verified | model | error")
    cm4 = m.Comm(4, "all-reduce", *K["f05_4gpu"], 4096)
    cm8 = m.Comm(8, "all-reduce", *K["f05_8gpu"], 4096, engine_per_link=True)
    end1 = g.run((1,) * 8)[1]
    rows = [
        ("x4 engine tile s=1 (Step 6)", "step6_paper/runs/f05_4gpu_tile_s1",
         cm4.window(g.run((1,) * 8)[0])),
        ("x4 engine tile plan (Step 6)", "step6_paper/runs/f05_4gpu_tile_plan",
         cm4.window(g.run(PLAN)[0])),
        ("x4 engine bulk", os.path.join(RUNS, "x4_bulk"), end1 + cm4.window([0.0] * 64)),
        ("x4 direct software", os.path.join(RUNS, "x4_direct"), None),
        ("x8 engine tile s=1", os.path.join(RUNS, "x8_tile_s1"), cm8.window(g.run((1,) * 8)[0])),
        ("x8 engine tile plan", os.path.join(RUNS, "x8_tile_plan"), cm8.window(g.run(PLAN)[0])),
        ("x8 direct software", os.path.join(RUNS, "x8_direct"), None),
    ]
    res = {}
    for label, path, model in rows:
        r = run(path)
        res[label] = r["window"]
        mod = f"{model:.1f} | {model / r['window'] - 1:+.1%}" if model else "- | -"
        print(f"{label} | {r['window']:.1f} | {r['ok']} | {mod}")
    print(f"x4: plan vs direct software {res['x4 engine tile plan (Step 6)'] / res['x4 direct software'] - 1:+.1%}, "
          f"s=1 vs direct {res['x4 engine tile s=1 (Step 6)'] / res['x4 direct software'] - 1:+.1%}, "
          f"bulk vs direct {res['x4 engine bulk'] / res['x4 direct software'] - 1:+.1%}")
    print(f"x8: plan vs s=1 {res['x8 engine tile plan'] / res['x8 engine tile s=1'] - 1:+.1%}, "
          f"plan vs direct software {res['x8 engine tile plan'] / res['x8 direct software'] - 1:+.1%}, "
          f"s=1 vs direct {res['x8 engine tile s=1'] / res['x8 direct software'] - 1:+.1%}")

    # 4. Rank test.
    cm2 = m.Comm(2, "all-reduce", *K["f05"], 4096)
    plans = [
        ("s=1", (1,) * 8, "step5_splitk/runs/f05_tile_s1"),
        ("Step 5 edges s=2", (2, 2, 1, 1, 1, 1, 2, 2), "step5_splitk/runs/f05_tile_s2"),
        ("Step 5 edges s=4", (4, 1, 1, 1, 1, 1, 1, 4), "step5_splitk/runs/f05_tile_s4"),
        ("chosen 4,4,4,4,1,1,2,2", PLAN, "step5b_bounds/runs/c_f05_plan"),
        ("2nd best distinct", (4, 4, 4, 4, 4, 4, 4, 1), os.path.join(RUNS, "rank_2nd")),
        ("10th best distinct", (1, 2, 4, 1, 1, 1, 1, 1), os.path.join(RUNS, "rank_10th")),
        ("median", (4, 1, 4, 4, 2, 1, 2, 4), os.path.join(RUNS, "rank_median")),
        ("worst", (1, 1, 1, 1, 4, 4, 4, 4), os.path.join(RUNS, "rank_worst")),
    ]
    allw = sorted(round(cm2.window(g.run(p)[0]), 1)
                  for p in itertools.product((1, 2, 4), repeat=8))
    print("\n4. Rank test (2 GPUs, f~0.5, cold): plan | measured | verified | model | error | "
          "model rank of 6561 (ties: best position)")
    meas, mod = [], []
    for label, plan, path in plans:
        r = run(path)
        w = round(cm2.window(g.run(plan)[0]), 1)
        meas.append(r["window"])
        mod.append(w)
        print(f"{label} {m.name(plan)} | {r['window']:.1f} | {r['ok']} | {w:.1f} | "
              f"{w / r['window'] - 1:+.1%} | {1 + sum(1 for x in allw if x < w)}")
    rho = spearman(mod, meas)
    best = min(range(len(plans)), key=lambda i: meas[i])
    chosen = 3
    print(f"Spearman rho (model vs measured, {len(plans)} plans, ties averaged) = {rho:.3f}")
    print(f"best measured: {plans[best][0]} {meas[best]:.1f} us; model's choice "
          f"{meas[chosen]:.1f} us, {meas[chosen] / meas[best] - 1:+.1%} from it")

    d = os.path.join(RUNS, "defaults_433", "stdout.txt")
    text = open(d).read()
    w = re.search(r"window=([\d.]+)", text)
    print(f"\ndefaults: window {float(w.group(1)) * 1e6:.3f} us, verified {'Passed!' in text}")


if __name__ == "__main__":
    main()
