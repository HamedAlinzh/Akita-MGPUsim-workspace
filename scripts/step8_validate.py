#!/usr/bin/env python3
"""Step 8 item 0b: the event-driven message model (step8_model) against the
Step 6 queue model on every engine run of Steps 5-7 (a priori work-group
times, cold unless warm). Prints one line per run and the error ranges."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

R = os.path.join(m.ROOT, "results")


def cases():
    K = m.knobs()
    cold = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    warm = m.Gemm(8, 8, 32, m.SIM_STEADY, None, m.SIM_LAUNCH)
    sc = 138.5 / m.T
    f1 = dict(m.SIM_FIRST)
    f1[1] = 167.2
    g512 = m.Gemm(16, 10, 32, {s: v * sc for s, v in m.SIM_STEADY.items()}, f1, m.SIM_LAUNCH)
    s1, s2, s4 = (1,) * 8, (2, 2, 1, 1, 1, 1, 2, 2), (4, 1, 1, 1, 1, 1, 1, 4)
    P = (4, 4, 4, 4, 1, 1, 2, 2)
    old = lambda k, p=2, c="all-reduce", pl=False: m.Comm(p, c, *K[k], 4096, engine_per_link=pl)
    new = lambda k, p=2, c="all-reduce", pl=False: e.Comm(p, c, *K[k], 4096, per_peer=pl)
    rs = "reduce-scatter"
    return [
        ("step5_splitk/runs/f05_tile_s1", cold, s1, ("f05",), False),
        ("step5_splitk/runs/f05_tile_s2", cold, s2, ("f05",), False),
        ("step5_splitk/runs/f05_tile_s4", cold, s4, ("f05",), False),
        ("step5_splitk/runs/f08_tile_s2", cold, s2, ("f08",), False),
        ("step5_splitk/runs/f08_tile_s4", cold, s4, ("f08",), False),
        ("step5_splitk/runs/f05_bulk_s2", cold, s2, ("f05",), True),
        ("step5b_bounds/runs/c_f05_plan", cold, P, ("f05",), False),
        ("step5b_bounds/runs/c_f08_plan", cold, P, ("f08",), False),
        ("step6_paper/runs/warm_f05_s1", warm, s1, ("f05",), False),
        ("step6_paper/runs/warm_f05_plan", warm, P, ("f05",), False),
        ("step6_paper/runs/f025_tile_s1", g512, (1,) * 16, ("f025", 2, rs), False),
        ("step6_paper/runs/f025_tile_plan", g512, (2, 2) + (1,) * 12 + (2, 2), ("f025", 2, rs), False),
        ("step6_paper/runs/f025_bulk_s1", g512, (1,) * 16, ("f025", 2, rs), True),
        ("step6_paper/runs/f05_4gpu_tile_s1", cold, s1, ("f05_4gpu", 4), False),
        ("step6_paper/runs/f05_4gpu_tile_plan", cold, P, ("f05_4gpu", 4), False),
        ("step7_baselines/runs/x4_bulk", cold, s1, ("f05_4gpu", 4), True),
        ("step7_baselines/runs/x8_tile_s1", cold, s1, ("f05_8gpu", 8, "all-reduce", True), False),
        ("step7_baselines/runs/x8_tile_plan", cold, P, ("f05_8gpu", 8, "all-reduce", True), False),
        ("step7_baselines/runs/rank_2nd", cold, (4, 4, 4, 4, 4, 4, 4, 1), ("f05",), False),
        ("step7_baselines/runs/rank_10th", cold, (1, 2, 4, 1, 1, 1, 1, 1), ("f05",), False),
        ("step7_baselines/runs/rank_median", cold, (4, 1, 4, 4, 2, 1, 2, 4), ("f05",), False),
        ("step7_baselines/runs/rank_worst", cold, (1, 1, 1, 1, 4, 4, 4, 4), ("f05",), False),
    ], old, new


def main():
    rows, old, new = cases()
    eo, en = [], []
    print("run | measured | queue model (Step 6) | error | event model | error")
    for path, g, plan, args, bulk in rows:
        r = s5.parse(os.path.join(R, path))
        ready, end = g.run(plan)
        if bulk:
            wo = end + old(*args).window([0.0] * len(ready))
        else:
            wo = old(*args).window(ready)
        wn = new(*args).window(ready, bulk_at=end if bulk else None)
        eo.append(wo / r["window"] - 1)
        en.append(wn / r["window"] - 1)
        print(f"{path.split('/')[-1]} | {r['window']:.1f} | {wo:.1f} | {eo[-1]:+.1%} | "
              f"{wn:.1f} | {en[-1]:+.1%}")
    mean = lambda xs: sum(abs(x) for x in xs) / len(xs)
    print(f"queue model: error {min(eo):+.1%} to {max(eo):+.1%}, mean |error| {mean(eo):.1%}")
    print(f"event model: error {min(en):+.1%} to {max(en):+.1%}, mean |error| {mean(en):.1%}")


if __name__ == "__main__":
    main()
