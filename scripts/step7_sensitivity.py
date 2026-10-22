#!/usr/bin/env python3
"""Step 7.1: split-cost sensitivity of the MI300X projection (model only).
Repeats step6_model.py's projection with the 4-way split cost at 1.0-1.6x and
the 2-way at 1.0-1.2x, re-searching the best plan at each, then finds the
4-way cost (2-way fixed at 1.04x) and the 2-way cost (4-way at 1.15x) at which
the f~0.5 gain falls below 10%."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step6_model as m  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step7_baselines")
S4 = (1.0, 1.15, 1.3, 1.45, 1.6)
S2 = (1.0, 1.04, 1.1, 1.2)


def setup(layer):
    cols = layer.model.hidden // rm.REAL_TILE
    rows = 4096 // rm.REAL_TILE
    t_wg = layer.t_gemm() * 1e6 * rm.REAL_CUS / (rows * cols)
    cm = m.Comm(8, layer.collective, rm.REAL_LINK_BW, rm.REAL_WIRE_NS,
                rm.REAL_MSG_LATENCY_NS - rm.REAL_WIRE_NS, rm.REAL_TILE ** 2 * rm.BF16,
                header=0, eff=rm.REAL_LINK_EFF, calib=1.0, engine_per_link=True)
    return rows, cols, t_wg, cm


def lower_bound(cm, ready, p):
    """No window can end before the first tile plus every message on the
    busiest link, nor before the last tile plus one message."""
    per_link = len(ready) * (2 if cm.collective == "all-reduce" else 1) / p
    return max(min(ready) + per_link * cm.x, max(ready) + cm.x) + cm.tail


def gain(layer, c2, c4):
    rows, cols, t_wg, cm = setup(layer)
    g = m.Gemm(rows, cols, rm.REAL_CUS, {1: t_wg, 2: c2 * t_wg / 2, 4: c4 * t_wg / 4})
    w1 = cm.window(g.run((1,) * rows)[0])
    best = (w1, (1,) * rows)
    for plan in m.rows_plans_structured(rows):
        ready, _ = g.run(plan)
        if lower_bound(cm, ready, 8) >= best[0]:
            continue
        w = cm.window(ready)
        if w < best[0]:
            best = (w, plan)
    return best[0] / w1 - 1, best[1], w1, best[0]


def threshold(layer, vary, fixed, lo, hi, target=-0.10, steps=7):
    """Cost of the varied split at which the gain first rises above target
    (gain is negative; 'below 10%' means gain > -10%)."""
    f = (lambda c: gain(layer, fixed, c)[0]) if vary == 4 else (lambda c: gain(layer, c, fixed)[0])
    if f(hi) <= target:
        return None, f(hi)
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid) <= target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, None


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = []
    out = lines.append
    out("MI300X projection, split-cost sensitivity (model only; same assumptions as "
        "results/step6_paper/model.txt). Gain = best plan vs s=1, re-searched per cell.")
    layers = [(pt.label, pt.layer) for pt in rm.points(4096)]
    for label, layer in layers:
        out(f"\n{label} ({layer.model.name} {layer.gemm} {layer.collective}): rows = 2-way "
            "cost, columns = 4-way cost; cell = gain (best plan)")
        out("2-way \\ 4-way | " + " | ".join(f"{c:.2f}" for c in S4))
        for c2 in S2:
            cells = []
            for c4 in S4:
                g, plan, _, _ = gain(layer, c2, c4)
                cells.append(f"{g:+.1%} ({''.join(map(str, plan))})")
            out(f"{c2:.2f} | " + " | ".join(cells))
            print("\n".join(lines[-1:]), flush=True)

    f05 = layers[1][1]
    t4, g4 = threshold(f05, 4, 1.04, 1.0, 4.0)
    out(f"\nf~0.5 threshold, 2-way at 1.04x: " + (
        f"the gain falls below 10% at a 4-way cost of {t4:.2f}x" if t4 else
        f"the gain stays at or above 10% up to a 4-way cost of 4.0x ({g4:+.1%} there)"))
    t2, g2 = threshold(f05, 2, 1.15, 1.0, 2.0)
    out(f"f~0.5 threshold, 4-way at 1.15x: " + (
        f"the gain falls below 10% at a 2-way cost of {t2:.2f}x" if t2 else
        f"the gain stays at or above 10% up to a 2-way cost of 2.0x ({g2:+.1%} there)"))
    for c in (1.3, 1.6, 2.0, 3.0):
        out(f"f~0.5, both splits at {c}x (2-way and 4-way): gain {gain(f05, c, c)[0]:+.1%}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "sensitivity.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
