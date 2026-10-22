#!/usr/bin/env python3
"""Step 6 summary: gains of the per-row split plan over s = 1 and over the
engine in bulk mode (the off-CU-only baseline), at every point, cold and warm
where measured; then the model's output (searches, model check, MI300X
projection) from step6_model.py.

Runs from earlier steps are reused where they already exist: f~0.5 cold s = 1
(step5_splitk f05_tile_s1) and plan (step5b_bounds c_f05_plan), f~0.8 plan
(step5b_bounds c_f08_plan), and the bulk and f~0.8 s = 1 runs of
step4b_capped (one GEMM kernel per GPU, so ordered dispatch and launch do not
change them: f05_tile_s1 with them is 632.4 us, as without)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402

R = os.path.join(s5.ROOT, "results")


def run(path):
    return s5.parse(os.path.join(R, path))


def main():
    rows = [
        ("f~0.25", "cold", "2 GPUs, reduce-scatter", "step6_paper/runs/f025_tile_s1",
         "step6_paper/runs/f025_tile_plan", "step6_paper/runs/f025_bulk_s1"),
        ("f~0.5", "cold", "2 GPUs", "step5_splitk/runs/f05_tile_s1",
         "step5b_bounds/runs/c_f05_plan", "step4b_capped/runs/f05_bulk"),
        ("f~0.5", "warm", "2 GPUs", "step6_paper/runs/warm_f05_s1",
         "step6_paper/runs/warm_f05_plan", None),
        ("f~0.5", "cold", "4 GPUs, per-pair mesh", "step6_paper/runs/f05_4gpu_tile_s1",
         "step6_paper/runs/f05_4gpu_tile_plan", None),
        ("f~0.8", "cold", "2 GPUs", "step4b_capped/runs/f08_tile",
         "step5b_bounds/runs/c_f08_plan", "step4b_capped/runs/f08_bulk"),
    ]
    print("point | caches | setup | verified | s=1 window | plan window | gain vs s=1 | "
          "bulk window | gain vs bulk | s=1 gemm | plan gemm | s=1 first ready | plan first ready")
    for point, caches, setup, s1p, planp, bulkp in rows:
        s1, plan = run(s1p), run(planp)
        bulk = run(bulkp) if bulkp else None
        if not s1 or not plan or "window" not in s1 or "window" not in plan:
            print(f"{point} | {caches} | {setup} | missing")
            continue
        ok = s1["ok"] and plan["ok"] and (bulk["ok"] if bulk else True)
        b = (f"{bulk['window']:.1f} | {plan['window'] / bulk['window'] - 1:+.1%}"
             if bulk else "- | -")
        print(f"{point} | {caches} | {setup} | {ok} | {s1['window']:.1f} | {plan['window']:.1f} | "
              f"{plan['window'] / s1['window'] - 1:+.1%} | {b} | {s1['gemm']:.1f} | "
              f"{plan['gemm']:.1f} | {s1['first']:.1f} | {plan['first']:.1f}")

    d = os.path.join(R, "step6_paper", "runs", "defaults_433", "stdout.txt")
    if os.path.exists(d):
        import re
        text = open(d).read()
        m = re.search(r"window=([\d.]+)", text)
        print(f"\ndefaults: window {float(m.group(1)) * 1e6:.3f} us, verified {'Passed!' in text}")
    print()
    sys.stdout.flush()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "step6_model.py")],
                   check=True)


if __name__ == "__main__":
    main()
