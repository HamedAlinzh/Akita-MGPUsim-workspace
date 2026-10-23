#!/usr/bin/env python3
"""Step 8 summary (results/step8_edges/summary.txt): item 0 (4 GPUs with
per-peer engine queues, the model fix), item 1 (software baseline: see
mccl_breakdown.txt and sw_sweep.txt), item 2 (f sweep), and items 3-4 (shapes
and reduce-scatter, from step8_batchb) once their runs exist. Predictions use
step8_model (event messages); f sweep and batch B lines come from their own
scripts, which print the run results next to the model."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

ROOT = m.ROOT
OUT = os.path.join(ROOT, "results", "step8_edges")
RUNS = os.path.join(OUT, "runs")
PLAN = (4, 4, 4, 4, 1, 1, 2, 2)


def sh(script):
    sys.stdout.flush()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), script)],
                   check=True, stdout=sys.stdout)


def main():
    print("0a. 4 GPUs, f~0.5, engine tile with -engine-peer-queues (Step 6: one queue)")
    g = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    cm = e.knob_comm("f05_4gpu", 4, per_peer=True)
    step6 = {"x4_s1_pq": 669.0, "x4_plan_pq": 559.2}
    got = {}
    for name, plan in (("x4_s1_pq", (1,) * 8), ("x4_plan_pq", PLAN)):
        r = s5.parse(os.path.join(RUNS, name))
        w = cm.window(g.run(plan)[0])
        got[name] = r["window"]
        print(f"{name} | {r['window']:.1f} us (verified {r['ok']}) | Step 6 {step6[name]} us, "
              f"{r['window'] / step6[name] - 1:+.1%} | model {w:.1f} ({w / r['window'] - 1:+.1%})")
    print(f"plan vs s=1 with per-peer queues: {got['x4_plan_pq'] / got['x4_s1_pq'] - 1:+.1%} "
          f"(Step 6: {559.2 / 669.0 - 1:+.1%})")

    print("\n0b. Model fix: event-driven messages instead of queue + round-trip tail "
          "(22 past engine runs)")
    sh("step8_validate.py")

    print("\n1a. One mccl call, from step7_baselines sw_s1's trace")
    print(open(os.path.join(OUT, "mccl_breakdown.txt")).read().strip())
    print("\n1c. Software-overlap sweep (model)")
    print(open(os.path.join(OUT, "sw_sweep.txt")).read().strip())

    print("\n2. f sweep (2 GPUs)")
    sh("step8_fsweep.py")
    print(open(os.path.join(OUT, "fsweep.txt")).read().strip())

    if os.path.exists(os.path.join(RUNS, "rs08_plan", "stdout.txt")):
        print("\n3-4. Shapes, K, odd row count, reduce-scatter")
        sh("step8_batchb.py")
        d = s5.parse(os.path.join(RUNS, "defaults_433"))
        if d:
            text = open(os.path.join(RUNS, "defaults_433", "stdout.txt")).read()
            import re
            w = re.search(r"window=([\d.]+)", text)
            print(f"\ndefaults: window {float(w.group(1)) * 1e6:.3f} us, verified {'Passed!' in text}")


if __name__ == "__main__":
    main()
