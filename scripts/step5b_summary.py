#!/usr/bin/env python3
"""Step 5b summary: B (warm vs cold rounds) and C (the per-row split plan),
with the C runs' queue-model errors: from the work-group model alone
(step5b_model.py, predicted before the runs) and from the run's own
work-group trace (tile ready times measured, comm queued as in Step 5)."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402
import step5_wgs  # noqa: E402
import step5b_model as model  # noqa: E402

RUNS = os.path.join(s5.ROOT, "results", "step5b_bounds", "runs")
PLAN = (4, 4, 4, 4, 1, 1, 2, 2)
REF = {"f05": (578.0, 632.4), "f08": (1585.7, 1695.4)}


def plan_ready(csv_path, plan):
    """Tile ready times (absolute us) for a -splitk-plan run: launches per GPU
    in launch order map to the plan's bands (split rows: s launches each;
    consecutive unsplit rows: one launch)."""
    rows = step5_wgs.load(csv_path)
    bands = []
    for row, s in enumerate(plan):
        if s == 1 and bands and bands[-1][1] == 1:
            continue
        bands.extend([(row, s)] * s)
    launches = {}
    for r in rows:
        launches.setdefault(r["gpu"], set()).add(int(r["kernel_id"].split("@")[0]))
    ready = [0.0] * model.TILES
    for r in rows:
        i = sorted(launches[r["gpu"]]).index(int(r["kernel_id"].split("@")[0]))
        row0 = bands[i][0]
        tile = (row0 + int(r["wg_y"])) * model.COLS + int(r["wg_x"])
        ready[tile] = max(ready[tile], r["end"])
    return ready


def main():
    print("B: warm vs cold (1 GPU, GEMM alone; cold = the -warmup launch)")
    sys.stdout.flush()
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "step5b_rounds.py"),
                    os.path.join(RUNS, "warm_plain"), os.path.join(RUNS, "warm_s2")], check=True)

    print("\nC: plan 4,4,4,4,1,1,2,2 (2 GPUs, tile mode, pre-sum)")
    print("point | verified | first_ready_us | gemm_us | window_us | vs step5 best | vs s=1 | "
          "wg_model (kept/removed) | error | trace queue model | error")
    for p in ("f05", "f08"):
        r = s5.parse(os.path.join(RUNS, f"c_{p}_plan"))
        if not r or "window" not in r:
            print(f"{p}: missing")
            continue
        kept = model.window(p, model.gemm(PLAN, True)[0])
        removed = model.window(p, model.gemm(PLAN, False)[0])
        ready = plan_ready(os.path.join(RUNS, f"c_{p}_plan", "wgs.csv"), PLAN)
        q = s5.queue_model(p, ready, r["first"], [s5.per_msg(p)[0]] * model.TILES)
        best, s1 = REF[p]
        print(f"{p} | {r['ok']} | {r['first']:.1f} | {r['gemm']:.1f} | {r['window']:.1f} | "
              f"{r['window'] / best - 1:+.1%} | {r['window'] / s1 - 1:+.1%} | "
              f"{kept:.1f}/{removed:.1f} | {kept / r['window'] - 1:+.1%}/{removed / r['window'] - 1:+.1%} | "
              f"{q:.1f} | {q / r['window'] - 1:+.1%}")

    g = s5.parse(os.path.join(RUNS, "repro_gemm_s2"))
    if g:
        print(f"\nrepro gemm_s2: gemm {g['gemm']:.3f} us (Step 5: 334.490), first ready "
              f"{g['first']:.3f} (118.096), verified {g['ok']}")
    d = os.path.join(RUNS, "defaults_433", "stdout.txt")
    if os.path.exists(d):
        import re
        m = re.search(r"window=([\d.]+)", open(d).read())
        print(f"defaults: window {float(m.group(1)) * 1e6:.3f} us, verified {'Passed!' in open(d).read()}")


if __name__ == "__main__":
    main()
