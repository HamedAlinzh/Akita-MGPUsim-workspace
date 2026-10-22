#!/usr/bin/env python3
"""Step 9 item 2: split each plan-vs-s=1 window change into the GEMM's own
change (the Stream-K effect: the split GEMM is shorter or longer) and the
rest (better overlap).

GEMM time is each run's gemm_wait (launch to the last GEMM queue drained).
With the engines it runs beside the GEMM, so the bulk-mode GEMM-alone runs
(engines start after the GEMM) check how much the overlap itself slows the
GEMM: gemm_plan / gemm_plan_warm against the plan runs' gemm_wait, and
step4b f05_bulk / step5b warm_plain at s = 1 (one kernel, which the L1 fix
does not change). For the FlashOverlap-style runs the GEMM shares the GPU
with the comm kernels and has 3 of 4 CUs. Pairs marked with a step number
ran before the L1 fix.

Prints one table to stdout and results/step9_baselines/split.txt.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402

R = os.path.join(m.ROOT, "results")
OUT = os.path.join(R, "step9_baselines")

S9 = "step9_baselines/runs/"
PAIRS = [  # point, s = 1 run, plan run; Step 9 runs (after the L1 fix) first
    ("f0.5, 2 GPUs, cold", S9 + "eng_f05_s1", S9 + "eng_f05_plan"),
    ("f0.5, 2 GPUs, A+C warm", S9 + "wab_s1", S9 + "wab_plan"),
    ("f0.5, 2 GPUs, warm", S9 + "warm_s1", S9 + "warm_plan"),
    ("f0.8, 2 GPUs", S9 + "eng_f08_s1", S9 + "eng_f08_plan"),
    ("f0.5, 8 GPUs", S9 + "eng_x8_s1", S9 + "eng_x8_plan"),
    ("FlashOverlap-style, f0.5", S9 + "fo_f05_s1", S9 + "fo_f05_plan"),
    ("FlashOverlap-style, f0.8", S9 + "fo_f08_s1", S9 + "fo_f08_plan"),
    ("FlashOverlap-style, 8 GPUs", S9 + "fo_x8_s1", S9 + "fo_x8_plan"),
    ("f0.5, 4 GPUs (Step 8)", "step8_edges/runs/x4_s1_pq", "step8_edges/runs/x4_plan_pq"),
    ("f0.25 (512x320, RS; Step 6)", "step6_paper/runs/f025_tile_s1", "step6_paper/runs/f025_tile_plan"),
    ("f0.35 (Step 8)", "step8_edges/runs/f035_s1", "step8_edges/runs/f035_plan"),
    ("f0.65 (Step 8)", "step8_edges/runs/f065_s1", "step8_edges/runs/f065_plan"),
    ("f0.9 (Step 8)", "step8_edges/runs/f090_s1", "step8_edges/runs/f090_plan"),
    ("1.5 waves, 192x256 (Step 8)", "step8_edges/runs/w15_s1", "step8_edges/runs/w15_plan"),
    ("3.4 waves, 384x288 (Step 8)", "step8_edges/runs/w337_s1", "step8_edges/runs/w337_plan"),
    ("9 tile rows (Step 8)", "step8_edges/runs/r9_s1", "step8_edges/runs/r9_plan"),
    ("K/GPU 128 (Step 8)", "step8_edges/runs/k128_s1", "step8_edges/runs/k128_plan"),
    ("K/GPU 512 (Step 8)", "step8_edges/runs/k512_s1", "step8_edges/runs/k512_plan"),
    ("RS f0.55 (Step 8)", "step8_edges/runs/rs05_s1", "step8_edges/runs/rs05_plan"),
    ("RS f0.81 (Step 8)", "step8_edges/runs/rs08_s1", "step8_edges/runs/rs08_plan"),
]

CHECKS = [  # GEMM alone (bulk) vs the same GEMM beside the engines
    ("s=1 cold", "step4b_capped/runs/f05_bulk", S9 + "eng_f05_s1"),
    ("plan cold", S9 + "gemm_plan", S9 + "eng_f05_plan"),
    ("s=1 warm", "step5b_bounds/runs/warm_plain", S9 + "warm_s1"),
    ("plan warm", S9 + "gemm_plan_warm", S9 + "warm_plan"),
]


def parse(run):
    path = os.path.join(R, run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    mm = re.search(r" window=([\d.]+) gemm_wait=([\d.]+)", text)
    if not mm or "Passed!" not in text:
        return None
    return float(mm.group(1)) * 1e6, float(mm.group(2)) * 1e6


def main():
    lines = ["GEMM alone (bulk) vs beside the engines (gemm_wait, us):"]
    for name, alone, beside in CHECKS:
        a, b = parse(alone), parse(beside)
        if a and b:
            lines.append(f"  {name}: alone {a[1]:.1f}, beside {b[1]:.1f} ({b[1] / a[1] - 1:+.1%})")
    lines.append("\npoint | s=1 window | plan window | change | GEMM s=1 / plan | GEMM part | "
                 "overlap part (us, % of s=1 window)")
    for name, s1, plan in PAIRS:
        a, b = parse(s1), parse(plan)
        if not (a and b):
            continue
        dw, dg = b[0] - a[0], b[1] - a[1]
        lines.append(f"{name} | {a[0]:.1f} | {b[0]:.1f} | {dw:+.1f} ({dw / a[0]:+.1%}) | "
                     f"{a[1]:.1f} / {b[1]:.1f} | {dg:+.1f} ({dg / a[0]:+.1%}) | "
                     f"{dw - dg:+.1f} ({(dw - dg) / a[0]:+.1%})")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "split.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
