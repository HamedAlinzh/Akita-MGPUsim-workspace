#!/usr/bin/env python3
"""Step 11 summary (results/step11_firstround/summary.txt): the plan's gain
split into the GEMM part (change in gemm_wait) and the overlap part (the
rest), as in Step 9, for 16 CUs by cache state and for 4, 8 and 16 CUs cold;
then the first-round trace analysis (step11_trace.py) and the model errors
(step11_model.py). Reads only the runs' log lines and traces."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step11_model as s11  # noqa: E402

OUT = s11.OUT
S9, S10, S11 = "step9_baselines/runs/", "step10_cushare/runs/", "step11_firstround/runs/"

BY_STATE = [  # point, s = 1 run, plan run
    ("16 CUs, A, B and C warm", S11 + "w16_s1", S11 + "w16_plan"),
    ("16 CUs, A and C warm (Step 10b)", S10 + "wab_s1", S10 + "wab_plan"),
    ("16 CUs, cold (Step 10)", S10 + "eng_s1", S10 + "eng_plan"),
    ("4 CUs, A, B and C warm (Step 9)", S9 + "warm_s1", S9 + "warm_plan"),
    ("4 CUs, A and C warm (Step 9)", S9 + "wab_s1", S9 + "wab_plan"),
    ("4 CUs, cold (Step 9)", S9 + "eng_f05_s1", S9 + "eng_f05_plan"),
]
BY_CUS = [
    (4, S9 + "eng_f05_s1", S9 + "eng_f05_plan"),
    (8, S11 + "c8_s1", S11 + "c8_plan"),
    (16, S10 + "eng_s1", S10 + "eng_plan"),
]


def split(s1, plan):
    a, b = s11.parse(s1), s11.parse(plan)
    if not (a and b and a["ok"] and b["ok"]):
        return None
    dw, dg = b["window"] - a["window"], b["gemm"] - a["gemm"]
    return a, b, dw, dg


def pct(x, base):
    return f"{x:+.1f} ({x / base:+.1%})"


def run_script(name):
    return subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), name)],
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()


def main():
    L = ["1. Gain split by cache state (us; parts as % of the s=1 window)",
         "point | s=1 window | plan window | change | GEMM s=1 / plan | GEMM part | overlap part"]
    for name, s1, plan in BY_STATE:
        r = split(s1, plan)
        if not r:
            L.append(f"{name} | (not run or not verified)")
            continue
        a, b, dw, dg = r
        L.append(f"{name} | {a['window']:.1f} | {b['window']:.1f} | {pct(dw, a['window'])} | "
                 f"{a['gemm']:.1f} / {b['gemm']:.1f} | {pct(dg, a['window'])} | "
                 f"{pct(dw - dg, a['window'])}")
    L.append("\n2. Gain split by CU count, cold (us; parts as % of the s=1 window)")
    L.append("CUs | shape | s=1 window | plan window | change | GEMM s=1 / plan | GEMM part | "
             "overlap part | GEMM share of the gain")
    for cus, s1, plan in BY_CUS:
        r = split(s1, plan)
        M, N = s11.POINTS[cus][:2]
        if not r:
            L.append(f"{cus} | {M}x{N} | (not run or not verified)")
            continue
        a, b, dw, dg = r
        L.append(f"{cus} | {M}x{N} | {a['window']:.1f} | {b['window']:.1f} | "
                 f"{pct(dw, a['window'])} | {a['gemm']:.1f} / {b['gemm']:.1f} | "
                 f"{pct(dg, a['window'])} | {pct(dw - dg, a['window'])} | {dg / dw:.0%}")
    L.append("\n3. First round vs later rounds (scripts/step11_trace.py, trace.txt)")
    L.extend("  " + x for x in run_script("step11_trace.py"))
    L.append("\n4. Model before and after (scripts/step11_model.py, model.txt)")
    L.extend("  " + x for x in run_script("step11_model.py"))
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
