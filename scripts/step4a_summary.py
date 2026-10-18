#!/usr/bin/env python3
"""Dispatch-shaping spike summary.

Item 1: per representative point at tileSize 8, the first-ready bound
max(GEMM, first ready + comm) + round trip against the ideal
max(GEMM, comm) + round trip; their difference is all that moving the first
tile earlier can win. comm and the round trip come from the Step 2
calibration (stream time per message, single-message latency).

Item 3: 2-GPU tile-mode windows with shaping, and the share of the item-1
gap each closes.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "step4a_shaping", "runs")

# Calibration (results/step2_engine/calibration.csv): stream ns per message,
# single-message latency ns.
CAL = {"f08": (28552, 40302), "f05": (8157, 11511), "f025": (4485, 6313)}
# GEMM alone and first ready (us), measured: tileSize 8, 384x256 (2 GPUs)
# or 512x320 (1-GPU tracker, first completion after the launch).
POINTS = {
    "f08": dict(gemm=647.5, first=281.0, msgs=96, rt_msgs=2),
    "f05": dict(gemm=647.5, first=281.0, msgs=96, rt_msgs=2),
    "f025": dict(gemm=1084.3, first=302.0, msgs=80, rt_msgs=1),
}


def bounds(p, first=None, gemm=None):
    q = POINTS[p]
    per, single = CAL[p]
    comm = q["msgs"] * per / 1e3
    rt = q["rt_msgs"] * single / 1e3
    first = q["first"] if first is None else first
    gemm = q["gemm"] if gemm is None else gemm
    return max(gemm, first + comm) + rt, max(gemm, comm) + rt, comm, rt


def parse(name):
    try:
        text = open(os.path.join(RUNS, name, "stdout.txt")).read()
    except FileNotFoundError:
        return None
    m = re.search(r"window=([\d.]+) gemm_wait=([\d.]+) \w+=([\d.]+)(?: first_ready=([\d.]+))?", text)
    if not m:
        return None
    return dict(ok="Passed!" in text, window=float(m.group(1)) * 1e6,
                gemm=float(m.group(2)) * 1e6,
                first=float(m.group(4)) * 1e6 if m.group(4) else None)


def main():
    print("1. potential | point | first_ready_bound_us | ideal_us | gap_us | gap_share")
    for p in ("f08", "f05", "f025"):
        fb, ideal, comm, rt = bounds(p)
        print(f"| {p} | {fb:.0f} | {ideal:.0f} | {fb - ideal:.0f} | {(fb - ideal) / fb:.0%}")

    base = {"f05": 1074.8, "f08": None}
    b08 = parse("f08_tile_none")
    if b08:
        base["f08"] = b08["window"]
    print("\n3. shaping | point | setting | verified | first_ready_us | gemm_drain_us | window_us | "
          "baseline_us | change | gap_closed")
    for p in ("f05", "f08"):
        fb, ideal, _, _ = bounds(p)
        for s in ("none", "ramp1", "cap4"):
            if p == "f05" and s == "none":  # Step 3's own_t8_tile_n1
                r = dict(ok=True, window=1074.8, gemm=647.9, first=281.0)
            else:
                r = parse(f"{p}_tile_{s}")
            if not r or base[p] is None:
                continue
            closed = (base[p] - r["window"]) / (fb - ideal)
            print(f"| {p} | {s} | {r['ok']} | {r['first']:.1f} | {r['gemm']:.1f} | {r['window']:.1f} | "
                  f"{base[p]:.1f} | {r['window'] / base[p] - 1:+.1%} | {closed:.0%}")

    print("\n4/5. other runs | run | verified | window_us | gemm_drain_us | first_ready_us")
    for n in ("mesh4_mccl", "mesh4_bulk", "mesh4_tile", "f08_tile_t4_mc", "defaults_433"):
        r = parse(n)
        if r:
            print(f"| {n} | {r['ok']} | {r['window']:.1f} | {r['gemm']:.1f} | "
                  f"{r['first'] if r['first'] else float('nan'):.1f}")


if __name__ == "__main__":
    main()
