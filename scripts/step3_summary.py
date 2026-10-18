#!/usr/bin/env python3
"""Summarize Step 3 runs (results/step3_tiles/runs, plus the Step 2 runs that
serve as tileSize 8 for both curves).

Per run: GEMM time alone (the mccl run's GEMM drain, nothing else running),
first-tile ready time, window. Per tile size and curve: off-CU gain (mccl ring
window / engine bulk window) and overlap gain (bulk window / tile window), and
the readiness bound: first ready + comm + one tile round trip, with
comm = messages per direction x max(transfer, -msg-overhead) and round trip =
2 x (overhead + latency) x 1.07 (measured excess) + 2 x transfer.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "step3_tiles", "runs")
STEP2 = os.path.join(ROOT, "results", "step2_engine")
BW, LAT, OVH = 0.6325e9, 544e-9, 2618e-9
M, N_COLS = 384, 256


def parse(path):
    try:
        text = open(os.path.join(path, "stdout.txt")).read()
    except FileNotFoundError:
        return None
    r = {"ok": "Passed!" in text}
    m = re.search(r"phases: window=([\d.]+) gemm_wait=([\d.]+) \w+=([\d.]+)"
                  r"(?: first_ready=([\d.]+))?", text)
    if m:
        r["window"] = float(m.group(1)) * 1e6
        r["gemm_wait"] = float(m.group(2)) * 1e6
        r["first_ready"] = float(m.group(4)) * 1e6 if m.group(4) else None
    counts = re.findall(r"messages: from=\d+ to=\d+ count=(\d+)", text)
    r["msgs"] = sorted(set(int(c) for c in counts))
    return r


def run(name):
    for d in (RUNS, os.path.join(STEP2, "correctness"), os.path.join(STEP2, "timing")):
        r = parse(os.path.join(d, name))
        if r:
            return r
    return None


def bound(ts, n, first_ready):
    payload = (4 * ts) ** 2 * 4 * n
    x = (payload + 16) / (BW * 0.8)
    msgs = (M // (4 * ts)) * (N_COLS // (4 * ts)) // n
    comm = msgs * max(x, OVH)
    rt = 2 * (OVH + LAT) * 1.07 + 2 * x
    return (first_ready + (comm + rt) * 1e6), comm * 1e6


def names(ts, curve):
    if ts == 8:  # identical under both curves; Step 2 runs
        return dict(mccl="mccl_ring_ar_f05", bulk="ar_bulk_n{n}", tile="own_t8_tile_n{n}",
                    nm="nm_t8_tile_n1")
    return dict(mccl=f"mccl_t4_{curve}", bulk=f"own_t4_{curve}_bulk_n{{n}}",
                tile=f"own_t4_{curve}_tile_n{{n}}", nm=f"nm_t4_{curve}_tile_n1")


def main():
    print("tile | curve | N | verified | gemm_alone_us | first_ready_us | tile_window_us | "
          "bulk_window_us | mccl_window_us | offCU_gain | overlap_gain | bound_us | msgs/dir")
    for ts in (4, 8):
        for curve in ("native", "mc"):
            nm = names(ts, curve)
            mccl = run(nm["mccl"])
            bulk1 = run(nm["bulk"].format(n=1))
            for n in (1, 2, 4):
                t = run(nm["tile"].format(n=n))
                b = run(nm["bulk"].format(n=n)) or bulk1
                if not t or "window" not in t:
                    continue
                gemm = mccl["gemm_wait"] if mccl and "gemm_wait" in mccl else float("nan")
                bnd, _ = bound(ts, n, t["first_ready"])
                off = mccl["window"] / bulk1["window"] if mccl and bulk1 else float("nan")
                ov = b["window"] / t["window"] if b else float("nan")
                print(f"{ts} | {curve} | {n} | {t['ok']} | {gemm:.1f} | {t['first_ready']:.1f} | "
                      f"{t['window']:.1f} | {b['window'] if b else float('nan'):.1f}"
                      f"{'' if run(nm['bulk'].format(n=n)) else '*'} | "
                      f"{mccl['window'] if mccl else float('nan'):.1f} | {off:.2f} | {ov:.2f} | "
                      f"{bnd:.1f} | {t['msgs']}")
    print("* bulk at N=1 (no bulk run at this N)")

    print("\nplacement at N=1 | tile | curve | owner_window_us | near_memory_window_us | verified")
    for ts in (4, 8):
        for curve in ("native", "mc"):
            nm = names(ts, curve)
            o, q = run(nm["tile"].format(n=1)), run(nm["nm"])
            if o and q and "window" in q:
                print(f"| {ts} | {curve} | {o['window']:.1f} | {q['window']:.1f} | {q['ok']}")
            if ts == 8:
                break


if __name__ == "__main__":
    main()
