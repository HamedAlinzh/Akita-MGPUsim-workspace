#!/usr/bin/env python3
"""Capped-baseline summary: three versions per point, ramp on top, and
small-tiles-first, with the ideal bound max(GEMM, comm) + round trip and the
first-ready bound max(GEMM, first ready + comm) + round trip. comm is the
messages per direction x max(transfer, overhead), transfer = traffic /
(0.8 x link bandwidth) as calibrated in Step 2; the round trip is
2 x 1.07 x (overhead + latency) + 2 x transfer.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "step4b_capped", "runs")
KNOBS = {"f05": (0.7807e9, 441e-9, 2121e-9), "f08": (0.223e9, 1542e-9, 7424e-9)}
TILES = 64  # 256x256 at tileSize 8


def parse(name):
    try:
        text = open(os.path.join(RUNS, name, "stdout.txt")).read()
    except FileNotFoundError:
        return None
    m = re.search(r"window=([\d.]+) gemm_wait=([\d.]+) \w+=([\d.]+)(?: first_ready=([\d.]+))?", text)
    if not m:
        return dict(ok=False)
    msgs = sorted(set(int(c) for c in re.findall(r"count=(\d+)", text)))
    return dict(ok="Passed!" in text, window=float(m.group(1)) * 1e6,
                gemm=float(m.group(2)) * 1e6,
                first=float(m.group(4)) * 1e6 if m.group(4) else None, msgs=msgs)


def bounds(p, gemm, first):
    bw, lat, ovh = KNOBS[p]
    x = (4096 + 16) / (0.8 * bw)
    comm = TILES * max(x, ovh) * 1e6
    rt = (2 * 1.07 * (ovh + lat) + 2 * x) * 1e6
    return max(gemm, comm) + rt, max(gemm, first + comm) + rt, comm, rt


def main():
    print("point | version | verified | first_ready_us | gemm_us | window_us | vs_capped_tile | msgs/dir")
    for p in ("f05", "f08"):
        tile = parse(f"{p}_tile")
        bulk = parse(f"{p}_bulk")
        for v in ("mccl", "bulk", "tile", "tile_ramp1", "small_r32", "small_r64", "small_r128"):
            r = parse(f"{p}_{v}")
            if not r or "window" not in r:
                print(f"{p} | {v} | missing")
                continue
            rel = r["window"] / tile["window"] - 1 if tile else float("nan")
            first = f"{r['first']:.1f}" if r["first"] else "-"
            print(f"{p} | {v} | {r['ok']} | {first} | {r['gemm']:.1f} | {r['window']:.1f} | "
                  f"{rel:+.1%} | {r['msgs']}")
        if tile and bulk:
            ideal, fbound, comm, rt = bounds(p, tile["gemm"], tile["first"])
            mccl = parse(f"{p}_mccl")
            print(f"{p} | off-CU gain {mccl['window'] / bulk['window']:.2f}, overlap gain "
                  f"{bulk['window'] / tile['window']:.2f}; comm {comm:.0f}, round trip {rt:.1f}, "
                  f"ideal {ideal:.0f}, first-ready bound {fbound:.0f}")

    print("\nsmall-tiles-first GEMM alone (1 GPU) | R | verified | first_ready_us | gemm_us")
    for r in (32, 64, 128):
        g = parse(f"gemm_small_r{r}")
        if g and "window" in g:
            print(f"| {r} | {g['ok']} | {g['first']:.1f} | {g['gemm']:.1f}")


if __name__ == "__main__":
    main()
