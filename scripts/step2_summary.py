#!/usr/bin/env python3
"""Summarize Step 2 runs: verification, window, and engine message counts
against the expected values (per direction: 2*tiles/(N*p) messages for an
all-reduce, tiles/(N*p) for a reduce-scatter; traffic bytes = messages x
(4096*N + 16)).

Usage: python3 scripts/step2_summary.py results/step2_engine/correctness
"""
import os
import re
import sys

TILE_BYTES, HEADER = 32 * 32 * 4, 16


def parse(path):
    text = open(os.path.join(path, "stdout.txt")).read()
    cmd = text.splitlines()[0] if text else ""
    r = {"name": os.path.basename(path.rstrip("/")),
         "verified": "Passed!" in text}
    m = re.search(r"phases: window=([\d.]+) gemm_wait=([\d.]+) (?:comm_tail|allreduce)=([\d.]+)", text)
    if m:
        r["window_us"], r["gemm_us"], r["after_gemm_us"] = (float(x) * 1e6 for x in m.groups())
    r["dirs"] = {(int(a), int(b)): (int(c), int(d)) for a, b, c, d in re.findall(
        r"messages: from=(\d+) to=(\d+) count=(\d+) traffic_bytes=(\d+)", text)}
    w = re.search(r"wall_s=([\d.]+)", text)
    r["wall_s"] = float(w.group(1)) if w else None
    return r


def expected(name, dims):
    m, n = dims
    tiles = (m // 32) * (n // 32)
    N = int(re.search(r"_n(\d+)", name).group(1))
    p = 4 if "4gpu" in name else 2
    per = tiles // (N * p) * (1 if name.startswith("rs") else 2)
    return per, per * (TILE_BYTES * N + HEADER)


def main():
    root = sys.argv[1]
    dims = (384, 256)
    print("run | verified | window_us | gemm_us | after_gemm_us | msgs/dir (expected) | bytes/dir (expected) | counts ok | wall_s")
    for d in sorted(os.listdir(root)):
        r = parse(os.path.join(root, d))
        exp_n, exp_b = expected(d, dims)
        counts = sorted(set(r["dirs"].values()))
        ok = bool(r["dirs"]) and all(v == (exp_n, exp_b) for v in r["dirs"].values())
        print(f"{d} | {r['verified']} | {r.get('window_us', 0):.1f} | {r.get('gemm_us', 0):.1f} | "
              f"{r.get('after_gemm_us', 0):.1f} | {','.join(str(c[0]) for c in counts)} ({exp_n}) | "
              f"{','.join(str(c[1]) for c in counts)} ({exp_b}) | {ok} | {r['wall_s']}")


if __name__ == "__main__":
    main()
