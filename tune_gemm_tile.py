#!/usr/bin/env python3
"""
Tune the GEMM work-group tile in HamedAlinzh/Akita-MGPUsim-workspace.

What this tunes
---------------
The OpenCL kernel computes a fixed 4x4 output micro-tile per work-item.
This script sweeps the square local work-group edge ("tileSize" in
benchmarks/allreducegemm/benchmark.go):

    tileSize=4   ->  4x4 work-items   -> 16x16 output tile
    tileSize=8   ->  8x8 work-items   -> 32x32 output tile (current default)
    tileSize=16  -> 16x16 work-items  -> 64x64 output tile

It runs the full K-split GEMM + AllReduce benchmark and ranks candidates by
simulated end-to-end total_time. This is deliberately the right objective for
the later research question: the fastest standalone GEMM tile need not be the
best tile once communication overlaps with it.

The script temporarily edits benchmark.go and restores it on exit.

Example
-------
From the Akita-MGPUSim-workspace root:

    python3 tune_gemm_tile.py \
        --tiles 4,8,16 \
        --m 256 --k 256 --n 256 \
        --gpus 2 --algorithm ring

Output:
    gemm_tile_tuning/results.csv
    gemm_tile_tuning/tile_*/metrics.csv
    gemm_tile_tuning/tile_*/stdout.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


FLOAT_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)


def run(cmd, cwd: Path, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(
        list(map(str, cmd)),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def last_float(text: str):
    vals = FLOAT_RE.findall(text)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except ValueError:
        return None


def metric_value(metrics: Path, needle: str):
    """Find a metric by name without assuming a particular CSV schema."""
    if not metrics.exists():
        return None

    with metrics.open(newline="") as f:
        rows = list(csv.reader(f))

    matches = []
    for row in rows:
        joined = ",".join(row).lower()
        if needle.lower() not in joined:
            continue

        # Usually the metric value is the final numeric column.
        numeric = []
        for cell in row:
            try:
                numeric.append(float(cell))
            except ValueError:
                pass
        if numeric:
            matches.append((joined, numeric[-1]))

    if not matches:
        return None

    # Prefer an exact-ish metric name, and Driver if present.
    matches.sort(
        key=lambda x: (
            0 if "driver" in x[0] else 1,
            0 if re.search(rf"(^|[, ]){re.escape(needle.lower())}([, ]|$)", x[0]) else 1,
        )
    )
    return matches[0][1]


def metric_sum(metrics: Path, needle: str):
    """Diagnostic only: sum all CSV rows whose text contains needle."""
    if not metrics.exists():
        return None
    total = 0.0
    found = False
    with metrics.open(newline="") as f:
        for row in csv.reader(f):
            if needle.lower() not in ",".join(row).lower():
                continue
            numeric = []
            for cell in row:
                try:
                    numeric.append(float(cell))
                except ValueError:
                    pass
            if numeric:
                total += numeric[-1]
                found = True
    return total if found else None


def patch_source(original: str, tile: int) -> str:
    patched, n = re.subn(
        r"const\s+tileSize\s*=\s*\d+",
        f"const tileSize = {tile}",
        original,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not find `const tileSize = ...` in benchmark.go")

    # Current source hard-codes 32*32*4 bytes, which is only correct for
    # tileSize=8. Make the local-memory allocation follow the workgroup tile.
    # For a square T x T workgroup and a 4x4 microtile, A's LDS tile is
    # (4T) x (4T) float = (4T)*(4T)*4 bytes.
    patched, n = re.subn(
        r"BlockA:\s*32\s*\*\s*32\s*\*\s*4\s*,",
        "BlockA: (4 * tileSize) * (4 * tileSize) * 4,",
        patched,
        count=1,
    )
    # It is fine if this was already converted manually.
    already = re.search(
        r"BlockA:\s*\(4\s*\*\s*tileSize\)\s*\*\s*\(4\s*\*\s*tileSize\)\s*\*\s*4\s*,",
        patched,
    )
    if n == 0 and not already:
        raise RuntimeError(
            "Could not recognize the BlockA local-memory allocation in benchmark.go"
        )

    return patched


def validate(tile: int, m: int, k: int, n: int, gpus: int):
    out_tile = 4 * tile

    if tile * tile > 256:
        return False, f"{tile}x{tile}={tile*tile} work-items exceeds the conservative 256-thread limit"

    if k % gpus:
        return False, f"K={k} is not divisible by GPUs={gpus}"

    kslice = k // gpus
    bad = []
    if m % out_tile:
        bad.append(f"M={m}")
    if n % out_tile:
        bad.append(f"N={n}")
    if kslice % out_tile:
        bad.append(f"K/GPUs={kslice}")

    if bad:
        return False, f"{', '.join(bad)} not divisible by output/K tile {out_tile}"

    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Akita-MGPUSim-workspace root (default: current directory)",
    )
    ap.add_argument("--tiles", default="4,8,16")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--algorithm", choices=["ring", "naive"], default="ring")
    ap.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra runner arguments appended verbatim",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    mgpu = root / "mgpusim-3.0.3"
    bench_go = mgpu / "benchmarks" / "allreducegemm" / "benchmark.go"
    sample = mgpu / "samples" / "allreducegemm"
    outroot = root / "gemm_tile_tuning"
    outroot.mkdir(exist_ok=True)

    if not bench_go.exists() or not sample.exists():
        raise SystemExit(
            "Expected mgpusim-3.0.3/benchmarks/allreducegemm/benchmark.go "
            "and mgpusim-3.0.3/samples/allreducegemm"
        )

    tiles = [int(x.strip()) for x in args.tiles.split(",") if x.strip()]
    original = bench_go.read_text()
    results = []

    try:
        for tile in tiles:
            valid, why = validate(tile, args.m, args.k, args.n, args.gpus)
            if not valid:
                print(f"\nSKIP tileSize={tile}: {why}")
                results.append(
                    {
                        "tile_size": tile,
                        "workgroup": f"{tile}x{tile}",
                        "output_tile": f"{4*tile}x{4*tile}",
                        "total_time": "",
                        "kernel_metric_sum": "",
                        "status": f"SKIP: {why}",
                    }
                )
                continue

            print(
                f"\n=== tileSize={tile}: {tile}x{tile} work-items, "
                f"{4*tile}x{4*tile} output tile ==="
            )

            bench_go.write_text(patch_source(original, tile))

            # Keep Go formatting canonical.
            fmt = run(["gofmt", "-w", bench_go], cwd=root, check=False)
            if fmt.returncode:
                print(fmt.stdout)

            # Rebuilding Go is enough for this first sweep. The HSACO's
            # microtile is still fixed at 4x4; get_local_size() is runtime.
            build = run(["go", "build", "-o", "allreducegemm_tuned", "."], cwd=sample)
            if build.stdout:
                print(build.stdout)

            run_dir = outroot / f"tile_{tile}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            # MGPUSim writes metrics.csv in the process working directory.
            binary = sample / "allreducegemm_tuned"
            cmd = [
                binary,
                "-timing",
                "--report-all",
                f"-gpus=1,2",
                f"-m={args.m}",
                f"-k={args.k}",
                f"-n={args.n}",
                f"-algorithm={args.algorithm}",
            ] + args.extra

            proc = run(cmd, cwd=run_dir, check=False)
            (run_dir / "stdout.txt").write_text(proc.stdout)

            metrics = run_dir / "metrics.csv"
            total = metric_value(metrics, "total_time")

            # Helpful diagnostic only. Depending on reporter version there may
            # be several kernel_time rows, so don't use this for ranking.
            kernel_sum = metric_sum(metrics, "kernel_time")

            # Fallback if this MGPUSim version reports the metric only to stdout.
            if total is None:
                for line in proc.stdout.splitlines():
                    if "total_time" in line.lower():
                        total = last_float(line)
                        if total is not None:
                            break

            status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
            if total is None and status == "OK":
                status = "OK, but total_time not found"

            row = {
                "tile_size": tile,
                "workgroup": f"{tile}x{tile}",
                "output_tile": f"{4*tile}x{4*tile}",
                "total_time": "" if total is None else total,
                "kernel_metric_sum": "" if kernel_sum is None else kernel_sum,
                "status": status,
            }
            results.append(row)

            print(
                f"tile={tile:2d} output={4*tile:2d}x{4*tile:<2d} "
                f"total_time={total} status={status}"
            )

    finally:
        # Never leave the student's source modified by an interrupted sweep.
        bench_go.write_text(original)
        subprocess.run(["gofmt", "-w", str(bench_go)], cwd=root)

    csv_path = outroot / "results.csv"
    fields = [
        "tile_size",
        "workgroup",
        "output_tile",
        "total_time",
        "kernel_metric_sum",
        "status",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    good = [r for r in results if isinstance(r["total_time"], float)]
    good.sort(key=lambda r: r["total_time"])

    print("\n=== Ranking by end-to-end simulated total_time ===")
    if good:
        best = good[0]["total_time"]
        for i, r in enumerate(good, 1):
            rel = r["total_time"] / best
            print(
                f"{i:2d}. tileSize={r['tile_size']:2d} "
                f"output={r['output_tile']:>5s} "
                f"time={r['total_time']:.9g}  "
                f"{rel:.3f}x best"
            )
    else:
        print("No parseable total_time results. Inspect tile_*/stdout.txt and metrics.csv.")

    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
