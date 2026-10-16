#!/usr/bin/env python3
"""
Sweep GEMM tile size x band size x overlap for the K-split GEMM + All-Reduce
benchmark (samples/allreducegemm), in parallel.

For each tile size this builds one simulator binary (patching tileSize in
benchmark.go via tune_gemm_tile.patch_source, then restoring it), and then
runs every (GPUs, tile, band rows, overlap) combination as its own process:

    band rows 0       -> one GEMM launch over all of M, then one All-Reduce
    band rows R       -> M/R bands, each GEMM launch + All-Reduce of R*N floats
      --overlap off   -> band r's All-Reduce finishes before band r+1 starts
      --overlap on    -> band r is all-reduced while band r+1 computes

--chunks adds the All-Reduce scratch-buffer size (elements; 0 = the
benchmark's default of 65536) as a fourth dimension.

Every run passes -verify. Per run, the summary keeps the Driver's total_time
and kernel_time from metrics.csv and the benchmark's own phase split
(window, gemm_wait, allreduce) from stdout.

Example:
    python3 scripts/sweep_overlap.py --m 512 --k 512 --n 512 \\
        --gpus 2,4 --tiles 4,8,16 --bands 0,64,128,256 --jobs 3

Output (rerunning skips configurations that already finished):
    results/overlap_sweep/results.csv
    results/overlap_sweep/<config>/{metrics.csv,stdout.txt}
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_gemm_tile import metric_value, patch_source  # noqa: E402

PHASES_RE = re.compile(
    r"phases: window=(\S+) gemm_wait=(\S+) allreduce=(\S+)")

FIELDS = [
    "m", "k", "n", "gpus", "tile_size", "output_tile", "band_rows",
    "num_bands", "overlap", "allreduce_chunk", "total_time", "kernel_time", "window",
    "gemm_wait", "allreduce", "verified", "status", "wall_seconds",
]


def config_name(gpus, tile, band, overlap, chunk=0):
    mode = "ov" if overlap else "seq"
    name = f"g{gpus}_t{tile}_b{band}_{mode}"
    return f"{name}_c{chunk}" if chunk else name


def build_binaries(root: Path, tiles, bindir: Path):
    mgpu = root / "simulator" / "mgpusim-3.0.3"
    bench_go = mgpu / "benchmarks" / "allreducegemm" / "benchmark.go"
    sample = mgpu / "samples" / "allreducegemm"
    bindir.mkdir(parents=True, exist_ok=True)

    original = bench_go.read_text()
    binaries = {}
    try:
        for tile in tiles:
            bench_go.write_text(patch_source(original, tile))
            binary = bindir / f"allreducegemm_t{tile}"
            print(f"building tileSize={tile} -> {binary}", flush=True)
            subprocess.run(
                ["go", "build", "-o", str(binary), "."], cwd=sample, check=True)
            binaries[tile] = binary
    finally:
        bench_go.write_text(original)

    return binaries


def run_one(binary: Path, run_dir: Path, args, gpus, band, overlap, chunk):
    run_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = ",".join(str(i) for i in range(1, gpus + 1))
    cmd = [
        str(binary), "-timing", "--report-all", "-verify",
        f"-gpus={gpu_ids}",
        f"-m={args.m}", f"-k={args.k}", f"-n={args.n}",
        f"-algorithm={args.algorithm}",
        f"-band-rows={band}",
        f"-overlap={'true' if overlap else 'false'}",
        f"-allreduce-chunk={chunk}",
    ]
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=run_dir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    wall = time.time() - start
    (run_dir / "stdout.txt").write_text(" ".join(cmd) + "\n" + proc.stdout)

    metrics = run_dir / "metrics.csv"
    phases = PHASES_RE.search(proc.stdout)
    return {
        "total_time": metric_value(metrics, "total_time"),
        "kernel_time": metric_value(metrics, "kernel_time"),
        "window": float(phases.group(1)) if phases else None,
        "gemm_wait": float(phases.group(2)) if phases else None,
        "allreduce": float(phases.group(3)) if phases else None,
        "verified": "Passed!" in proc.stdout,
        "status": "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})",
        "wall_seconds": round(wall, 1),
    }


def main():
    # A shutdown sends SIGTERM; exiting through SystemExit runs the finally
    # block in build_binaries, so benchmark.go is never left patched.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(1))

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Workspace root (default: parent of scripts/)")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--gpus", default="2,4")
    ap.add_argument("--tiles", default="4,8,16")
    ap.add_argument(
        "--bands", default="0,64,128,256",
        help="Band rows to sweep; 0 = one band (always run without overlap)")
    ap.add_argument(
        "--chunks", default="0",
        help="All-Reduce chunk sizes in elements; 0 = benchmark default (65536)")
    ap.add_argument("--algorithm", choices=["ring", "naive"], default="ring")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    outroot = (args.outdir or root / "results" / "overlap_sweep").resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    csv_path = outroot / "results.csv"

    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]  # noqa: E731
    gpu_counts, tiles, bands = ints(args.gpus), ints(args.tiles), ints(args.bands)
    chunks = ints(args.chunks)

    done = {}
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "OK" and row["verified"] == "True":
                    key = config_name(
                        int(row["gpus"]), int(row["tile_size"]),
                        int(row["band_rows"]), row["overlap"] == "True",
                        int(row.get("allreduce_chunk") or 0))
                    done[key] = row

    todo = []
    for gpus in gpu_counts:
        for tile in tiles:
            out_tile = 4 * tile
            if args.k % gpus or (args.k // gpus) % out_tile \
                    or args.m % out_tile or args.n % out_tile:
                print(f"skip g{gpus} t{tile}: dimensions not tile-aligned")
                continue
            for band in bands:
                if band and (band % out_tile or args.m % band):
                    continue
                for overlap in ([False] if band == 0 else [False, True]):
                    for chunk in chunks:
                        name = config_name(gpus, tile, band, overlap, chunk)
                        if name not in done:
                            todo.append((gpus, tile, band, overlap, chunk))

    print(f"{len(done)} configurations already done, {len(todo)} to run")
    rows = list(done.values())

    def write_csv():
        rows.sort(key=lambda r: (int(r["gpus"]), int(r["tile_size"]),
                                 int(r["band_rows"]), str(r["overlap"]),
                                 int(r.get("allreduce_chunk") or 0)))
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    if not todo:
        write_csv()
        return

    binaries = build_binaries(
        root, sorted({t for _, t, _, _, _ in todo}), outroot / "bin")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_one, binaries[tile],
                outroot / config_name(gpus, tile, band, overlap, chunk),
                args, gpus, band, overlap, chunk,
            ): (gpus, tile, band, overlap, chunk)
            for gpus, tile, band, overlap, chunk in todo
        }
        for finished, fut in enumerate(as_completed(futures), 1):
            gpus, tile, band, overlap, chunk = futures[fut]
            res = fut.result()
            row = {
                "m": args.m, "k": args.k, "n": args.n, "gpus": gpus,
                "tile_size": tile, "output_tile": f"{4*tile}x{4*tile}",
                "band_rows": band,
                "num_bands": args.m // band if band else 1,
                "overlap": overlap, "allreduce_chunk": chunk, **res,
            }
            rows.append(row)
            write_csv()
            print(
                f"[{finished}/{len(todo)}] "
                f"{config_name(gpus, tile, band, overlap, chunk)}: "
                f"total={res['total_time']} verified={res['verified']} "
                f"{res['status']} ({res['wall_seconds']}s)",
                flush=True,
            )

    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
