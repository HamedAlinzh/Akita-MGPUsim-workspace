# results/

Raw simulator output. Everything here except this file is gitignored, because the
data can grow large. Put the plots and figures you derive from it in `../diagrams/`.
Those are tracked.

Layout: one subfolder per experiment.

- `overlap_sweep/`: output of `scripts/sweep_overlap.py` (tile size × band rows ×
  overlap), with `results.csv` (the summary) plus one `<config>/` folder per run holding
  `metrics.csv` and `stdout.txt`.
- `overlap_sweep_k1024/`, `overlap_sweep_k4096/`: the same sweep for the 256×256×1024 and
  256×256×4096 shapes, on the fixed mccl. These back `reports/gemm-allreduce-overlap.html`.
  `overlap_sweep/` (512³) ran before the mccl fix and is a reference only.
- `allreduce_breakdown/{before,after}/`: `samples/allreducebench` output, one CSV per GPU
  count, splitting each Ring All-Reduce into setup copies, kernel time, link time and bytes,
  before and after the mccl queue fix.
- `stage1_minrepro/`, `stage1_fixed/`: Stage 1 of the follow-up (the 4-GPU overlap
  correctness bug). `run*.sh` in each folder reproduce the runs; `stage1_fixed/` was run
  with the L1 acquire on (the default since then).
- `stage2_h2d/`: Stage 2, sensitivity to the driver's per-copy latency (`-h2d-cycles`
  14500 / 7250 / 1450): `gemm_h<cycles>/` sweeps and `allreduce/h<cycles>.csv`.
- `pending_experiments.log`: log of `scripts/run_pending_experiments.sh`.
- `gemm_tile_tuning/`: **outdated.** Output of `scripts/tune_gemm_tile.py`, taken before
  commit `3be1037` with the one-workgroup-row-per-launch benchmark (tile size also set the
  launch count) and always on 2 GPUs. Kept for the record only.
- `allreducegemm/`: **outdated.** Metrics from manual `samples/allreducegemm` runs taken with
  the same pre-`3be1037` benchmark.
