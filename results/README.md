# results/

Raw simulator output. Everything here except this file is gitignored, because the
data can grow large. Put the plots and figures you derive from it in `../diagrams/`.
Those are tracked.

Layout: one subfolder per experiment.

- `overlap_sweep/`: output of `scripts/sweep_overlap.py` (tile size × band rows ×
  overlap), with `results.csv` (the summary) plus one `<config>/` folder per run holding
  `metrics.csv` and `stdout.txt`.
- `gemm_tile_tuning/`: **outdated.** Output of `scripts/tune_gemm_tile.py`, taken before
  commit `3be1037` with the one-workgroup-row-per-launch benchmark (tile size also set the
  launch count) and always on 2 GPUs. Kept for the record only.
- `allreducegemm/`: **outdated.** Metrics from manual `samples/allreducegemm` runs taken with
  the same pre-`3be1037` benchmark.
