# results/

Raw simulator output. Everything here except this file is gitignored, because the
data can grow large. Put the plots and figures you derive from it in `../diagrams/`.
Those are tracked.

Layout: one subfolder per experiment.

- `gemm_tile_tuning/`: output of `scripts/tune_gemm_tile.py`, with `results.csv`
  (the summary) plus `tile_<N>/metrics.csv` and `tile_<N>/stdout.txt` for each run.
- `allreducegemm/`: metrics from manual `samples/allreducegemm` runs.
