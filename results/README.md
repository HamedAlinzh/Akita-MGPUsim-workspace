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
- `step1_threads/`: All-Reduce kernel thread count (`-collective-threads` 1024 / 4096 /
  16384) at 1 MB on 2 GPUs, and `gemm_t16384/` for one GEMM case at 16384 threads.
- `step0_ratio/`: output of `scripts/ratio_model.py`. `report.txt` is the printed summary;
  `real_layers.csv` (f and waves per real layer), `points.csv` (the three representative
  points, real vs simulated, and the derived knobs), `sensitivity_eff.csv`,
  `sensitivity_latency.csv`, `moe.csv`. Simulated GEMM times come from `step1_tracker/`.
  Caveats: simulated waves at the f≈0.8 and f≈0.5 points (2.0) are a bit above real (1.68),
  so the first tiles finish slightly early (0.42 of the GEMM vs 0.50) and overlap gains there
  will be slightly optimistic. Real f at the f≈0.5 and f≈0.25 points moves by about ±0.1
  with the efficiency guesses, so Step 4 sweeps `-link-bw` as a curve with the three points
  marked. The MoE rows are a discussion item only: in-node, hardware triggering keeps f at
  0.15 or below, so the win there is removing launch latency, not overlap.
- `stepD_knobs/`: before/after check of `-link-bw`/`-link-latency` (`run.sh`). Defaults
  reproduce the 433 us baseline exactly (window 432.742 us); the peer fabric at PCIe-like
  values (34 GB/s, 140 ns) gives 430.195 us.
- `step1_tracker/`: Step 1 tile-tracker spike, produced by
  `TILETRACKER_OUT=<this dir> go test -run TestTileTracker -v ./benchmarks/allreducegemm/`.
  `test_output.txt` is the summary; `tiles_cu<CUs>_M_N_K_t<tileSize>.csv` lists every tile
  completion in order (times relative to arming, just before the GEMM launch).
  `spread_summary.txt` (from `scripts/tile_spread.py`) gives first/median/last completion and
  completion groups per CSV; the `cu4_` files are the representative-point shapes, run with
  `-lds-occupancy` on (`TestTileTrackerRepresentative`); `cu16_M768_N512` is the 16-CU
  scale check (`TestTileTrackerScaleCheck`, log in `scale_check.log`).
- `step2_engine/`: Step 2, the communication engine. `correctness/` (from
  `scripts/step2_correctness.sh`; summary `correctness_summary.txt` from
  `scripts/step2_summary.py`): 2 GPUs, 4 CUs, 384x256, K/GPU 256, tileSize 8, f≈0.5 knobs,
  N = 1/2/4 in tile and bulk mode, reduce-scatter at N = 1, and a 4-GPU all-reduce; every run
  verifies and message/byte counts equal the expected values. `timing/` (from
  `scripts/step2_timing.sh`): mccl ring all-reduce at f≈0.5 and ring reduce-scatter at
  f≈0.25. `calibration.csv` from `TestCalibration` in `timing/commengine`.
- `pending_experiments.log`: log of `scripts/run_pending_experiments.sh`.
- `gemm_tile_tuning/`: **outdated.** Output of `scripts/tune_gemm_tile.py`, taken before
  commit `3be1037` with the one-workgroup-row-per-launch benchmark (tile size also set the
  launch count) and always on 2 GPUs. Kept for the record only.
- `allreducegemm/`: **outdated.** Metrics from manual `samples/allreducegemm` runs taken with
  the same pre-`3be1037` benchmark.
