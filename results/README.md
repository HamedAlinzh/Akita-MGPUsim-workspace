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
  values (34 GB/s, 140 ns) gives 430.195 us. `defaults_after_step2/` and `repeat_a/b/` rerun
  the defaults with the Step 2 binary: window 432.742 us every time; total_time varies by
  4 ns between identical runs (589.961 or 589.965 us), a pre-existing nondeterminism in the
  copy-back after the window.
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
- `step3_tiles/`: Step 3 (`scripts/step3_sweep.sh main|variants`, summary
  `summary.txt` from `scripts/step3_summary.py`). `calib/`: 1-GPU GEMM times at tileSize
  4/8/16 and the `-valu-cycles-scale` fit for tileSize 4. `runs/`: near-memory placement
  (all-reduce tile/bulk, reduce-scatter), tileSize 4 x N = 1/2/4 under the native and
  matrix-core curves (owner tile, bulk N=1/4, mccl ring, near-memory N=1), tileSize 8 owner
  tile reruns with first-ready logging, flatter/steeper curve variants at N=2, and 4-GPU
  mccl ring / engine bulk. Mapping: sim tileSize 4 = real 128x128, 8 = 256x256; link
  knobs held at the f≈0.5 point. The T3/CAIS comparison is in
  `reports/fused-gemm-comm-design.md` ("Reduction placements and T3"). Caveat: the
  matrix-core curve is extrapolated from MI355X data (HipKittens), and tileSize 4 has 8
  sim waves vs 3.37 real, so its readiness is finer than reality.
- `step4a_shaping/`: dispatch-shaping spike. `tracker/` (1 GPU, f≈0.5 shape,
  `TestTileTrackerShaping`, `spread_summary.txt`): `-wg-cap` 4/6/8 and `-wg-ramp` 1/4.
  `runs/` (`scripts/step4a_runs.sh`, `summary.txt` from `scripts/step4a_summary.py`): 2-GPU
  tile mode N=1 at f≈0.5 and f≈0.8 with `-wg-ramp 1` and `-wg-cap 4`, the f≈0.8 baseline,
  tileSize 4 at f≈0.8 (matrix-core curve), 4-GPU mccl/bulk/tile on the per-pair mesh
  (`-peer-mesh`), and the default 433 us check (window 432.742 us).
- `step4b_capped/`: the capped baseline (`-gemm-wg-cap 8`). `calib/`: 1-GPU GEMM at 256x256
  (tileSize 8; tileSize 4 at -gemm-valu-scale 0.647 and 0.5 -> 0.738 for the matrix-core
  curve). `runs/` (`scripts/step4b_runs.sh`, `summary.txt` from `scripts/step4b_summary.py`):
  mccl ring / engine bulk / engine tile at f≈0.5 and f≈0.8 with re-derived knobs, `-wg-ramp
  1` on top, small-tiles-first at R = 32/64/128 (2 GPUs) and its GEMM alone (1 GPU), and the
  default 433 us check. `runs_queue_order_bug/`: the first small-tiles-first runs, invalid:
  the main GEMM's queue was created first and the driver serves queues in creation order, so
  the small-tile kernel did not start first. Caveat found in Step 5: even after that fix the
  CP interleaves concurrent kernels' work-groups (no `-ordered-dispatch` then), so the
  small-tile work-groups did not all dispatch first in these runs.
- `step5_splitk/`: split-K for the first and last bands of C (`-splitk-ways s -splitk-rows R`,
  s = 2 at R = 64, s = 4 at R = 32), on the capped baseline's f≈0.5/f≈0.8 shape and knobs
  (2 GPUs, 4 CUs, 256x256, K/GPU 256, tileSize 8, `-gemm-wg-cap 8`, `-cp-dispatchers 16 -ordered-dispatch`).
  `runs/` (`scripts/step5_runs.sh`, `summary.txt` from `scripts/step5_summary.py`), 9 runs,
  each with a `wgs.csv` work-group trace (`-wg-trace`):
  1. `gemm_s2`, `gemm_s4`: GEMM alone (1 GPU, K 256, engine tile mode), pre-sum.
  2. `f05_tile_s2`, `f05_tile_s4`: engine tile mode at f≈0.5, pre-sum.
  3. `f05_tile_s2_nopresum`: s = 2 without pre-sum (`-splitk-presum off`).
  4. `f08_tile_s2`, `f08_tile_s4`: engine tile mode at f≈0.8, pre-sum.
  5. `f05_bulk_s2`: engine bulk mode at f≈0.5, s = 2, pre-sum.
  6. `defaults_433`: default flags, expect 432.742 us.
  7. `f05_tile_s1` (added after the first results): the s = 1 baseline with the same flags
     and a trace, to confirm 632.4 us and time plain work-groups.

  Result (all verify; defaults 432.742 us; s = 1 with the new flags still 632.4 us). At f≈0.5,
  s = 2 with pre-sum gives 578.0 us (-8.6%) and GEMM 339.4 us (-5.9%); s = 4 gives 612.2
  (-3.2%); no pre-sum gives 657.6 (+4.0%). At f≈0.8: s = 2 1614.8 (-4.8%), s = 4 1585.7
  (-6.5%). Bulk s = 2: 766.9 (-2.2%), all from the faster GEMM. First ready: 205.6 / 119.2 /
  97.1 us for s = 1 / 2 / 4. The GEMM is faster because plain round-1 work-groups, all
  starting together, take 172 us vs 137 in round 2; the split bands replace that round.
  Residency is 8 per CU in every band, and a split work-group's time per unit of K is 1.04x
  (s = 2) and 1.43x (s = 4) a plain one's. The queue model (tiles served in ready order)
  predicts every window within 1-3%; max(GEMM, first ready + comm) misses s = 4 at f≈0.5 by
  12%, because the link idles between the first band and the middle.
  `runs_interleaved/`: the first GEMM-alone runs, without `-ordered-dispatch`; the
  dispatchers interleaved all five or nine kernels, so the bands did not go in order.
- `step5b_bounds/`: what is left after split-K. `model.txt` (`scripts/step5b_model.py`, no
  runs): a work-group list-scheduling model (32 slots, times from the Step 5 traces) feeding
  the queue model; bounds for an even tile stream and a search over a split factor per tile
  row. `runs/` (`scripts/step5b_runs.sh`, `summary.txt` from `scripts/step5b_summary.py`):
  B. `warm_plain`, `warm_s2`: 1 GPU, GEMM alone, launched twice (`-warmup`), with
  `-mem-timeline`. Plain round-1 work-groups take 171.1 us cold and 142.7 warm (round 2:
  136.8 / 133.3); cold round 1 reads 397 KB from DRAM (6,196 L2 misses, 104 L2-TLB misses),
  warm about 0. Warm, split s = 2 no longer speeds the GEMM up (281 vs 280 us from first
  dispatch to last end). C. `c_f05_plan`, `c_f08_plan`: `-splitk-plan 4,4,4,4,1,1,2,2`
  (21 launches per GPU, `-cp-dispatchers 32`): 524.9 us at f≈0.5 (-9.2% vs Step 5's 578.0,
  -17.0% vs s = 1) and 1,587.6 us at f≈0.8 (+0.1% vs 1,585.7). `repro_gemm_s2`: Step 5's
  gemm_s2 with this binary, 334.764 vs 334.490 us (driver ordered launch). `defaults_433`:
  432.742 us. `runs_misordered/`: the first `warm_s2`, whose second launch reached the CP out
  of order (before the driver's ordered launch).
- `pending_experiments.log`: log of `scripts/run_pending_experiments.sh`.
- `gemm_tile_tuning/`: **outdated.** Output of `scripts/tune_gemm_tile.py`, taken before
  commit `3be1037` with the one-workgroup-row-per-launch benchmark (tile size also set the
  launch count) and always on 2 GPUs. Kept for the record only.
- `allreducegemm/`: **outdated.** Metrics from manual `samples/allreducegemm` runs taken with
  the same pre-`3be1037` benchmark.
