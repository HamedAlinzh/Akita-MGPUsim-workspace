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
- `step6_paper/`: the per-row split plan, made paper-ready. `model.txt` from
  `scripts/step6_model.py` (work-group schedule + per-link message queues; also the MI300X
  projection, model only); `runs/` from `scripts/step6_runs.sh`, `summary.txt` from
  `scripts/step6_summary.py`. Every split run and its s = 1 baseline use `-ordered-dispatch`
  (CP order and driver launch order) and `-cp-dispatchers 32`; every run verifies. Runs:
  1. `warm_f05_s1`, `warm_f05_plan`: f≈0.5, 2 GPUs, s = 1 and plan 4,4,4,4,1,1,2,2, both
     with `-warmup`.
  2. `f025_tile_s1`, `f025_bulk_s1`, `f025_tile_plan`: f≈0.25, 512x320, K/GPU 256, cap 8,
     reduce-scatter, knobs from `points_step6.csv`; the plan is the model's best after
     recalibrating work-group times on `f025_tile_s1`.
  3. `f05_4gpu_tile_s1`, `f05_4gpu_tile_plan`: f≈0.5 shape on 4 GPUs (`-peer-mesh`, K/GPU
     256), knobs from `points_step6.csv`, plan 4,4,4,4,1,1,2,2 (the model's best, tied).
  4. `defaults_433`: default flags, expect 432.742 us.

  Result (all verify; defaults 432.742 us). Gain of the plan over s = 1 / over engine bulk:
  f≈0.25 (cold, reduce-scatter, plan 2,2,1x10,2,2) -3.7% / -24.0% (785.5 vs 815.5 / 1,033.4
  us; achieved f 0.246); f≈0.5 cold -17.0% / -33.1%, warm -15.5% (516.1 vs 610.8 us);
  f≈0.5 on 4 GPUs -16.4% (559.2 vs 669.0 us); f≈0.8 cold -6.4% / -14.1%. The model
  (`model.txt`) predicted every Step 6 run within -0.5% to +4.3% (it reads high). Cold = A,
  B and C all cold in L2 and TLBs (first launch); warm = all warm (second launch); a real
  layer, with weights from HBM and activations possibly warm, lies between. The f≈0.25 sim
  shape is 5.0 waves at cap 8 (the real layer 3.37). MI300X projection (model only, split
  costs assumed from the simulator): -8.4% at f≈0.8, -22.6% at f≈0.5, -11.6% at f≈0.25, the
  last mostly from a shorter GEMM tail (wave quantization), not overlap.
- `step7_baselines/`: fair baselines and model validation (batch 1 of 3). `model.txt` from
  `scripts/step7_model.py`, `sensitivity.txt` from `scripts/step7_sensitivity.py` (model
  only), `runs/` from `scripts/step7_runs.sh`, `summary.txt` from `scripts/step7_summary.py`.
  f≈0.5 shape (256x256, K/GPU 256, 4 CUs, tileSize 8, `-gemm-wg-cap 8`); split runs and
  their s = 1 baselines use `-ordered-dispatch` and enough `-cp-dispatchers`; every run
  verifies. Runs:
  1. `sw_s1`, `sw_plan`: software overlap on 2 GPUs (`-sw-overlap`, mccl ring per band,
     `-collective-threads 16384`), bands 128,128 at s = 1 and bands 128,64,64 with the plan
     4,4,4,4,1,1,2,2 (the model's best layouts with at least 2 bands).
  2. `x4_bulk`, `x4_direct`: 4 GPUs on the per-pair mesh, engine bulk vs mccl's new direct
     reduce-scatter + all-gather after the GEMM (`-algorithm direct`), Step 6 4-GPU knobs.
  3. `x8_tile_s1`, `x8_tile_plan`, `x8_direct`: 8 GPUs (`-peer-mesh`, K/GPU 256), knobs
     "f~0.5 8 GPUs" in `points_step6.csv`, engine runs with `-engine-peer-queues`; engine tile
     s = 1, plan 4,4,4,4,1,1,2,2 (tied best in the 8-GPU search), direct collective.
  4. `rank_2nd`, `rank_10th`, `rank_median`, `rank_worst`: 2 GPUs, engine tile, plans
     4,4,4,4,4,4,4,1 / 1,2,4,1,1,1,1,1 / 4,1,4,4,2,1,2,4 / 1,1,1,1,4,4,4,4 (2nd and 10th best
     distinct predicted windows, median and worst of the 6,561-plan search).
  5. `defaults_433`: default flags, expect 432.742 us.

  Result (all 12 verify; defaults 432.742 us). Software overlap: s = 1 1,281.9 us, plan
  1,735.1 us (+35.4%; the engine's plan is -17.0%), so none of the gain survives; mccl after
  the GEMM without overlap is 1,306.4 us. The model, with F = 277 us per all-reduce call and
  50% link efficiency fitted to the capped mccl runs, read 10-12% high but predicted the
  plan's loss (+469 vs +453 us). The collectives' 256 work-groups sit on the GEMM's CUs
  (about 20 resident per CU) and slow its work-groups about 2x. 4 GPUs: engine bulk 797.0 us,
  direct software 1,298.9 us (engine tile s = 1 669.0, plan 559.2 from Step 6). 8 GPUs: engine
  tile s = 1 655.1 us, plan 567.9 us (-13.3%), direct software 1,824.0 us (plan -68.9%);
  model +10.3% / +5.5%. Rank test over 8 measured plans: Spearman 0.970; the model's chosen
  plan 4,4,4,4,1,1,2,2 is the best measured (524.9 us; 4,4,4,4,4,4,4,1 526.4 us). MI300X
  split-cost sensitivity: the f≈0.5 gain stays above 10% when either split keeps its
  measured cost (the search uses the other), and falls below 10% at 1.63x when both cost
  the same; no published per-split cost exists for MI300X or MI250.
- `step8_edges/`: realistic software baseline and edge cases (batch 2 of 3). Model: the
  Step 6 work-group schedule with an event-driven message model (`scripts/step8_model.py`;
  `scripts/step8_validate.py` checks it on 22 past runs); every run is predicted before it
  runs. `mccl_breakdown.txt` (`scripts/step8_mccl_breakdown.py`), `sw_sweep.txt`
  (`scripts/step8_sw.py`), `fsweep.txt` (`scripts/step8_fsweep.py`): model only. `runs/` from
  `scripts/step8_runs.sh`; split runs and their s = 1 baselines use `-ordered-dispatch`
  and `-cp-dispatchers 40`, 4- and 8-GPU engine runs `-engine-peer-queues`; every run
  verifies. Batch A (items 0 and 2):
  1. `x4_s1_pq`, `x4_plan_pq`: Step 6's 4-GPU f≈0.5 engine runs (s = 1, plan 4,4,4,4,1,1,2,2)
     with `-engine-peer-queues`.
  2. `f035_s1`, `f035_plan`, `f065_s1`, `f065_plan`, `f090_s1`, `f090_plan`: 2 GPUs, f = 0.35,
     0.65, 0.9 (knobs from `fsweep.txt`), s = 1 and the model's best plan at that f.

  Batch A result (all 8 verify): 4 GPUs with per-peer queues s = 1 641.4 us, plan 531.5 us
  (-17.1%; Step 6 with one queue 669.0 / 559.2). f sweep, plan vs s = 1: -14.4% at f 0.35,
  -12.9% at 0.65, -3.3% at 0.9 (model -18.7%, -14.5%, -3.8%; model errors -3.5% to +1.5%).
  Model fix: the Step 6 tail (a full round trip after the last link finish) double-counted
  the last transfer, which grows with the per-message cost (+10.3% at 8 GPUs); messages are
  now simulated with the sum waiting for every partial (22 past runs: -4.9% to +1.5%, mean
  |error| 1.5%, was 3.0%). One mccl call (128 KB band, 2 GPUs): 530 us = 138 launch (68 host
  copies) + 211 transfer + 163 push beyond transfer + 17 reduce; nothing in it is an mccl
  artifact removable without new kernels (the driver's per-launch code-object copy is about
  23 us per call), so the software runs were not repeated. Software sweep: splits help from
  about 20 us per call (RCCL point-to-point minimum on MI300A, arXiv 2508.11298): -5.6% at the
  measured CU loss, -9.2% with none; still 50-67% behind the engine, mostly from the
  software path's 50% link efficiency (at the engine's 79%, 5 us and no CU loss: within 4.8%).

  Batch B (items 3 and 4; `scripts/step8_runs_b.sh`, knobs and plans from `batchb.txt`,
  `scripts/step8_batchb.py`; f ≈ 0.555 unless noted, latency/transfer 0.39), each as `_s1`
  and `_plan` (the model's best plan):
  3. `w15` 192x256 (1.5 waves), `w337` 384x288 (3.375 waves), `r9` 288x256 (9 tile rows,
     2.25 waves), `k128` and `k512` (256x256 at K/GPU 128 and 512).
  4. `rs05`, `rs08`: reduce-scatter only (`-reduce-scatter`) at f ≈ 0.555 and 0.814.
  5. `defaults_433`: default flags, expect 432.742 us.

  Batch B result (all 15 verify; defaults 432.742 us), plan vs s = 1: 1.5 waves -20.1%,
  3.375 waves -10.8%, 9 tile rows -16.3%, K/GPU 128 -13.2%, K/GPU 512 -20.2%,
  reduce-scatter -16.8% (f 0.555) and -6.5% (f 0.814). Model (processor-sharing schedule
  + event messages) errors -3.2% to +1.5%. Steady split cost per work-group, normalized to
  8 resident: 1.03-1.09 (s = 2) and 1.13-1.25 (s = 4) at K/GPU 256, 1.18 / 1.42 at 128 (a
  32-wide K chunk is mostly fixed cost), 1.12 / 1.14 at 512.
- `step9_baselines/`: fair software baseline, gain split, realistic caches (batch 3 of 3).
  `runs/` from `scripts/step9_runs.sh` (run list `runs.txt`, written by
  `scripts/step9_model.py`); f≈0.5 shape (256x256, K/GPU 256, 4 CUs, tileSize 8, cap 8,
  `-lds-occupancy`), `-ordered-dispatch -cp-dispatchers 40`, every run verifies, 8-GPU runs
  use `-peer-mesh`. FlashOverlap-style runs (`-sw-flash`, no engines) use `-sw-comm-cus 1`,
  `-sw-signal-cost 0`, `-collective-threads 1024`; their group partitions are the model's.
  1. `cal_one`, `cal_rows`: calibration of the comm kernels on 1 CU at the f≈0.5 link knobs
     (s = 1; one 256-row group, i.e. no overlap, and eight 32-row groups). They fix the
     model's per-kernel costs and link efficiency before the runs below are predicted.
     `cal_one_cus2`: `cal_one` with `-sw-comm-cus 2`, to see whether the comm kernels' link
     efficiency depends on their CU count. A first pair of calibration runs, with the GEMM
     shut out of the comm CU only while a comm kernel ran, is in `runs_dynamic/` (a GEMM
     work-group left on that CU straggled to 687 us); every run here takes the comm CU from
     the comm kernels' launch, before the GEMM's.
  2. `fo_f05_s1`, `fo_f05_plan`, `fo_f08_s1`, `fo_f08_plan` (2 GPUs, f≈0.5 / f≈0.8 knobs),
     `fo_x8_s1`, `fo_x8_plan` (8 GPUs, f≈0.5): FlashOverlap-style, s = 1 and the plan the
     model picks with its partition.
  3. `gemm_plan`, `gemm_plan_warm`: GEMM alone with plan 4,4,4,4,1,1,2,2 (`-engine-mode
     bulk`, so the engines start after the GEMM), cold and `-warmup`, for the gain split.
  4. `wab_s1`, `wab_plan`: f≈0.5, 2 GPUs, engine tile, s = 1 and plan 4,4,4,4,1,1,2,2 with
     `-warmup -warmup-skip-b` (A and C warm, B cold); `warm_s1`, `warm_plan`: the same with
     `-warmup` (all warm), so cold (`eng_f05_*`), A+C warm and warm come from one simulator.
  5. `eng_f05_s1`, `eng_f05_plan`, `eng_f08_s1`, `eng_f08_plan`, `eng_x8_s1`, `eng_x8_plan`:
     the engine (tile mode) s = 1 and plan 4,4,4,4,1,1,2,2 runs of Steps 4b/5b/7, rerun with
     the L1 fix below so the four-way comparison uses one simulator.
  6. `defaults_433`: default flags, expect 432.742 us.
  7. `fo_x8_s1_cus2`: `fo_x8_s1` with `-sw-comm-cus 2`, added after the 8-GPU runs reached
     only 0.11 of the link with one comm CU, to test whether that CU is the limit.

  Result (all 24 verify; defaults 432.742 us; `summary.txt`, `scripts/step9_summary.py`).
  Four-way, window in us (FlashOverlap-style s = 1 / + plan / engine s = 1 / + plan):
  f≈0.5, 2 GPUs 979.4 / 1009.0 / 632.4 / 525.1; f≈0.8, 2 GPUs 2721.9 / 3060.8 / 1695.4 /
  1587.6; f≈0.5, 8 GPUs 3758.2 / 3918.7 / 655.1 / 574.6 (2 comm CUs: 2388.6). The plan helps
  the engine (-17.0%, -6.4%, -12.3%) and hurts the software path (+3.0%, +12.5%, +4.3%):
  its pre-sum kernels queue on the one comm CU. The CU-driven path moves 0.50 of the link
  alone (1 or 2 comm CUs), 0.42-0.48 with several groups in flight at 2 GPUs, and 0.11
  (0.18 with 2 CUs) at 8 GPUs, where one CU cannot drive 7 links; FlashOverlap reports
  13% for one 192 KB tile message. Holding a comm CU costs the GEMM 360 -> 490-515 us.
  Gain split (engine): the plan's GEMM is 1-4% longer at every Step 9 point (no Stream-K
  tail gain at 2 waves), so the overlap part is -7% to -19% of the s = 1 window, more than
  the whole gain. Caches, f≈0.5 engine s = 1 / plan: cold 632.4 / 525.1 (-17.0%), A and C
  warm 623.0 / 516.6 (-17.1%), warm 610.8 / 516.2 (-15.5%). Model: engine runs -5.1% to
  +1.5%; FlashOverlap-style 2-GPU runs -10.7% to +2.8% after calibration (cal_one,
  cal_rows), 8 GPUs -74% (it has no CU limit). All-gather before the GEMM: design note
  `reports/allgather-before-gemm.md` (model: 8 GPUs, split plan -27% vs per-row overlap).

  L1 fix (found here): the write-around L1's invalidate (the CP's acquire at every kernel
  launch, `-l1-acquire`) reset the tag directory while fills were in flight, so a later
  hit could return another line's data. FlashOverlap-style `fo_f05_s1` failed to verify
  this way (a GEMM work-group read two wrong lines of A, 487 us, GPU 2 CU 2); the L1 now
  invalidates its blocks in place, so a block with a fill or a read in flight keeps its
  storage until done. Also `rand.Seed` does nothing
  since Go 1.24, so A and B differed every run (timing does not depend on them); the
  benchmark now uses its own seeded source. `runs_prefix/` has this step's runs from
  before both fixes (same names).
- `step10_cushare/`: FlashOverlap-style baseline at a realistic CU share. 16 CUs per GPU
  (`-cus-per-gpu 16`), 512x512, K/GPU 256 (`-k 512`), cap 8: 256 tiles over 16 x 8 slots =
  2.0 waves, as 256x256 on 4 CUs. Link knobs re-derived with `ratio_model.derive_knobs`
  (f 0.555 as the f≈0.5 point, latency/transfer 0.39, the 4-CU capped GEMM time as the
  estimate): `-link-bw 3.1218 -link-latency 110 -msg-overhead 530` (C is 4x larger, so the
  link is 4x faster). `-ordered-dispatch -cp-dispatchers 40`, every run verifies; same binary
  as Step 9 (`8435471`, defaults 432.742 us). `runs/` from `scripts/step10_runs.sh` (run list
  `runs.txt` from `scripts/step10_model.py`, which predicts each run first):
  1. `eng_s1`, `eng_plan`: engine tile mode, s = 1 and the model's plan.
  2. `fo_s1`, `fo_plan`: FlashOverlap-style (`-sw-flash`, `-sw-comm-cus 1` = 1/16 of the
     GPU, `-sw-signal-cost 0`, `-collective-threads 1024`), s = 1 and a plan; partitions from
     the model. Only if the engine runs take at most about twice the 4-CU runs' wall time
     (30-36 s in Step 9).

  Result: stopped after the engine runs (both verify). Wall time 135.4 s (s = 1) and
  162.7 s (plan) against 35.8 s and 30.3 s for the 4-CU pair in Step 9 (3.8x and 5.4x), so
  `fo_s1` and `fo_plan` were not run. Windows: s = 1 719.9 us (model 640.2, -11.1%), plan
  4,2,2,2,2,2,2,1,1,1,1,1,1,2,2,4 549.1 us (model 517.0, -5.8%): -23.7%. Achieved f 0.489.
  The cold first round is slower with 16 CUs on one L2 (261.7 us per work-group against
  172 at 4 CUs; later rounds 148.9 against 137), so the s = 1 GEMM takes 462.3 us against
  360 at 4 CUs, and here the plan's GEMM is shorter (386.5 us): unlike at 4 CUs, part of the
  gain is the GEMM's own (`model.txt`).

  Step 10b (same settings and binary; predictions `predictions_10b.txt` from the Step 9
  software model with its saved constants, not refitted):
  3. `fo_s1`, `fo_plan`: FlashOverlap-style, `-sw-comm-cus 1`, s = 1 and the engine's plan
     4,2,2,2,2,2,2,1,1,1,1,1,1,2,2,4, sixteen 32-row groups (the model's partition for both).
  4. `wab_s1`, `wab_plan`: engine s = 1 and plan with `-warmup -warmup-skip-b`.

  Step 10b result (all 4 verify; `model.txt`): FlashOverlap-style s = 1 999.7 us (model
  -3.1%), plan 943.4 us (-4.3%): the plan helps at 1 comm CU of 16 (-5.6%; at 1 of 4 it
  hurt, +3.0%). GEMM inside the software run vs beside the engine: s = 1 506.8 against
  462.3 us (+9.6%), plan 389.4 against 386.5 (+0.8%). Link efficiency of the software path
  0.480 (s = 1) and 0.422 (plan). A and C warm, engine: s = 1 715.3 us (-12.9%), plan
  525.9 us (-2.3%), -26.5%, of which the GEMM part is -10.8% (436.0 -> 358.5 us) and the
  overlap part -15.6%; cold: -23.7% = GEMM -10.5% + overlap -13.2%. Engine plan vs
  software plan, cold: -41.8%. Wall time 213-301 s per run (the warm-up runs about 300 s).
- `pending_experiments.log`: log of `scripts/run_pending_experiments.sh`.
- `gemm_tile_tuning/`: **outdated.** Output of `scripts/tune_gemm_tile.py`, taken before
  commit `3be1037` with the one-workgroup-row-per-launch benchmark (tile size also set the
  launch count) and always on 2 GPUs. Kept for the record only.
- `allreducegemm/`: **outdated.** Metrics from manual `samples/allreducegemm` runs taken with
  the same pre-`3be1037` benchmark.

## MGPUSim v3 issues found (simulator lessons)

Problems in stock MGPUSim v3.0.3 / Akita v3.0.0 (or its benchmarks) that changed results in
this project, in the order found. Fixes are on branch `All-reduce`; new behavior is behind
flags where it would change old results.

1. **L1 vector caches are never invalidated between kernels.** L1s are not coherent and the
   dispatcher's CU rotation continues across kernels, so a kernel could read stale lines
   another kernel's writes made obsolete (wrong 4-GPU overlap results). Fix: the CP
   invalidates every L1V at each launch and dispatches after all acknowledge (`-l1-acquire`,
   on by default; `8d33e6c`).
2. **mccl collectives ran on one GPU.** `createCommandQueues` created every communicator's
   queue on the last-selected GPU, so all push/reduce kernels of Ring All-Reduce and
   Broadcast ran there (`141ff40`).
3. **Upstream `mmmKernel_local` indexes A by the local Y id,** correct only for a single
   work-group row; any multi-row launch reads wrong rows of A. Patched binary in
   `benchmarks/allreducegemm` (`3be1037`); the copy in `benchmarks/amdappsdk/` still has it.
4. **Memory requests carry PID 0 after address translation.** The L2 caches hold every
   line under PID 0, so a component issuing physical-address requests with the process's
   PID misses the dirty lines and reads stale DRAM data (the engine first read zeros).
   Anything below the translators must use PID 0 (`fda18d0`).
5. **The multi-GPU fabric is one shared switch with one endpoint per GPU.** A GPU's traffic
   to different peers shares one port, so head-of-line blocking serializes its links; there
   is no per-peer link as on MI300X. Fix: `-peer-mesh`, one two-GPU network per pair, with
   per-peer RDMA routes and engine links (`0e73c0c`).
6. **Over-occupancy.** The dispatcher fills a CU up to the VGPR limit (12 GEMM work-groups),
   which oversubscribes it: capping at 8 makes the GEMM 19-23% faster. Not a bug, but the
   default is not the fastest configuration; results use `-gemm-wg-cap 8` (`0e73c0c`,
   `818f115`).
7. **The CP interleaves concurrent kernels' work-groups, whatever the launch order.** Each
   of its 8 dispatchers waits a few cycles after every dispatch and the others fill CU
   slots meanwhile; a 9th kernel waits for a freed dispatcher and, taking the lowest index,
   gets the highest priority. Fix: `-ordered-dispatch` (a kernel dispatches only after every
   earlier one has dispatched all its work-groups) and `-cp-dispatchers` (`13bae25`).
8. **The driver never clears L2-dirty flags, so argument copies flush L2.** Every launch
   marks all buffers L2-dirty, nothing marks them clean, and the overlap test counts
   touching ranges, so after the first launch most host copies, including each launch's
   kernel-argument copies, first flush L2 (write back; lines stay valid). Queues whose
   copies flush fall behind, so launches on different queues reach the CP out of order.
   Fix: `-ordered-dispatch` also makes the driver send each GPU's launches in enqueue order
   (`f23ed3f`); the flush itself is unchanged, since fixing it would change every old result.
9. **An L1 invalidate during in-flight fills corrupts the L1.** The vector L1 is Akita's
   write-around cache; its `GL0InvalidateReq` handler replaced the whole tag directory at
   once. If a fill was in flight, another miss could take that block and the old fill
   landed under the new tag, so a later hit returned another line's data. Only kernels
   that start while other kernels' loads are in flight on the same GPU trigger it (the CP
   invalidates every L1 at each launch, `-l1-acquire`); the FlashOverlap-style path did, and
   a GEMM work-group read two wrong lines of A. Fix: invalidate the blocks in place, so a
   block with a fill or a read in flight keeps its storage until done (Step 9; making the
   invalidate wait for in-flight transactions instead stalled every CU at every comm
   launch, +8% on `cal_rows`). Every earlier run verified; the effect on their timing is
   in Step 9 `summary.txt`, section 5.
10. **`rand.Seed` does nothing since Go 1.24,** so `allreducegemm`'s A and B came from an
   unseeded source and differed every run. Timing does not depend on the values, so no
   window changed, but verification data were not reproducible. Fix: a seeded source of
   its own (Step 9).
11. Smaller ones: Akita's link bandwidth is whole bytes per cycle at the network clock, so
   low or fractional bandwidths need a lower network frequency; component name indices must
   be integers; MGPUSim v3 loads only code object v2; `total_time` varies by about 4 ns
   between identical runs (the measured windows are exact).
