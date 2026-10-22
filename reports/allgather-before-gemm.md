# All-gather before the GEMM on the engines: design note

Status: design and feasibility only (Step 9, item 4). No code. Numbers come from the queue
model `scripts/step9_allgather.py`; its output is in `results/step9_baselines/allgather.txt`.

## What is gathered

A tensor-parallel MLP or attention block has two GEMMs. Steps 1-8 built the second one: a
row-parallel GEMM followed by an all-reduce or reduce-scatter of C. Under sequence
parallelism, the first one starts with an all-gather. Each GPU holds M/p rows of the
activations A, with full K. Every GPU needs all M rows to compute C = A × B_local, where
B_local is its column shard of the weights. The gather is on the GEMM's input, so a
work-group can start only once its A rows have arrived. On the output side, by contrast, the
GEMM never waited for communication.

Simulated shape: the f≈0.5 GEMM per GPU (256×256 output, K 256, 4 CUs, tileSize 8, cap 8).
Each GPU owns 256/p rows of A, so 4 rows of tiles at 2 GPUs and 1 at 8 GPUs.

## Mechanism

1. **Engine push.** At Arm, each engine reads its own A rows from local memory and writes
   them into every peer's A buffer. The read and write paths are the ones the reduce-scatter
   engine already uses; what is new is a source region that is not C and no reduction step.
   The send order is a list, so the schedule can choose it. One message is one row of tiles
   (32 rows) × one K chunk. The per-peer queues (`-engine-peer-queues`) carry the peers in
   parallel.
2. **Arrival signal.** The receiving engine raises a signal on the CP signal board (built in
   this step for the FlashOverlap baseline) once a message's last write is acknowledged by
   L2. Example name: `gpu3/a/row5/chunk1`.
3. **GEMM held per row.** The GEMM launches one kernel per row of tiles, or per (row, K
   chunk) for a split row. Each launch carries `WaitSignals` for its rows of A. The CP holds
   it, as it already holds the FlashOverlap comm kernels. Under ordered dispatch the
   work-groups then run in launch order. Local rows are launched first. Remote rows follow
   in expected arrival order, so a held launch blocks only launches that would also have to
   wait.
4. **Visibility.** A released kernel's L1 acquire invalidates the L1s before its loads. The
   engine's writes are in L2, so the kernel reads the new A. This depends on the L1
   invalidate fix from this step: before it, an invalidate during in-flight fills could
   return wrong lines.
5. **Split rows.** A row split s ways reads only its K chunk of A. It can start when that
   chunk's message has arrived, before the rest of the row has. Its s partials must be added
   at the end. That costs one local pass over the row, and the engine's pre-sum read path can
   do it without a kernel. The model does not include this cost.

## Schedule choices

- **Local rows first.** This is always right: they are ready at the launch.
- **Send order (ring vs direct).** On the per-pair mesh the direct order uses every link. A
  ring all-gather uses one link per direction, so the shard from distance d arrives after d
  shard times. In the model the ring is 3.8× slower at 8 GPUs (2,456 µs against 640 µs), so
  a ring order is only worth considering on a ring fabric.
- **Which rows to split.** At 2 GPUs only the last remote row needs splitting; earlier rows
  arrive while the GEMM is still busy with the local ones. At 8 GPUs every GPU gets its 7
  remote rows at once, at the end of the transfer, one row per link. Splitting them all 4
  ways lets every K chunk start a quarter of the way through the transfer. This is the input
  side of the same co-design as Steps 5-8: the split factor sets the message granularity the
  GEMM can use.

## Queue model at f 0.5

The knobs follow the Step 6 rule. The bytes per link are one GPU's A shard, M/p × K × 4,
which is the same size as C/p, so the knobs are the reduce-scatter ones at f 0.5. The GEMM
alone takes 355 µs in the model.

| | 2 GPUs | 8 GPUs |
|---|---|---|
| Link knobs (bw GB/s, latency ns, overhead ns) | 0.487, 707, 3403 | 0.122, 2827, 13611 |
| Gather, then GEMM | 696 µs | 709 µs |
| Per-row overlap, s = 1, local first | 422 µs (-39%) | 640 µs (-10%) |
| + best split plan | 397 µs (-6% vs s = 1), plan 1,1,1,1,1,1,1,4 | 470 µs (-27% vs s = 1), plan 1,4,4,4,4,4,4,4 |

At 8 GPUs, per-row overlap without splits hides only the one local row (8 of 64 tiles). The
remaining 56 tiles wait for the whole transfer. With K chunks the GEMM starts on every
remote row a quarter of the way through the transfer. The best plan is then 32% above the
GEMM alone, against 80% for s = 1.

## Code size (estimate)

| Part | Lines (Go) |
|---|---|
| Engine: push mode (source region, send-order list, no owner reduction), arrival signals, tests | 300-400 |
| Benchmark: A shards, per-row / per-chunk launches with `WaitSignals`, partial sum of split rows, verification | 250-300 |
| Sample flags (`-allgather-a`, send order), model check against runs | 50 + Python 100 |
| **Total** | **about 600-800 Go lines plus tests** |

The signal board, held launches, ordered dispatch and per-peer queues already exist, so
nothing in the CP or the dispatcher needs to change.

## Verdict

**Worth building, as the next batch.**

- **The gain is largest where it matters.** At 8 GPUs (TP = 8, the realistic case) the model
  puts split plans 27% ahead of per-row overlap. That is a larger margin than the plan gets
  on the reduce side at 8 GPUs (-13% measured). It comes from the same mechanism: the split
  factor sets how early a work-group can use a message.
- **It is cheap now.** Most of the cost was the device-side waiting, and that was built and
  verified in this step.
- **It completes the layer.** With it, the whole tensor-parallel block (all-gather, GEMM 1,
  GEMM 2, reduce-scatter) runs on one engine, and the paper can report a full-layer result
  rather than half of one.

Risks:

- The model leaves out the partial sum of split rows.
- With 7 rows × 4 chunks, a GPU has 28 launches held per peer group, so it needs about 40
  dispatchers.
- The 8-GPU gain depends on all remote rows arriving together. A fabric with less bandwidth
  per link than the mesh would change that.

Check those three first, with the 8-GPU s = 1 and plan runs as the first runs of the batch.
