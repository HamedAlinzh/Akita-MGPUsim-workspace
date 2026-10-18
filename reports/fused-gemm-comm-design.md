# Fused GEMM + All-Reduce in MGPUSim: design note

Status: option (b) is implemented (Steps 1–2); see "As built" at the end. The proposal
below is kept as written at `485f19d`.

## Why

The overlap study banded the output matrix C so that band *r*'s All-Reduce could run
while band *r*+1's GEMM computed. Banding never beat a single GEMM launch, for three
measured reasons:

| Cost of banding | Evidence (256×256×1024, 2 GPUs, tileSize 8) |
|---|---|
| Each band is a full ring All-Reduce with its fixed launch cost | All-Reduce 255 → 660 µs for 1 → 4 bands |
| Each band is a short GEMM launch with too few work-groups | GEMM 226 → 607 µs at `-h2d-cycles 1450`; a 64-row band at tileSize 8 is 16 work-groups for 64 CUs |
| Collective kernels compete with the GEMM for CUs | At `-collective-threads 16384`, overlapped All-Reduce gained 53 µs and the exposed GEMM lost 53 µs |

A fused design avoids all three. The GEMM stays **one full-size launch**, each
work-group's finished output tile leaves for the peer GPU **as soon as it is written**,
and no host launch or compute-unit time is spent per tile.

## Requirement

- One GEMM launch per GPU covering all of M, with unchanged occupancy.
- Each finished output tile (4T×4T floats for a T×T work-group) is sent to its peer
  without a host launch per tile.
- The receiver reduces arriving tiles, and the host learns when the whole All-Reduce is
  complete.
- Results verify against the CPU reference, as every run does now.

## Option (a): a toolchain that still emits code object v2

MGPUSim v3 loads only code object v2 (an `amd_kernel_code_t` header in front of the
code). Every kernel in the repository is v2 (ELF ABI version 0); they were built with
clang 4.0 (the AMD APP SDK matmul), clang 11 (`fir`) and ROCm's clang 12 (the mccl
collectives). Upstream LLVM removed code object v2 in the LLVM 17 series (September
2023), so **any clang up to 16 still accepts `-mcode-object-version=2`**.

**What it takes to install.** Ubuntu 24.04's own archive still has `clang-16`,
`lld-16` and `libclc-16-dev` (also 14 and 15). They install without sudo into a
throwaway container from the `ubuntu:24.04` image already on this machine. Sketch,
not yet run:

```
docker run --rm -v "$PWD":/w -w /w ubuntu:24.04 bash -c \
  'apt-get update && apt-get install -y clang-16 lld-16 libclc-16-dev && \
   clang-16 -x cl -cl-std=CL1.2 -target amdgcn-amd-amdhsa -mcpu=gfx803 \
     -mcode-object-version=2 -nogpulib <kernel>.cl -o <kernel>.hsaco'
```

The OpenCL built-ins the GEMM kernel uses (`get_global_id`, `get_local_id`,
`get_local_size`, `get_global_size`, `barrier`) come from `libclc` or from a ~20-line
header that maps them to clang's AMDGPU intrinsics. No ROCm install is needed.
Estimated setup: half a day to a day, including checking that MGPUSim loads the output.

**What the fused kernel would need.** The GEMM epilogue writes its tile into the peer's
staging buffer (a remote store through RDMA, as mccl's push kernel already does) and
then publishes a ready flag. The receiver needs a consumer, either a persistent kernel
spinning on the flags or a second kernel per tile, which reduces the tile into C and
sends it back.

**Problems with (a):**
- **It reintroduces compute-unit contention.** The consumer runs on the same CUs as the
  GEMM, which is the effect that cancelled overlap's gain at 16,384 threads.
- **It needs a cross-GPU memory model that MGPUSim may not implement.** Flag
  publication needs release ordering across RDMA, and polling needs loads that bypass the
  non-coherent L1 (GLC). Whether MGPUSim's RDMA path and write-around L1 honour these is
  unverified. Stage 1 showed how silently L1 staleness corrupts results.
- **The baseline moves.** A recompiled GEMM has a different instruction schedule from
  today's patched binary. Every comparison needs a fresh baseline built with the same
  toolchain.
- **Newer codegen may hit unimplemented instructions.** MGPUSim implements a subset of
  GCN3; LLVM 16 may emit instructions the emulator lacks, which would panic. Kernels
  from clang 11/12 work, so the risk is moderate.

## Option (b): a simulator mechanism that sends tiles when they are written

The existing GEMM binary runs unchanged. The host programs each GPU, before the launch,
with the layout of C and a send plan. Hardware then tracks the stores into C, and sends,
reduces and gathers each tile when it is complete. This is the approach of T3 (Pati et
al., ASPLOS 2024): a tracker watches the producer's stores and triggers pre-programmed
DMA transfers, so communication uses no GPU compute, and reductions happen close to
memory.

### Where it sits

```
   CU ─► L1V (write-around) ─► L2 bank (write-back) ─► DRAM
                               ▲   │
                 TileTracker ──┘   │ (port hooks: WriteReq in, WriteDoneRsp out)
                       │           │
                       ▼           │
                  CommEngine ──────┘ reads/writes through L2;
                       │             remote addresses go to the RDMA engine
                       └─────────► RDMA ─► peer RDMA ─► peer L2
```

**The tracker belongs at L2, not at the memory controller as in T3.** In this model
every store reaches L2, because L1 is write-around. But L2 is write-back, so the DRAM
controller sees a store only when the line is evicted or flushed, which may be long after
the tile is finished. Tracking at L2 sees each tile complete when it really completes.

**TileTracker** (one per GPU, no change to cache code). It is an Akita hook on each L2
bank's top port.
1. It is programmed with C's base address, row pitch (N×4 bytes), and the tile
   height and width (4T×4T).
2. On `HookPosPortMsgRecvd` of a `WriteReq` inside the region, it records the request ID
   → tile index and the number of bytes written (from the dirty mask).
3. On `HookPosPortMsgSend` of the matching `WriteDoneRsp`, it adds those bytes to the
   tile's count.
4. A tile is complete when its count reaches 4T×4T×4 bytes. That relies on
   `mmmKernel_local` writing every output element exactly once, which it does: each
   work-item stores four `float4`s at the end.
5. Counting acknowledgements rather than arrivals matters: the L2 locks a block while
   writing it, so a later read of that line sees the new data.

**CommEngine** (one per GPU, a new `TickingComponent`, modeled as a DMA engine with an
adder).
- It plugs into the GPU's L1↔L2 connection exactly like an L1. It reads and writes
  local memory through L2, never through any L1, so the Stage 1 staleness cannot occur.
- Remote addresses route to the RDMA engine through the same low-module finder the L1s
  use, so sending a tile is ordinary remote writes.

### How a tile is reduced (2 GPUs)

A reduce-scatter followed by an all-gather, at tile granularity:
1. **Ownership.** Each tile has an owner GPU, for example by tile-index parity.
2. **Send.** When a GPU completes a tile it does not own, its CommEngine reads the tile
   from local L2 and writes it into the owner's staging buffer. Only after every one of
   those writes is acknowledged does it write the tile's **ready flag** at the owner. That
   ordering is the release.
3. **Arrival.** The owner's TileTracker also watches the staging flags. Flag writes
   arrive from RDMA into L2's top port like any other store, so the same hook sees them.
4. **Reduce.** When the owner's own copy of the tile is complete and the peer's flag has
   arrived, its CommEngine reads both and adds them. It writes the sum into local C, then
   writes the sum into the peer's C, followed by a "final" flag.
5. **Cost model.** The add costs a configurable number of cycles per 64-byte line, a
   stand-in for T3's near-memory compute.

This produces the sum directly, so the benchmark's mean-then-multiply step is skipped in
fused mode.

### How completion is detected

- Each CommEngine counts the tiles it owns that are reduced and sent, and the tiles it
  doesn't own whose final copy has arrived. It is done when both counts are complete.
- The host sees completion through a new driver command, `EnqueueTileReduceWait`. It
  stays at the head of the queue until every GPU's CP reports that its CommEngine is done.
- The host flow becomes: program the trackers and engines → launch the GEMM on every
  GPU → enqueue the wait → drain. That's one launch per GPU and no per-tile host work.

### Programming interface

- Driver: `ArmTileReduce(ctx, gpuIDs, C ptr per GPU, M, N, tileRows, tileCols,
  staging ptr per GPU)`. It sends the configuration through each CP to its tracker and
  engine.
- Benchmark: `-fused` on `allreducegemm`, which replaces the band loop with arm → launch
  → wait.

## Smallest first experiment

- **Correctness first:** 2 GPUs, **128×128×256** (K per GPU 128), tileSize 8. That's 16
  output tiles of 32×32 (4 KB each), and 8 sends in each direction. It passes when
  `-verify` matches the CPU sum.
- A tracker-only spike comes before that: a 1-GPU GEMM with the tracker armed, checking
  that every tile reports complete exactly once and none before its last store. This
  de-risks the hook approach in about half a day.
- **Then timing on the balanced shape, 256×256×1024, 2 GPUs, tileSize 8.**

**Baseline:** the same shape as one GEMM launch followed by one Ring All-Reduce. Use the
current best (`-collective-threads 16384`, window **433 µs**) and the default (1024
threads, **520 µs**), both with the L1 acquire on. The ideal fused window is about
max(GEMM, communication) plus the last tile's send, reduce and gather, roughly 265 µs plus
a tail.

**Follow-up:** sweep tileSize 4, 8 and 16. In the fused design the tile is the message
(1 KB, 4 KB, 16 KB), which is the "tile size becomes message size" direction.

## Effort and risks

| | (a) Toolchain + hand-written fused kernels | (b) Tracker + CommEngine in the simulator |
|---|---|---|
| Setup | 0.5–1 day (container, built-in shim, loader check) | none |
| Core work | 3–5 days: fused GEMM epilogue, consumer kernel, flag protocol | 1–1.5 days tracker; 3–4 days CommEngine; 1–2 days driver/CP plumbing and wiring |
| Tests | 1–2 days | 1–2 days (tracker unit test, 2-GPU correctness test) |
| Total to first correct 2-GPU result | **2–3+ weeks, highly uncertain** | **about 1.5–2 weeks** |
| Main risks | cross-GPU flags and GLC loads may be unsupported over RDMA; consumer kernel steals CUs; recompiled GEMM moves the baseline; newer codegen may hit unimplemented instructions | port-hook semantics under the parallel engine (use the serial engine first); exact `WriteDoneRsp` ordering in the write-back L2 (the spike checks it); CommEngine traffic contends for L2 bandwidth (realistic, but must be reported); it models hardware that doesn't exist yet, so the add cost and engine bandwidth are assumptions that need sensitivity runs |
| Generality | any kernel you can write | any producer kernel that writes each output element exactly once, in a layout the host can describe |

## Recommendation

**Option (b).** It is the only option that keeps the existing GEMM binary, and with it
the baseline every earlier result was measured against. It also adds no compute-unit
work during the GEMM, the effect that cancelled overlap in Step 1. And it reuses parts
that already exist: Akita port hooks, the L1↔L2 connection, and the RDMA path. Its
risks are local to new components and testable in small steps, beginning with the
half-day tracker spike.

Option (a) is cheap to set up, and worth keeping in mind for writing new kernels later.
But as the vehicle for this experiment, it changes too much at once: the kernel, the
memory-model assumptions, and the baseline.

## As built (Steps 1–2)

Differences from the proposal above:
- **Tracker** (`timing/tiletracker`) as proposed, plus listeners and an ignored source port
  so the engine's own writes into C don't count.
- **Engine-to-engine messages instead of RDMA writes and ready flags.** Each engine
  (`timing/commengine`) has its own port on a separate GPU-to-GPU network (the peer fabric,
  `-link-bw`, `-link-latency`); a tile group travels as one message, so there is no flag
  protocol and no remote store ordering to get right.
- **Ownership by group**, not tile parity: `-tiles-per-msg` consecutive tiles form a group,
  owned by GPU g mod p.
- **Completion** through `commengine.Session.Wait` (`driver.WaitFor`), not a new driver
  command: the host blocks until every engine reports done.
- **Per-message cost** `-msg-overhead`, serialized per engine, overlapped with the previous
  message's transmission. Reduce cost: 1 cycle per 64-byte line per partial.
- **Modes**: `tile` (send as groups finish) and `bulk` (same engine, after the whole GEMM);
  `-reduce-scatter` stops after the owner's sum.

## Sources

- T3: S. Pati, S. Aga, M. Islam, N. Jayasena, M. D. Sinclair, "T3: Transparent Tracking
  & Triggering for Fine-grained Overlap of Compute & Collectives," ASPLOS 2024.
  <https://arxiv.org/abs/2401.16677>
- Removal of code object v2 from LLVM ("[AMDGPU] Remove Code Object V2", PR #65715,
  September 2023, LLVM 17 series).
  <https://repo.hca.bsc.es/gitlab/rferrer/llvm-epi/-/commit/fe2f67e4ba62ce9f0c3a9adcff592a77606e5228>
- Measurements: `reports/gemm-allreduce-overlap.html`, and `results/stage2_h2d/` and
  `results/step1_threads/` on this branch.
