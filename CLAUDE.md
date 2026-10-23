# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

```
go.work, go.work.sum   Go workspace wiring the two simulator modules together
simulator/             Simulator source code (the two Go modules below)
scripts/               Experiment drivers (sweep_overlap.py, ratio_model.py, run_pending_experiments.sh, tune_gemm_tile.py)
results/               Raw simulation output (metrics.csv, stdout); gitignored except README
diagrams/              Plots/figures derived from results/; tracked in git
reports/               Standalone HTML write-ups of experiments, plus design notes (*.md)
```

Keep this separation: generated data goes in `results/<experiment>/`, never inside
`simulator/`; derived figures go in `diagrams/`; write-ups go in `reports/`.

The Go workspace (`go.work`, Go 1.26) contains two modules under `simulator/`, developed together:

- `akita-3.0.0` — module `github.com/sarchlab/akita/v3`. A general-purpose discrete-event
  simulation *engine* (analogous to a game engine), not a simulator itself. Provides the core
  event loop, component/port/connection model, generic memory components (`mem/`), and
  network-on-chip components (`noc/`).
- `mgpusim-3.0.3` — module `github.com/sarchlab/mgpusim/v3`. A GPU simulator built on top of
  Akita that models GPUs running the AMD GCN3 instruction set, including multi-GPU simulation.

`mgpusim` depends on `akita` via the module path `github.com/sarchlab/akita/v3`; the `go.work`
file wires local `./simulator/akita-3.0.0` in place of a tagged release so cross-module changes are picked
up immediately without touching `go.mod`. Each module also has commented-out `replace` directives
in its `go.mod` for standalone (non-workspace) development.

## Keep in mind
Write the progress of each step in the CLAUD.md so for the next steps you can understand what is happening even if the session was cleared.

## Common commands

Run from inside the relevant module directory (`simulator/akita-3.0.0` or
`simulator/mgpusim-3.0.3`) unless noted.

```bash
# Build everything in a module
go build ./...

# Run the full test suite (Ginkgo BDD framework, used throughout both modules)
go install github.com/onsi/ginkgo/v2/ginkgo
ginkgo -r

# Run tests for a single package with the standard Go test runner
go test ./sim/...              # e.g. from akita-3.0.0
go test ./timing/cu/...        # e.g. from mgpusim-3.0.3

# Run a single Ginkgo spec by name (focus), from inside the package directory
ginkgo -focus "should allow push and pop"

# Lint (config in .golangci.yml, one per module)
golangci-lint run ./...

# Regenerate mocks (gomock) declared via //go:generate directives ahead of tests
go generate ./...
```

`run_before_merge.sh` in each module chains these steps (`go get -u`, `go mod tidy`,
`go generate`, `go build`, `golangci-lint run`, `ginkgo -r`) — use it as the reference for what CI
expects, but don't run the `go get -u`/`go mod tidy` steps casually since they upgrade
dependencies.

mgpusim also has extra, slower CI-only checks that are not part of routine iteration:
- `tests/deterministic/test.py` — verifies repeated runs produce identical results.
- `tests/acceptance/` — builds `./acceptance` and runs it against real GPU configs
  (`-num-gpu=1`, etc.) to check end-to-end simulation correctness.

Test files follow the `*_test.go` convention with one `Describe`/`It` (Ginkgo/Gomega) spec file
per source file, plus one `<package>_suite_test.go` per package that calls `RunSpecs`. Mocks are
generated per-package into `mock_*_test.go` files via `//go:generate mockgen` comments at the top
of suite files — check those directives before hand-writing a mock or changing an interface that
mocks depend on, since the mocks need regenerating (`go generate ./...`) after interface changes.

## Akita engine architecture (`akita-3.0.0/sim`)

Akita simulates hardware as a graph of **components** connected by **ports** and
**connections**, driven by a discrete-event **engine**:

- `Engine` (`engine.go`) processes a time-ordered queue of `Event`s (`event.go`). Two
  implementations exist: `SerialEngine` (single-threaded) and `ParallelEngine`
  (`parallelengine.go`, groups same-timestamp events for concurrent dispatch). Time is
  `VTimeInSec` (simulated seconds, a float64), independent of wall-clock time.
- `Component` (`component.go`) is anything simulated (a cache, a compute unit, a GPU). Most
  components embed `ComponentBase` for naming/hooking/port-ownership, and most are
  `TickingComponent`s (`ticker.go`): instead of hand-scheduling events, you implement a
  `Ticker.Tick(now) bool` method describing one cycle's worth of state update, and the
  `TickScheduler` handles re-scheduling itself every cycle (`TickLater`) as long as
  `Tick` reports it made progress, and stops (event-driven, not polling) once idle until a port
  receives/frees data (`NotifyRecv`/`NotifyPortFree`) wakes it again.
- `Port` (`port.go`) is a component's message inbox/outbox with a bounded `Buffer`. Components
  only interact with their own ports (`Send`/`Retrieve`/`Peek`); the wiring between ports is a
  `Connection` (`connection.go`, e.g. `DirectConnection`) or the more elaborate NoC components
  in `noc/`.
- `Msg` (`msg.go`) is the payload passed between ports; every message embeds `MsgMeta`
  (src/dst ports, timestamps, ID). `GeneralRsp`/`ControlMsg` and their builder types are common
  reusable message shapes.
- Cross-cutting concerns (tracing, monitoring, logging) are implemented via a `Hookable`/`Hook`
  system (`hook.go`) rather than modifying component logic — see `tracing/` for the tracer
  hooks used to collect per-task timing data, and `analysis/` for buffer/port utilization
  analyzers built the same way.
- `mem/` and `noc/` are first-party, reusable components (caches, TLB, DRAM controller,
  switches) built on the same Component/Port/Connection model, meant to be composed into a
  larger simulated system rather than modified in place.
- `daisen/` is Akita's separate web-based visualization tool (own Node/JS build); `monitoring/`
  is the Real-Time Monitoring (RTM) web tool. Both are auxiliary to the simulation engine.

## mgpusim architecture

mgpusim assembles a full GPU (and multi-GPU) system out of Akita components:

- `insts/` — GCN3 ISA definitions and disassembler (`insts/gcn3disassembler`); decodes AMD
  GCN3 machine code (`.hsaco` binaries) into instruction objects.
- `emu/` — the *functional* (untimed) emulator: ALU implementations per instruction category
  (`alu*.go`), used to compute correct execution results independent of timing accuracy.
- `timing/` — the *timing* (cycle-accurate-ish) model, one subpackage per hardware unit:
  - `timing/cu` — Compute Unit: fetch/decode/issue arbiters, SIMD units, scalar/vector memory
    units, LDS unit, register file, wavefront pool/scheduler. This is the most complex package
    and the one most likely to need changes when modeling new GPU microarchitecture behavior.
  - `timing/cp` — Command Processor: dispatches kernels/wavefronts to compute units, handles
    DMA.
  - `timing/rdma`, `timing/pagemigrationcontroller` — multi-GPU memory movement components.
  - `timing/rob`, `timing/wavefront` — supporting structures (reorder buffer, wavefront state).
- `driver/` — the host-side driver API (`driver/api.go`) that benchmarks call into (memory
  alloc/copy, kernel launch), analogous to a real GPU driver/runtime; `driver/internal` handles
  device memory allocation bookkeeping.
- `kernels/` — kernel/grid/workgroup/wavefront data structures shared between `emu` and
  `timing`.
- `protocol/` — message/request types passed between the timing components.
- `benchmarks/` — GPU kernel benchmarks (grouped by suite: `amdappsdk`, `dnnmark`,
  `heteromark`, `polybench`, `rodinia`, `shoc`), each exposing a `NewBenchmark(driver)` used by a
  sample.
- `samples/` — one `main` package per runnable benchmark (e.g. `samples/fir`), plus shared
  scaffolding in `samples/runner/`:
  - `samples/runner.Runner` — parses common CLI flags, builds a `Platform` (GPUs + driver +
    engine), registers benchmarks, and runs the simulation. Nearly every sample's `main.go`
    follows the same three-line pattern: build a `Runner`, construct a benchmark from
    `runner.Driver()`, `runner.AddBenchmark(...)`, then `runner.Run()`.
  - `samples/runner.Platform`/`GPU` — describes the assembled hardware graph (CommandProcessor,
    RDMA engine, page migration controller, CUs, caches, TLBs, memory controllers) that
    `r9nanobuilder.go`/`emugpubuilder.go` construct.
  - The "How to Prepare Your Own Experiment" section of `mgpusim-3.0.3/README.md` documents the
    expected workflow for building a new experiment outside this repo: copy `samples/experiment`
    into a new Go module, edit `main.go`/`runner.go`/`platform.go`/`r9nano.go`/`shaderarray.go` to
    customize the hardware/benchmark configuration, and modify or add components as needed.
- `accelsim_tracing/`, `server/`, `bitops/` — supporting/auxiliary packages (trace generation
  for accelsim interop, a simulation server mode, bit manipulation helpers).
- `tests/acceptance` and `tests/deterministic` — end-to-end and determinism regression tests
  (see Common Commands above), separate from the per-package Ginkgo unit tests.

When adding a new GPU component or modifying timing behavior, follow the existing pattern: build
it as an Akita `Component`/`TickingComponent`, wire it up via ports/connections, and expose a
builder (see `*builder.go` files, e.g. `timing/cu/cubuilder.go`, `timing/cp/builder.go`) that
`samples/runner` or a custom experiment can call to assemble it into a `Platform`.

## Changes on branch `All-reduce`

The simulator here is no longer stock MGPUSim v3.0.3. Know these before reading results or
changing timing code. Commit hashes are on branch `All-reduce`.

- **Patched GEMM kernel** (`3be1037`). `benchmarks/allreducegemm/kernels.hsaco` is the upstream
  `mmmKernel_local` binary with A indexed by the global Y id instead of `lIdY`, so one launch can
  cover all of M. It is produced by `benchmarks/allreducegemm/native/patch_gidy.py` from the
  upstream binary; don't hand-edit it. The upstream copy in `benchmarks/amdappsdk/` still has the
  bug.
- **mccl ring collectives** (`141ff40`). `createCommandQueues` now selects each communicator's GPU
  before creating its queue; before, every push/reduce kernel of Ring All-Reduce and Ring
  Broadcast ran on the last-selected GPU. `mccl.CollectiveThreads` (`485f19d`, flag
  `-collective-threads`, default 1024) sets the threads per collective kernel.
- **L1 acquire at kernel dispatch** (`8d33e6c`). Per-CU L1 vector caches are not coherent, and
  the dispatcher's CU rotation continues across kernels, so a kernel could read stale L1 lines.
  The CP now sends a `GL0InvalidateReq` to every L1V (through `L1VInvalidatePorts` on each L1V's
  top-port connection) and dispatches once all acknowledge. On by default; `-l1-acquire=false`
  restores old behavior. Regression test: `benchmarks/mccl/l1acquire_test.go`.
- **Runner flags added:** `-l1-acquire`, `-h2d-cycles` (`156e143`; fixed driver latency per
  host-to-device copy, default 14500; each kernel launch makes three copies, about 22.75 µs).
- **`allreducegemm`** gained `-band-rows`, `-overlap`, `-allreduce-chunk`, `-per-band-scratch`,
  `-collective-threads`. It logs `allreducegemm phases: window=… gemm_wait=… allreduce=…`, and
  `Verify` prints a mismatch summary by band and ring chunk before panicking.
- **Tile tracker** (Step 1). `timing/tiletracker` hooks every L2 bank's top port (attached
  to each GPU as `GPU.TileTracker`, idle until `Arm`). Counts bytes of `WriteReq`s into a
  region by request ID and credits them on the matching `WriteDoneRsp`; L2 sees physical
  addresses, so `Arm` takes a physical→virtual page map (`driver.PhysicalAddress`).
  `allreducegemm` gained `TileSize` (0 = `const tileSize`, which `tune_gemm_tile.py` still
  patches), `OnPartialC` and `ResultC`. Platform builder gained `WithNumCUPerGPU`.
- **LDS occupancy** (`-lds-occupancy`, default off). The dispatcher can count a work-group's
  dynamic LDS (`driver.LocalPtr` arguments, in the packet's `GroupSegmentSize`) instead of
  only the code object's static LDS. For `mmmKernel_local` it changes nothing: 74 VGPRs
  already cap a CU at 12 work-groups (tileSize 4, 8) or 3 (tileSize 16), below the LDS limit.
  `-cus-per-gpu` (default 64) sets CUs per GPU.
- **Peer fabric** (`-link-bw` GB/s per direction, `-link-latency` one-way ns). Setting either
  moves every GPU's RDMA port off the shared PCIe network onto its own one-switch network
  (`createPeerFabric` in `timingplatform.go`); host traffic stays on PCIe. Flit 16 B, clock =
  bandwidth / 16, latency rounded to whole cycles of that clock (logged at startup). With 2
  GPUs, one switch input pipeline = the one-way latency. Defaults reproduce old results
  exactly.
- **Communication engine** (Step 2, `timing/commengine`). One per GPU when `-engine-mode`
  is `tile` or `bulk`: plugs into the L1→L2 connection for local reads/writes and onto the
  peer fabric (created automatically) for engine-to-engine `TileMsg`s. Triggered by the
  tile tracker; groups of `-tiles-per-msg` consecutive tiles, group g owned by GPU g mod p;
  non-owners send partials, the owner adds them and writes C, then (all-reduce) sends the sum
  back. `-msg-overhead` ns is a per-message cost serialized per engine but overlapped with
  earlier transmissions. `-reduce-scatter` selects reduce-scatter (engine) or
  `mccl.ReduceScatterRing` (without an engine). Host side: `commengine.Session` (Arm,
  Wait via `driver.WaitFor`), `allreducegemm.Benchmark.Comm`. `-tile-size` sets T.
  Gotcha: behind the address translators all memory requests carry PID 0, so anything that
  issues physical-address requests to L2 must use PID 0 or it misses dirty lines.
- **Near-memory placement** (Step 3, `-placement=near-memory`). Owned tiles expect p writes
  per byte and every store into them is an add: the tracker rewrites each store's data to
  the running sum at the L2 input (`Region.Accumulate`/`Expected`, `tiletracker.Update`
  tags peer updates). The owner's engine reads only the reduced tile. Default
  `-placement=owner`.
- **Compute-speed knob** `-valu-cycles-scale` (default 1): multiplies every VALU
  instruction's cycles with a fractional carry; memory, LDS and scalar timing unchanged.
  Matrix-core curve: tileSize 4 (= real 128×128) uses 0.647 so its GEMM is 2.77× tileSize 8
  (= real 256×256), from HipKittens MI355X data (α = 1.47); flatter 0.451 (2.0×), steeper
  0.96 (4.0×). Natively tileSize 4 is 4.18× slower (16-thread work-groups leave 48 of 64
  lanes idle), tileSize 16 0.97×.
- **Dispatch shaping** (`-wg-cap K`, `-wg-ramp c`, default off; `CUResourceImpl`): cap a CU's
  work-groups in flight, or start the cap at c and raise it by one per completed work-group
  (reset per kernel). Finding: 12 resident GEMM work-groups per CU over-subscribe it; capping
  at 4–8 makes the 384×256 GEMM 19–23% faster as well as finishing the first tile earlier.
- **Per-pair peer mesh** (`-peer-mesh`, default off): one two-GPU network (same as the shared
  switch, `-link-bw`, `-link-latency`) per GPU pair; RDMA gets `PeerRoutes` (a port per peer,
  replies on the arrival port) and the engine `Links`. A single switch with one endpoint per
  GPU serializes a GPU's links (head-of-line blocking), so 4-GPU runs need `-peer-mesh`.
- **Per-kernel hints** (`insts.HsaCo.WGCap`, `.VALUCyclesScale`; 0 = the platform flag).
  `allreducegemm` sets them on its GEMM code objects only: `-gemm-wg-cap`,
  `-gemm-valu-scale`, so collective kernels keep their own settings.
- **Small tiles first** (`-small-rows R`, `-small-tile-size`, `-small-valu-scale`,
  `-small-tiles-per-msg`): a tileSize-4 GEMM over rows [0, R) on its own queue (created
  first: the driver serves queues in creation order) plus the main GEMM over the rest; the
  tracker uses the small tile grid and the engine takes explicit groups
  (`commengine.Options.Groups`). R must be a multiple of the main tile edge (32).
- **Split-K first and last bands** (Step 5; `-splitk-ways s -splitk-rows R`, `-splitk-presum
  on|off`, engines only, owner placement): each band runs as s launches, one per K chunk,
  over A stored as s chunk buffers (M x K/s, since `WidthA` is both K and A's row stride) and
  B offset per chunk; chunk 0 writes C, the others partial buffers. The tracker counts a
  split tile complete after all s writes (`tiletracker.Region.Partials`); the engine reads
  the s copies and adds them before sending (pre-sum) or sends each (`Config.SplitSend`).
  Queues: first-band chunks, main GEMM, last-band chunks, in that order.
- **Split-K plan** (Step 5b; `-splitk-plan 4,4,4,4,1,1,2,2`): a split factor per row of
  tiles, launched top to bottom, each split row as its own s launches, consecutive unsplit
  rows as one; replaces `-splitk-ways/-splitk-rows` (which map to the same band list with
  the same allocation and queue order). `-warmup` runs the GEMM once untimed first;
  `-mem-timeline file` writes L1V/L2 hits and misses, TLB misses and DRAM bytes per 10 us.
- **CP dispatch** (`-cp-dispatchers`, default 8; `-ordered-dispatch`, default off). Without
  ordered dispatch the CP's dispatchers interleave concurrent kernels' work-groups (each
  waits a few cycles after every dispatch and the others fill slots), whatever the launch
  order; with it a kernel dispatches only after every earlier one has dispatched all its
  work-groups, like one kernel in work-group ID order. More than 8 concurrent kernels per
  GPU need more dispatchers, or the extra kernel takes a freed low-index dispatcher.
  `-ordered-dispatch` also makes the driver send each GPU's launches in enqueue order
  (`Driver.OrderedLaunch`): queues race otherwise (see the flush gotcha below).
- **Software baselines** (Step 7). `mccl.AllReduceDirect` (`-algorithm direct`): direct
  reduce-scatter + all-gather, each GPU sends each peer its share once, pushes to different
  peers on their own queues; the fair software collective on the per-pair mesh.
  `mccl.Accumulate` adds buffers on every GPU (split-K pre-sum). `-sw-overlap BANDS`
  (no engines): every band's GEMM launched up front (split per `-splitk-plan`), and the
  host all-reduces each band with mccl once it is done. mccl's kernels carry
  `insts.HsaCo.UnorderedDispatch`, so under `-ordered-dispatch` they run beside the GEMM
  instead of behind it.
- **`-engine-peer-queues`** (default off): one engine send queue per peer, each paying
  `-msg-overhead` (a DMA queue per link). Needed at 8 GPUs, where one queue would spend 1.75
  messages' overhead per tile against 0.25 messages per link.
- **`-wg-trace file`** (`samples/allreducegemm`): CSV of every work-group's dispatch and
  completion time; `scripts/step5_wgs.py` summarizes it per kernel.
- **Device-side signals and the FlashOverlap-style baseline** (Step 9). `timing/signal.Board`
  is shared by every CP: a launch whose code object has `insts.HsaCo.WaitSignals` is held at
  the CP (no dispatcher) until those signals are raised, and raises `Signal` (after
  `SignalDelay` s) when it completes; such launches are exempt from ordered launch.
  `HsaCo.CommCUs` restricts a kernel to the CP's comm CUs (`CommandProcessor.SetCommCUs`),
  which other kernels cannot take from the comm kernel's arrival at the CP until it ends.
  `-sw-flash ROWS,...` (no engines): the trackers count the GEMM's tiles in row groups and
  raise a group signal; pre-enqueued mccl kernels (`mccl.AllReduceDirectOnSignal`,
  `AccumulateOnSignal`: pre-sum, direct reduce-scatter, chained reduces, all-gather, one
  queue per kernel) run on `-sw-comm-cus` CUs; `-sw-signal-cost` ns per signal (+ link
  latency to a peer). `tiletracker.Region.IgnoreLate` ignores stores into completed tiles;
  `commengine.TrackRegion` builds a region; `Driver.WaitIdle` lets the host wait for the
  engine to go idle. `-warmup-skip-b`: the warm-up reads a dummy B (A and C warm).
- **L1 invalidate fix** (Step 9, Akita `mem/cache/writearound`): the vector L1's invalidate
  marks its blocks invalid in place; replacing the directory (`Reset`) under an in-flight
  fill let another miss take the block, and hits returned other lines' data.
  `allreducegemm` seeds its own `rand` source.
- **All-gather before the GEMM** (Step 12; `-allgather-a`, `timing/commengine/push.go`,
  `benchmarks/allreducegemm/allgather.go`). Every GPU holds M/p rows of A (full K, here `-k`
  is not split) and its own K x N block of B (B is K x pN), C = A x B_local, verified per GPU.
  Engine push mode (`Config.Push`, `Session.ArmPush`): reads this GPU's rows of tiles of A,
  writes them into every peer's A (one row of tiles per message, send order a list, default
  `DirectOrder`), raises `gpuN/a/rowR` on the CP signal board once L2 acknowledged the row's
  last write. GEMM: one launch per row of tiles on its own queue, local rows first, remote
  rows held (`WaitSignals`) in expected arrival order; `-allgather-bulk` one launch held for
  all rows; `-allgather-local` A on every GPU (floor); `-allgather-mccl`
  `mccl.AllGatherDirect` then the GEMM. Needs `-engine-mode` (rows/bulk), `-ordered-dispatch`;
  8 GPUs `-peer-mesh -engine-peer-queues`. The window starts after `Driver.WaitIdle`, at the
  push start; launches are enqueued at the same time (their ~46 us driver latency is in the
  window, but held launches wait at the CP during the push). `CommandProcessor
  .MaxBusyDispatchers` counts dispatchers in use (logged as `allgather dispatchers`).
- **`samples/allreducebench`** runs Ring All-Reduce alone and splits its time into setup copies,
  kernel time, link-busy time and bytes; resumable via its `-out` CSV.

Gotchas that cost time before:
- MGPUSim v3 loads only **code object v2**. Upstream LLVM ≤ 16 can still emit it
  (`-mcode-object-version=2`); Ubuntu 24.04 ships clang/lld 14–16.
- Akita's PCIe link bandwidth is whole bytes per cycle at the network clock, so very low or
  fractional bandwidths need a lower network frequency.
- In a pattern for `pkill -f`, the pattern also matches the shell running it; kill by PID.
- The driver marks every buffer L2-dirty at each launch and never clears it, and its
  overlap test counts touching ranges, so after the first launch most host copies (including
  each launch's argument copies) first flush L2 (write back, lines stay valid). Queues whose
  copies flush fall behind, so launches on different queues can reach the CP out of order.
- Don't edit a shell script while a service is running it: bash reads it as it goes, and the
  edit broke the Step 5 batch mid-run.
- The vector L1 is Akita's `writearound` cache, not `writethrough` (that one is the scalar
  and instruction L1). L1s and L2s see physical addresses.
- `rand.Seed` is a no-op since Go 1.24; use `rand.New(rand.NewSource(seed))`.
- mgpusim's host goroutine runs beside the engine: after `WaitFor` or a drain returns, the
  engine may still be running, so the next host enqueue lands at a racy simulated time.
  Call `Driver.WaitIdle` first when that matters.

## Running experiments

- Every run uses `-verify`. Results go to `results/<experiment>/` with the script that produced
  them; list new folders in `results/README.md`.
- One simulation needs about 0.8–1.3 GB of RAM; run at most 3 at a time on this machine. Wall
  time grows 2–10× when runs share the machine.
- Run long batches as a systemd user service so they survive VS Code or Claude Code closing,
  and make them resumable (`sweep_overlap.py` and `allreducebench` skip finished work):
  `systemd-run --user --unit=<name> --collect -E PATH="$PATH" --working-directory="$PWD" bash -c '<script> >> <log> 2>&1'`.
- Agreed working rules with the user: smallest shape that shows the effect; state run count and
  wall time before a batch and ask first if it exceeds 30 minutes; commit after each step, then
  stop and report a short table.

## Fused GEMM + collective project: status

Goal: co-design GEMM tile size and message granularity under hardware-triggered communication,
compared across reduction placements (related work: T3, ASPLOS 2024; CAIS, arXiv 2605.05628).
Design: option (b) in `reports/fused-gemm-comm-design.md`, an L2-side tile tracker (Akita port
hooks on L2 top ports) plus a DMA-like send engine over RDMA, so the existing GEMM binary runs
unchanged.

| Step | What | Status |
|---|---|---|
| 0 | Ratio-matched sizing (`scripts/ratio_model.py`) | Done, `210936c`. Proposal: 256×256×512, 2 GPUs, 64 CUs, tileSize 8, link bandwidth 0.7–40 GB/s per real layer |
| 1 | Tracker spike: 1 GPU, every tile completes exactly once after its last L2-acknowledged store (tileSize 4/8/16, 128×128×256) | Done: go. `timing/tiletracker`, test `benchmarks/allreducegemm/tiletracker_test.go`, `results/step1_tracker/` |
| 2 | Send engine, per-tile reduce-scatter + all-gather, `-tiles-per-msg N`; verify at N = 1, 2, 4 | Done: `timing/commengine`, `results/step2_engine/` |
| 3 | Reduction placement: (a) owner reduces via engine, (b) T3-style near memory; tile size × N with GCN3-native and matrix-core compute curves | Done: `results/step3_tiles/` |
| 3b | Dispatch-shaping spike (`-wg-cap`, `-wg-ramp`) and the 4-GPU per-pair mesh | Done: `results/step4a_shaping/` |
| 5 | Split-K first and last bands (no new kernel), s = 2 and 4 | Done, `13bae25`: `results/step5_splitk/`. f≈0.5: s = 2 578 us (-8.6%), s = 4 612 (-3.2%), no pre-sum 658 (+4.0%); f≈0.8: -4.8% / -6.5% |
| 5b | Queue-model bounds, cold vs warm first round, per-row split plan | Done, `f23ed3f`: `results/step5b_bounds/`. Even-stream bound 490 us at f≈0.5 (-15% vs 578), 1,591 at f≈0.8 (+0.4%); plan 4,4,4,4,1,1,2,2: 524.9 us at f≈0.5 (-9.2%), f≈0.8 unchanged; slow first round is mostly cold caches |
| 6 | Paper-ready split plan: warm/cold, f≈0.25, 4 GPUs, MI300X projection | Done, `fab2f71`: `results/step6_paper/`. Plan vs s = 1: f≈0.25 -3.7%, f≈0.5 -17.0% cold / -15.5% warm, 4 GPUs -16.4%, f≈0.8 -6.4%; model within 0.5-4.3% |
| 7 | Baselines and model validation (batch 1 of 3) | Done, `dee876c`: `results/step7_baselines/`. Software overlap: plan +35% vs s = 1 (gain does not survive). 8 GPUs: plan -13.3% vs s = 1, -68.9% vs direct software. Rank test Spearman 0.97, chosen plan = best measured. MI300X f≈0.5 gain < 10% only at 1.63x split cost |
| 8 | Realistic software baseline and edge cases (batch 2 of 3) | Done, `aaa48da` + `706d7b1`: `results/step8_edges/`. Event-driven model (mean error 1.5%); splits help software overlap only below ~20 us per call (published RCCL minimum ~20 us); f sweep -14% / -13% / -3% at f 0.35 / 0.65 / 0.9; shapes, K 128/512, 9 rows, reduce-scatter all keep the gain (-6.5% to -20.2%), model within -3.2% to +1.5% |
| 9 | Fair software baseline, gain split, caches, all-gather design (batch 3 of 3) | Done: `results/step9_baselines/`. FlashOverlap-style (1 comm CU) vs engine, f≈0.5 2 GPUs: s = 1 979 / plan 1009 vs 632 / 525 us; 8 GPUs 3758 / 3919 vs 655 / 575. Plan gain is all overlap (GEMM 1-4% longer). A+C-warm plan -17.1%. L1 invalidate bug fixed. All-gather: design note, worth building |
| 4 | First timing: baseline (one GEMM + Ring All-Reduce, `-collective-threads 16384`) vs (a) vs (b) | Mostly covered by the capped baseline (`results/step4b_capped/`: mccl ring, engine bulk, engine tile) and Steps 5-8; near-memory (b) not rerun at the cap |
| 10 | FlashOverlap-style baseline at a realistic CU share (16 CUs, 1 comm CU) | Done, 10 + 10b (`results/step10_cushare/`). 512x512 cold: software s = 1 / plan 999.7 / 943.4 us, engine 719.9 / 549.1 us (engine plan -41.8% vs software plan). The plan helps software at 1/16 CU (-5.6%). Comm CU stretches the s = 1 GEMM 9.6%. Engine plan gain = GEMM about -11% + overlap about -13 to -16% (cold and A+C warm): slow cold first round with 16 CUs on one L2 |
| 11 | Plan gain split vs CU count and cache state | Done, `8f1e357`: `results/step11_firstround/`. All warm at 16 CUs the plan gains -12.9% = GEMM +5.7% + overlap -18.6% (as at 4 CUs); cold 4 / 8 / 16 CUs: GEMM part +2.3% / +0.3% / -10.5%. The 16-CU GEMM part is the cold first round (round-1 work-groups 1.76x the later ones cold, 1.07x warm): nearly every first-round L2 read misses or merges into an in-flight miss; not DRAM queueing, not TLB. Model with measured round times: s = 1 error 3.9% -> 0.8%, plans 2.5% -> 2.1% |
| 12 | All-gather of A before the GEMM on the engines, s = 1 | Done: `results/step12_allgather/`. 2 GPUs gather-then-GEMM 649.8 / per-row 401.4 us (model 696.2 / 421.5); 8 GPUs 662.0 / 604.9 (709.4 / 640.2); GEMM alone 361.1 / 362.3 (355.0); mccl all-gather + GEMM 931.9 (2 GPUs). Push at 0.79 of the link (0.76 at 8 GPUs from the push start) |
| 13 | Split rows on the input side (K-chunk messages) | Next |
| - | MoE all-to-all extension (design note section only) | Not started |

Step requirements, as the user set them:
1. **Tracker spike.** One GPU, tracker only. Every output tile reports complete exactly once,
   after its last store is acknowledged by L2. Test tileSize 4, 8 and 16 at 128×128×256. If it
   isn't reliable, stop and explain.
2. **Correctness on 2 GPUs.** Build the send engine and per-tile reduce-scatter + all-gather.
   The tracker supports `-tiles-per-msg N`: merge N adjacent finished tiles into one message
   (N=1 is per tile). Verify 128×128×256, tileSize 8, at N = 1, 2, 4 against the CPU sum.
   Correctness only.
3. **Two reduction placements**, configurable: (a) the tile owner reduces via the engine;
   (b) T3-style near-memory reduction at the memory controller. Re-verify both at the Step 2
   size. Read T3's mechanism section first and document what we model the same and what
   differs.
4. **First timing comparison** on one ratio-matched config from Step 0 that runs in a few
   minutes: strong baseline (one GEMM launch + Ring All-Reduce, `-collective-threads 16384`),
   placement (a), placement (b), at tileSize 8 and N = 1. Report the window, the gap to ideal
   (GEMM time plus the last message's round trip), and a per-message timeline for the first
   few messages.
5. **MoE plan, design note only.** Append to `reports/fused-gemm-comm-design.md`: extending the
   engine to MoE all-to-all (dispatch and combine) with skewed, uneven per-expert message
   sizes. mccl has no all-to-all, so it needs a host-side baseline all-to-all and an expert
   GEMM benchmark. Cite CAIS's MoE in-switch paper (arXiv 2605.05607) as related work. Don't
   implement.

No tile-size × N sweep until the user has seen Step 4.

**Baseline from Step 4 on (user decision):** the GEMM runs with `-gemm-wg-cap 8` (the cap that
minimizes its time; 12 resident work-groups oversubscribe a CU) in every version (mccl ring,
engine bulk, engine tile). With cap 8, f≈0.5 and f≈0.8 use 256×256, K/GPU 256, 4 CUs, tileSize
8 (2.0 waves, first share 0.54), and knobs from `results/step0_ratio/points_capped.csv`:
f≈0.5 `-link-bw 0.7807 -link-latency 441 -msg-overhead 2121`; f≈0.8 `-link-bw 0.223
-link-latency 1542 -msg-overhead 7424`. tileSize 4 on the matrix-core curve under the cap:
`-valu-cycles-scale`/`-small-valu-scale` 0.738. Default flags still reproduce 433 µs.
Plan for Step 4 on the per-pair mesh: a ring is not the fair software baseline there; add a
direct reduce-scatter + all-gather to mccl (each GPU sends its 1/n share straight to each peer).

Representative points (user-approved; use only these in Step 4 and later sweeps). Knobs from
`scripts/ratio_model.py` (`results/step0_ratio/points.csv`), all 2 GPUs, 4 CUs
(`-cus-per-gpu 4`), tileSize 8, K/GPU 256, `-lds-occupancy`:

| Point | Real layer | Sim shape (M×N×K) | `-link-bw` GB/s | `-link-latency` ns | `-msg-overhead` ns |
|---|---|---|---|---|---|
| f≈0.8 | Llama-3 70B attn out-proj all-reduce | 384×256×512 | 0.1807 | 1904 | 9163 |
| f≈0.5 | Llama-3 70B MLP down-proj all-reduce | 384×256×512 | 0.6325 | 544 | 2618 |
| f≈0.25 | Llama-3.1 405B MLP down-proj reduce-scatter | 512×320×512 | 1.1504 | 299 | 1440 |

Step 6 knobs (`results/step0_ratio/points_step6.csv`): f≈0.25 at the cap, 512x320,
reduce-scatter, `-link-bw 1.632 -link-latency 211 -msg-overhead 1015` (GEMM time estimated,
achieved f 0.246); f≈0.5 on 4 GPUs (`-peer-mesh`, K/GPU 256), `-link-bw 0.3903 -link-latency
881 -msg-overhead 4243` (the direct reduce-scatter + all-gather puts 2C/p on each link, so
half the bandwidth keeps the link time; latency and overhead double to keep the ratio).
f≈0.5 on 8 GPUs, same scaling: `-link-bw 0.1952 -link-latency 1763 -msg-overhead 8485`,
with `-engine-peer-queues`.

Latency/transfer for a 1-tile message is matched to the real 0.39 (1 µs DMA copy = 172 ns wire
+ 828 ns engine, MI300A data; 128 KiB tile at 51.2 GB/s); the sensitivity range is 0.27–0.78.
Calibrated in Step 2 (`TestCalibration`, `results/step2_engine/calibration.csv`): the
engine streams at 0.794 of `-link-bw` (assumed 0.80), and one message's latency beyond its
transfer time is 6–7% above `-link-latency` + `-msg-overhead` (≈9 fixed fabric cycles), both
within 10%, so the knobs stand. Step 4 versions: mccl ring after the GEMM
(`-collective-threads 16384`), engine `-engine-mode bulk`, engine `-engine-mode tile`.

Rule added after Step 1: every simulated config must have tiles finishing spread out over
time (at 64 CUs a 256×256 GEMM finishes all tiles within 1 µs, leaving nothing to overlap).
Waves = tiles / (CUs × resident work-groups per CU); check spread with the tracker
(`TestTileTrackerRepresentative`, `scripts/tile_spread.py`). Candidate shapes: 4 CUs,
tileSize 8, K/GPU 256: 384×256 (2.0 waves) and 512×320 (3.3 waves). A 16-CU 768×512 run
(also 2.0 waves) has the same spread, so 4 CUs is not a toy.

Modelling tools (Python, `scripts/`; run from the repo root, no simulations):
- `step6_model.py`: work-group list schedule (`Gemm`: 32 slots, per-split work-group times
  from the Step 5 traces, optional cold first round) and per-row plan search helpers;
  `knobs()` reads every point's link knobs from `results/step0_ratio/`.
- `step9_model.py`: FlashOverlap-style path (GEMM on the CUs comm leaves, per-group direct
  all-reduce at the calibrated link efficiency, contention when several groups overlap) and the
  Step 9 run list; `step9_allgather.py`: all-gather-before-GEMM queue model.
- `step10_model.py`: the 16-CU point (knob derivation, engine plan search over the structured
  16-row family with at most 36 launches, the Step 9 software model on 16 x 16 tiles with its
  saved constants) and the Step 10 run list.
- `step8_model.py`: the engine model. `Comm` is event-driven (an owner's sum waits for every
  partial; per-engine or per-peer lanes; wire latency calibrated as 1.07 x (overhead +
  latency) - overhead); `PSGemm` time-shares each CU among up to 8 resident work-groups, for
  partial last rounds. `step8_validate.py` checks it on 22 engine runs (-4.9% to +1.5%).
  Predict every run with it before running.
- `step7_model.py` / `step8_sw.py`: software-overlap model (F per call + bytes at eta of the
  link, fitted to the capped mccl runs: F 277 us, eta 0.50; CU loss phi).
- `step7_sensitivity.py`: MI300X projection over split costs.
- Work-group time is linear in its K chunk, 5.4 + 0.514 x K us; steady split cost per
  work-group (normalized to 8 resident) is 1.03-1.09x (s = 2) and 1.13-1.25x (s = 4) at
  K/GPU 256, 1.18x / 1.42x at 128.

Findings from Steps 5-8:
- The per-row split plan (pick it with the model; 4,4,4,4,1,1,2,2 at f≈0.5) beats s = 1
  with the hardware engine everywhere tried: 2, 4 and 8 GPUs (-17%, -17%, -13%), warm and cold,
  f 0.35-0.9 (-14% to -3%), 1.5-3.4 waves, K/GPU 128-512, 9 tile rows, reduce-scatter.
- The gain comes from the first tile ready earlier and the link kept busy, not from faster
  work-groups (residency is 8 per CU in every band). Part of Step 5's GEMM speedup was the
  cold first round (172 vs 137 us per work-group), which warm caches remove.
- Software overlap does not get it: at ~277 us per mccl call (138 us of it launches) and
  collectives slowing the GEMM about 2x, splitting makes it worse (+35%). Splits only help
  software once a call costs about 20 us or less, and software still trails the engine by
  50-67% at its 50% link efficiency.
- Model rank order matches measurement (Spearman 0.97); its chosen plan was the best measured.
- MI300X projection (model only): f≈0.5 gain stays above 10% unless both split costs exceed
  about 1.63x; no published per-split cost exists for MI300X or MI250.

Findings from Step 9:
- A FlashOverlap-style software path (signals from the tracker, comm kernels pre-launched on a
  reserved CU) loses to the engine: the engine plan's window is 48% shorter at 2 GPUs and 85% at
  8. CU-driven stores move at most 0.50 of the link here (0.11 at 8 GPUs with one comm CU), and
  the held CU slows the GEMM by about 40%.
  Split plans make it worse (pre-sums queue on the comm CU), so the plan is an engine result.
- The plan's GEMM is never faster (+1% to +4% at 2 waves); the whole gain is overlap.
- With A and C warm and B cold (streamed weights) the plan still gains 17.1%.
- `step9_model.py` covers the software path (calibrated on cal_one / cal_rows); it misses the
  8-GPU CU limit (-74%) and the plan's pre-sum contention (about -10%).

Findings from Step 12 (all-gather before the GEMM, s = 1, f 0.5, 256x256x256 per GPU, 4 CUs;
knobs 2 GPUs `-link-bw 0.487 -link-latency 707 -msg-overhead 3403`, 8 GPUs `0.122 / 2827 /
13611`):
- Every run within -6.7% to +2.0% of the design note's queue model (`step9_allgather.py`),
  which over-predicts the held runs because it charges the 46 us launch after the data
  arrives, while the held launch already waits at the CP.
- Per-row overlap: -38% vs gather-then-GEMM at 2 GPUs, only -8.6% at 8 GPUs, where all 7 remote
  rows arrive together at 354 us (one per link) and only the local row (8 of 64 tiles)
  overlaps: the case K-chunk splits (Step 13) target.
- At 8 GPUs: 56 held launches (7 per GPU), at most 7 of 40 dispatchers busy, no release
  later than 4 ns after its signal; the 7 rows arriving together are written into L2 one
  after another (signals 354.8 to 357.9 us, about 0.5 us per 32 KB row).
- mccl's direct all-gather + GEMM (2 GPUs) is 931.9 us (gather 584.8), 1.43x gather-then-GEMM
  on the engines and 2.3x per-row overlap.

Findings from Step 10 (16 CUs per GPU, 512x512, K/GPU 256, 2.0 waves, knobs `-link-bw 3.1218
-link-latency 110 -msg-overhead 530`, f 0.489):
- At 1 comm CU of 16 the FlashOverlap-style path still trails the engine (plan 943 vs 549 us,
  -42%), but the plan now helps it a little (-5.6%; +3.0% at 1 of 4). The comm CU stretches the
  s = 1 GEMM 9.6% (about 40% at 4 CUs); link efficiency stays 0.42-0.48, so one CU drives the 4x
  faster link at 2 GPUs.
- With 16 CUs on one L2 the cold first round is slow (262 vs 172 us per work-group), and
  warming A and C barely helps s = 1 (715 vs 720 us). About 40% of the engine plan's gain is
  then the GEMM getting shorter (GEMM -10.5% / -10.8% + overlap -13.2% / -15.6%, cold / A+C
  warm); at 4 CUs it was none.
- The models assume the 4-CU work-group times, so they under-predict 16-CU s = 1 runs by
  11-13%; plan and software runs are within -2% to -6%.
- 16-CU runs take 135-300 s of wall time each (about 4-5x the 4-CU runs).

Earlier findings (details in `reports/gemm-allreduce-overlap.html`):
- Banding a GEMM to overlap All-Reduce never beat one launch: extra launches, low-occupancy
  short GEMMs, and collective kernels competing for CUs. This holds at `-h2d-cycles` 1450.
- Ring All-Reduce cost is per-launch setup plus latency-bound kernels; the link is busy under
  10% of the time.
