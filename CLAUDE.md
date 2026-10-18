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
- **`samples/allreducebench`** runs Ring All-Reduce alone and splits its time into setup copies,
  kernel time, link-busy time and bytes; resumable via its `-out` CSV.

Gotchas that cost time before:
- MGPUSim v3 loads only **code object v2**. Upstream LLVM ≤ 16 can still emit it
  (`-mcode-object-version=2`); Ubuntu 24.04 ships clang/lld 14–16.
- Akita's PCIe link bandwidth is whole bytes per cycle at the network clock, so very low or
  fractional bandwidths need a lower network frequency.
- In a pattern for `pkill -f`, the pattern also matches the shell running it; kill by PID.

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
| 3 | Reduction placement: (a) owner reduces via engine, (b) T3-style near memory | Not started |
| 4 | First timing: baseline (one GEMM + Ring All-Reduce, `-collective-threads 16384`) vs (a) vs (b) | Not started |
| 5 | MoE all-to-all extension (design note section only) | Not started |

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

Representative points (user-approved; use only these in Step 4 and later sweeps). Knobs from
`scripts/ratio_model.py` (`results/step0_ratio/points.csv`), all 2 GPUs, 4 CUs
(`-cus-per-gpu 4`), tileSize 8, K/GPU 256, `-lds-occupancy`:

| Point | Real layer | Sim shape (M×N×K) | `-link-bw` GB/s | `-link-latency` ns | `-msg-overhead` ns |
|---|---|---|---|---|---|
| f≈0.8 | Llama-3 70B attn out-proj all-reduce | 384×256×512 | 0.1807 | 1904 | 9163 |
| f≈0.5 | Llama-3 70B MLP down-proj all-reduce | 384×256×512 | 0.6325 | 544 | 2618 |
| f≈0.25 | Llama-3.1 405B MLP down-proj reduce-scatter | 512×320×512 | 1.1504 | 299 | 1440 |

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

Findings that shape the design (details in `reports/gemm-allreduce-overlap.html`):
- Banding a GEMM to overlap All-Reduce never beat one launch: extra launches, low-occupancy
  short GEMMs, and collective kernels competing for CUs. This holds at `-h2d-cycles` 1450.
- Ring All-Reduce cost is per-launch setup plus latency-bound kernels; the link is busy under
  10% of the time.
