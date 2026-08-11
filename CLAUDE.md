# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a Go workspace (`go.work`, Go 1.26) containing two modules developed together:

- `akita-3.0.0` — module `github.com/sarchlab/akita/v3`. A general-purpose discrete-event
  simulation *engine* (analogous to a game engine), not a simulator itself. Provides the core
  event loop, component/port/connection model, generic memory components (`mem/`), and
  network-on-chip components (`noc/`).
- `mgpusim-3.0.3` — module `github.com/sarchlab/mgpusim/v3`. A GPU simulator built on top of
  Akita that models GPUs running the AMD GCN3 instruction set, including multi-GPU simulation.

`mgpusim` depends on `akita` via the module path `github.com/sarchlab/akita/v3`; the `go.work`
file wires local `./akita-3.0.0` in place of a tagged release so cross-module changes are picked
up immediately without touching `go.mod`. Each module also has commented-out `replace` directives
in its `go.mod` for standalone (non-workspace) development.

## Common commands

Run from inside the relevant module directory (`akita-3.0.0` or `mgpusim-3.0.3`) unless noted.

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
