package main

import (
	"flag"
	"log"
	"strconv"
	"strings"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
	"github.com/tebeka/atexit"
)

var mFlag = flag.Uint("m", 128, "Number of rows of matrix A and C.")
var kFlag = flag.Uint("k", 128,
	"Contraction (inner) dimension shared by A and B, split across GPUs.")
var nFlag = flag.Uint("n", 128, "Number of columns of matrix B and C.")
var algorithmFlag = flag.String("algorithm", "ring",
	`Which All-Reduce algorithm combines the per-GPU partial products:
"ring", "naive" or "direct" (reduce-scatter + all-gather straight to each peer).`)
var bandRowsFlag = flag.Uint("band-rows", 0,
	`Split the M output rows into bands of this many rows, each computed and
all-reduced separately (0 = one band covering all of M).`)
var overlapFlag = flag.Bool("overlap", false,
	"All-reduce band r while band r+1's GEMM runs.")
var allReduceChunkFlag = flag.Uint("allreduce-chunk", 0,
	"All-Reduce scratch buffer size in elements (0 = 65536).")
var collectiveThreadsFlag = flag.Int("collective-threads", 1024,
	"Threads per All-Reduce push/reduce kernel, a multiple of 64.")
var tilesPerMsgFlag = flag.Int("tiles-per-msg", 1,
	"With -engine-mode: finished tiles merged into one engine message.")
var reduceScatterFlag = flag.Bool("reduce-scatter", false,
	"Reduce-scatter instead of all-reduce: with -engine-mode in the engines, "+
		"otherwise a ring reduce-scatter (one band only).")
var placementFlag = flag.String("placement", "owner",
	`With -engine-mode: where tiles are reduced, "owner" (the owner's engine
adds the partials) or "near-memory" (T3-style op-and-store at the owner's
memory side).`)
var gemmWGCapFlag = flag.Int("gemm-wg-cap", 0,
	"Cap the GEMM kernels' work-groups in flight per CU (0 = -wg-cap).")
var gemmVALUScaleFlag = flag.Float64("gemm-valu-scale", 0,
	"Scale the main GEMM kernel's vector ALU cycles (0 = -valu-cycles-scale).")
var smallRowsFlag = flag.Uint("small-rows", 0,
	"Compute the first rows of C with a small-tile GEMM launched first.")
var smallTileSizeFlag = flag.Uint("small-tile-size", 4,
	"Work-group edge of the small-tile GEMM.")
var smallVALUScaleFlag = flag.Float64("small-valu-scale", 0,
	"Scale the small-tile GEMM's vector ALU cycles (0 = -valu-cycles-scale).")
var smallTilesPerMsgFlag = flag.Int("small-tiles-per-msg", 1,
	"With -engine-mode: small tiles per engine message.")
var splitKWaysFlag = flag.Int("splitk-ways", 0,
	"With -engine-mode: split the K loop of the first and last bands of C "+
		"across this many work-groups (0 = off).")
var splitKRowsFlag = flag.Uint("splitk-rows", 0,
	"Rows in each split-K band.")
var splitKPlanFlag = flag.String("splitk-plan", "",
	`With -engine-mode: a split factor per row of tiles, top to bottom, e.g.
"4,4,4,4,1,1,2,2" (replaces -splitk-ways and -splitk-rows).`)
var swOverlapFlag = flag.String("sw-overlap", "",
	`Software-overlap baseline (no -engine-mode): rows per band, top to bottom,
e.g. "32,32,32,32,64,32,32"; every band's GEMM is launched up front (split per
-splitk-plan) and each band is all-reduced with mccl once it is done.`)
var swFlashFlag = flag.String("sw-flash", "",
	`FlashOverlap-style software baseline (no -engine-mode): rows per tile
group, top to bottom, e.g. "64,64,128". The GEMM runs as with the engines
(split per -splitk-plan); when a group's tiles are all in L2 on a GPU, a
signal releases communication kernels enqueued before the GEMM and held at
the CP, which all-reduce the group with mccl's direct algorithm on the
-sw-comm-cus CUs.`)
var swCommCUsFlag = flag.Int("sw-comm-cus", 1,
	"With -sw-flash: CUs per GPU reserved for the communication kernels; "+
		"the GEMM cannot take them from the kernels' launch until they finish.")
var swSignalCostFlag = flag.Float64("sw-signal-cost", 0,
	"With -sw-flash: ns from a tile group (or a communication kernel) "+
		"completing to the kernels waiting for it being released; a signal to "+
		"another GPU also takes -link-latency.")
var splitKPresumFlag = flag.String("splitk-presum", "on",
	`"on": the engine adds a tile's split-K partials before sending it; "off":
it sends each partial and the owner adds them.`)
var wgTraceFlag = flag.String("wg-trace", "",
	"Write every work-group's dispatch and completion time to this CSV.")
var memTimelineFlag = flag.String("mem-timeline", "",
	"Write cache hits and misses, TLB misses and DRAM bytes per 10 us to this CSV.")
var warmupFlag = flag.Bool("warmup", false,
	"With -engine-mode or -sw-flash: run the GEMM once, untimed, before the "+
		"measured launch.")
var warmupSkipBFlag = flag.Bool("warmup-skip-b", false,
	"With -warmup: the warm-up run reads a dummy buffer in place of B, so A "+
		"and C start warm and B cold.")
var tileSizeFlag = flag.Uint("tile-size", 0,
	"Work-group edge T; each work-group computes a 4T x 4T tile (0 = 8).")
var allGatherAFlag = flag.Bool("allgather-a", false,
	`All-gather A before the GEMM (input side of a tensor-parallel layer): every
GPU holds M/p rows of A (full -k) and its own -k x -n block of B, and computes
C = A x B_local over all M rows. With -engine-mode the engines push each
GPU's rows to every peer and the GEMM runs one launch per row of tiles, local
rows first, each remote row's launch held at the CP until it has arrived.`)
var allGatherBulkFlag = flag.Bool("allgather-bulk", false,
	"With -allgather-a and -engine-mode: one GEMM launch, held until every "+
		"row has arrived (gather, then GEMM).")
var allGatherLocalFlag = flag.Bool("allgather-local", false,
	"With -allgather-a: A is already on every GPU; one GEMM launch (the floor).")
var allGatherMCCLFlag = flag.Bool("allgather-mccl", false,
	"With -allgather-a: mccl's direct all-gather, then one GEMM launch.")
var perBandScratchFlag = flag.Bool("per-band-scratch", false,
	"Give every band its own All-Reduce scratch buffers.")

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := allreducegemm.NewBenchmark(runner.Driver())
	benchmark.M = uint32(*mFlag)
	benchmark.K = uint32(*kFlag)
	benchmark.N = uint32(*nFlag)
	benchmark.BandRows = uint32(*bandRowsFlag)
	benchmark.Overlap = *overlapFlag
	benchmark.AllReduceChunk = uint32(*allReduceChunkFlag)
	benchmark.PerBandScratch = *perBandScratchFlag
	benchmark.TileSize = uint32(*tileSizeFlag)
	benchmark.GEMMWGCap = *gemmWGCapFlag
	benchmark.GEMMVALUScale = *gemmVALUScaleFlag
	benchmark.SmallRows = uint32(*smallRowsFlag)
	benchmark.SmallTileSize = uint32(*smallTileSizeFlag)
	benchmark.SmallVALUScale = *smallVALUScaleFlag
	benchmark.SmallTilesPerMsg = *smallTilesPerMsgFlag
	benchmark.SplitKWays = *splitKWaysFlag
	benchmark.SplitKRows = uint32(*splitKRowsFlag)
	benchmark.Warmup = *warmupFlag
	benchmark.WarmupSkipB = *warmupSkipBFlag
	if *swFlashFlag != "" {
		for _, f := range strings.Split(*swFlashFlag, ",") {
			rows, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				log.Panicf("bad -sw-flash %q: %v", *swFlashFlag, err)
			}
			benchmark.FlashGroups = append(benchmark.FlashGroups, uint32(rows))
		}
		benchmark.Flash = flashSetup(runner)
	}
	if *swOverlapFlag != "" {
		for _, f := range strings.Split(*swOverlapFlag, ",") {
			rows, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				log.Panicf("bad -sw-overlap %q: %v", *swOverlapFlag, err)
			}
			benchmark.SWBands = append(benchmark.SWBands, uint32(rows))
		}
	}
	if *splitKPlanFlag != "" {
		for _, f := range strings.Split(*splitKPlanFlag, ",") {
			s, err := strconv.Atoi(strings.TrimSpace(f))
			if err != nil {
				log.Panicf("bad -splitk-plan %q: %v", *splitKPlanFlag, err)
			}
			benchmark.SplitKPlan = append(benchmark.SplitKPlan, s)
		}
	}
	mccl.CollectiveThreads = *collectiveThreadsFlag

	if runner.EngineMode != "" {
		benchmark.Comm, benchmark.CommOptions = fusedComm(runner)
	} else {
		benchmark.ReduceScatter = *reduceScatterFlag
	}

	switch *algorithmFlag {
	case "ring":
		benchmark.Algorithm = allreducegemm.AlgorithmRing
	case "naive":
		benchmark.Algorithm = allreducegemm.AlgorithmNaive
	case "direct":
		benchmark.Algorithm = allreducegemm.AlgorithmDirect
	default:
		log.Panicf(`-algorithm must be "ring", "naive" or "direct", got %q`,
			*algorithmFlag)
	}

	if *allGatherAFlag {
		allGatherSetup(runner, benchmark)
	} else if *allGatherBulkFlag || *allGatherLocalFlag || *allGatherMCCLFlag {
		log.Panic("-allgather-bulk, -allgather-local and -allgather-mccl need -allgather-a")
	}

	if *wgTraceFlag != "" {
		wgs := traceWGs(runner.Platform())
		atexit.Register(func() { wgs.write(*wgTraceFlag) })
	}
	if *memTimelineFlag != "" {
		mt := traceMemory(runner.Platform(), 10)
		atexit.Register(func() { mt.write(*memTimelineFlag) })
	}

	runner.AddBenchmark(benchmark)

	runner.Run()
}

// fusedComm builds the communication-engine session over the GPUs the runner
// selected, from -engine-mode, -tiles-per-msg and -reduce-scatter.
func fusedComm(r *runner.Runner) (*commengine.Session, commengine.Options) {
	opt := commengine.Options{TilesPerMsg: *tilesPerMsgFlag}
	switch r.EngineMode {
	case "tile":
		opt.Mode = commengine.ModeTile
	case "bulk":
		opt.Mode = commengine.ModeBulk
	default:
		log.Panicf(`-engine-mode must be "tile" or "bulk", got %q`, r.EngineMode)
	}
	if *reduceScatterFlag {
		opt.Collective = commengine.ReduceScatter
	}
	switch *splitKPresumFlag {
	case "on":
	case "off":
		opt.SplitSend = true
	default:
		log.Panicf(`-splitk-presum must be "on" or "off", got %q`, *splitKPresumFlag)
	}
	switch *placementFlag {
	case "owner":
		opt.Placement = commengine.OwnerEngine
	case "near-memory":
		opt.Placement = commengine.NearMemory
	default:
		log.Panicf(`-placement must be "owner" or "near-memory", got %q`,
			*placementFlag)
	}

	var parts []commengine.Participant
	for _, id := range r.GPUIDs {
		gpu := r.Platform().GPUs[id-1]
		parts = append(parts, commengine.Participant{
			Tracker: gpu.TileTracker, Engine: gpu.CommEngine})
	}
	return commengine.NewSession(r.Driver(), parts), opt
}

// flashSetup gives every CP one signal board and reserves -sw-comm-cus CUs
// for communication, for the FlashOverlap-style baseline.
func flashSetup(r *runner.Runner) *allreducegemm.FlashSetup {
	board := signal.NewBoard(r.Platform().Engine)
	var trackers []*tiletracker.Tracker
	for _, id := range r.GPUIDs {
		gpu := r.Platform().GPUs[id-1]
		gpu.CommandProcessor.Signals = board
		gpu.CommandProcessor.SetCommCUs(*swCommCUsFlag)
		board.AddWaker(gpu.CommandProcessor)
		trackers = append(trackers, gpu.TileTracker)
	}
	local := *swSignalCostFlag * 1e-9
	peer := local + max(0, r.LinkLatencyNS)*1e-9
	return &allreducegemm.FlashSetup{Trackers: trackers, Signals: board,
		Delays: mccl.SignalDelays{Local: local, Peer: peer}}
}

// allGatherSetup selects the all-gather mode from the -allgather-* flags; with
// the engines it gives every CP one signal board for the rows' arrival
// signals, and reports each CP's busiest dispatcher count at exit.
func allGatherSetup(r *runner.Runner, b *allreducegemm.Benchmark) {
	switch {
	case *allGatherLocalFlag:
		b.AllGather = allreducegemm.AllGatherLocal
	case *allGatherMCCLFlag:
		b.AllGather = allreducegemm.AllGatherMCCL
	case *allGatherBulkFlag:
		b.AllGather = allreducegemm.AllGatherBulk
	default:
		b.AllGather = allreducegemm.AllGatherRows
	}
	if b.AllGather == allreducegemm.AllGatherRows || b.AllGather == allreducegemm.AllGatherBulk {
		if r.EngineMode == "" {
			log.Panic("-allgather-a needs -engine-mode unless -allgather-local or -allgather-mccl")
		}
		board := signal.NewBoard(r.Platform().Engine)
		for _, id := range r.GPUIDs {
			cp := r.Platform().GPUs[id-1].CommandProcessor
			cp.Signals = board
			board.AddWaker(cp)
		}
		b.AllGatherSignals = board
	}
	atexit.Register(func() {
		for i, id := range r.GPUIDs {
			log.Printf("allreducegemm allgather dispatchers: gpu=%d max_busy=%d of=%d", i,
				r.Platform().GPUs[id-1].CommandProcessor.MaxBusyDispatchers,
				len(r.Platform().GPUs[id-1].CommandProcessor.Dispatchers))
		}
	})
}
