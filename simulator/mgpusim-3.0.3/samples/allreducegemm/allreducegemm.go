package main

import (
	"flag"
	"log"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
)

var mFlag = flag.Uint("m", 128, "Number of rows of matrix A and C.")
var kFlag = flag.Uint("k", 128,
	"Contraction (inner) dimension shared by A and B, split across GPUs.")
var nFlag = flag.Uint("n", 128, "Number of columns of matrix B and C.")
var algorithmFlag = flag.String("algorithm", "ring",
	`Which All-Reduce algorithm combines the per-GPU partial products:
"ring" or "naive".`)
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
var tileSizeFlag = flag.Uint("tile-size", 0,
	"Work-group edge T; each work-group computes a 4T x 4T tile (0 = 8).")
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
	default:
		log.Panicf(`-algorithm must be "ring" or "naive", got %q`,
			*algorithmFlag)
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
