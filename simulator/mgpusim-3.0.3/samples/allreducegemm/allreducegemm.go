package main

import (
	"flag"
	"log"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
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
	mccl.CollectiveThreads = *collectiveThreadsFlag

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
