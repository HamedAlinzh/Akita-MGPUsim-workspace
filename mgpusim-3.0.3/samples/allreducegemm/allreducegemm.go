package main

import (
	"flag"
	"log"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

var mFlag = flag.Uint("m", 128, "Number of rows of matrix A and C.")
var kFlag = flag.Uint("k", 128,
	"Contraction (inner) dimension shared by A and B, split across GPUs.")
var nFlag = flag.Uint("n", 128, "Number of columns of matrix B and C.")
var algorithmFlag = flag.String("algorithm", "ring",
	`Which All-Reduce algorithm combines the per-GPU partial products:
"ring" or "naive".`)

func main() {
	flag.Parse()

	runner := new(runner.Runner).ParseFlag().Init()

	benchmark := allreducegemm.NewBenchmark(runner.Driver())
	benchmark.M = uint32(*mFlag)
	benchmark.K = uint32(*kFlag)
	benchmark.N = uint32(*nFlag)

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
