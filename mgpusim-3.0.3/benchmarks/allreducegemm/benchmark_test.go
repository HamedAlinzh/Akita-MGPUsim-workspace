package allreducegemm_test

import (
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

func runCase(t *testing.T, numGPU int, algorithm allreducegemm.Algorithm) {
	t.Helper()

	platform := runner.MakeR9NanoBuilder().WithNumGPU(numGPU).Build()
	gpuDriver := platform.Driver
	gpuDriver.Run()
	defer gpuDriver.Terminate()

	gpuIDs := make([]int, numGPU)
	for i := range gpuIDs {
		gpuIDs[i] = i + 1
	}

	b := allreducegemm.NewBenchmark(gpuDriver)
	b.Algorithm = algorithm
	b.M, b.K, b.N = 128, 128, 128
	b.SelectGPU(gpuIDs)
	b.EnableVerification()

	b.Run()
	b.Verify()
}

func TestGEMMRing1GPU(t *testing.T) {
	runCase(t, 1, allreducegemm.AlgorithmRing)
}

func TestGEMMRing2GPU(t *testing.T) {
	runCase(t, 2, allreducegemm.AlgorithmRing)
}

func TestGEMMRing4GPU(t *testing.T) {
	runCase(t, 4, allreducegemm.AlgorithmRing)
}

func TestGEMMNaive1GPU(t *testing.T) {
	runCase(t, 1, allreducegemm.AlgorithmNaive)
}

func TestGEMMNaive2GPU(t *testing.T) {
	runCase(t, 2, allreducegemm.AlgorithmNaive)
}

func TestGEMMNaive4GPU(t *testing.T) {
	runCase(t, 4, allreducegemm.AlgorithmNaive)
}
