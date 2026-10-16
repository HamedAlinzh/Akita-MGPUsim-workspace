package allreducegemm_test

import (
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

func runCase(t *testing.T, numGPU int, algorithm allreducegemm.Algorithm) {
	t.Helper()
	runBandedCase(t, numGPU, algorithm, 0, false)
}

func runBandedCase(
	t *testing.T,
	numGPU int,
	algorithm allreducegemm.Algorithm,
	bandRows uint32,
	overlap bool,
) {
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
	b.BandRows = bandRows
	b.Overlap = overlap
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

func TestGEMMRing2GPUBanded(t *testing.T) {
	runBandedCase(t, 2, allreducegemm.AlgorithmRing, 64, false)
}

func TestGEMMRing2GPUOverlap(t *testing.T) {
	runBandedCase(t, 2, allreducegemm.AlgorithmRing, 32, true)
}

func TestGEMMRing4GPUOverlap(t *testing.T) {
	runBandedCase(t, 4, allreducegemm.AlgorithmRing, 64, true)
}

func TestGEMMNaive2GPUOverlap(t *testing.T) {
	runBandedCase(t, 2, allreducegemm.AlgorithmNaive, 32, true)
}
