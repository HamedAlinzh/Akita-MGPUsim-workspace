// Package allreducegemm implements a distributed matrix multiplication
// (GEMM) benchmark that splits the contraction (K) dimension across GPUs.
// Each GPU computes a full-size partial product from its own private slice
// of A and B, and an All-Reduce (Ring or Naive) sums the partials into the
// final result. This makes All-Reduce the actual multi-GPU synchronization
// primitive, unlike a row/column split of the output, which would only
// need a gather.
package allreducegemm

import (
	// embed the compiled GEMM kernel
	_ "embed"
	"log"
	"math"
	"math/rand"

	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/kernels"
)

//go:embed kernels.hsaco
var hsacoBytes []byte

// tileSize is the local work-group edge length (8x8) that
// mmmKernel_local's tiling loop assumes; N and each GPU's K slice must be a
// multiple of 4*tileSize for the kernel's internal loop counts to come out
// exact.
const tileSize = 8

// rowBand is the number of output rows computed per kernel launch: exactly
// one workgroup-row (4*tileSize), since grids with more than one
// workgroup-row in Y produce wrong results in this simulator (see exec).
// M must be a multiple of rowBand.
const rowBand = 4 * tileSize

// naiveBufElements caps the AllReduce scratch buffer at a fixed chunk size
// so memory use does not grow with M*N, matching the convention used by
// the DNN data-parallel trainer (benchmarks/dnn/gputraining).
const allReduceBufElements = 65536

// Algorithm selects which All-Reduce implementation combines the per-GPU
// partial products.
type Algorithm string

const (
	// AlgorithmRing uses mccl.AllReduceRing.
	AlgorithmRing Algorithm = "ring"
	// AlgorithmNaive uses mccl.AllReduceNaive.
	AlgorithmNaive Algorithm = "naive"
)

// Benchmark is the distributed GEMM benchmark.
type Benchmark struct {
	driver  *driver.Driver
	context *driver.Context
	gpus    []int

	// M, K, N are the dimensions of the multiplication: A is M x K,
	// B is K x N, C = A*B is M x N.
	M, K, N uint32

	// Algorithm selects Ring (default) or Naive All-Reduce.
	Algorithm Algorithm

	kernel *insts.HsaCo

	matrixA, matrixB, matrixC *matrix

	verify bool
}

// NewBenchmark creates a new distributed GEMM benchmark using Ring
// All-Reduce by default; set Algorithm to AlgorithmNaive to use the naive
// gather-reduce-broadcast baseline instead.
func NewBenchmark(d *driver.Driver) *Benchmark {
	b := new(Benchmark)
	b.driver = d
	b.context = d.Init()
	b.Algorithm = AlgorithmRing
	return b
}

// SelectGPU selects which GPUs participate. The K dimension is split evenly
// across len(gpus) GPUs.
func (b *Benchmark) SelectGPU(gpus []int) {
	b.gpus = gpus
}

// SetUnifiedMemory is unsupported: K-splitting relies on each GPU holding a
// private slice of A and B rather than one memory space shared by all GPUs.
func (b *Benchmark) SetUnifiedMemory() {
	log.Panic("allreducegemm does not support unified memory")
}

// EnableVerification turns on CPU-reference verification for Verify().
func (b *Benchmark) EnableVerification() {
	b.verify = true
}

// Run runs the benchmark.
func (b *Benchmark) Run() {
	b.loadKernel()
	b.initMem()
	b.exec()
}

func (b *Benchmark) loadKernel() {
	b.kernel = kernels.LoadProgramFromMemory(hsacoBytes, "mmmKernel_local")
	if b.kernel == nil {
		log.Panic("failed to load mmmKernel_local kernel binary")
	}
}

func (b *Benchmark) initMem() {
	rand.Seed(0)

	b.matrixA = newMatrix(b.K, b.M)
	for i := range b.matrixA.Data {
		b.matrixA.Data[i] = rand.Float32()
	}

	b.matrixB = newMatrix(b.N, b.K)
	for i := range b.matrixB.Data {
		b.matrixB.Data[i] = rand.Float32()
	}
}

func (b *Benchmark) exec() {
	numGPU := uint32(len(b.gpus))
	b.validateDimensions(numGPU)

	kSlice := b.K / numGPU
	partialC := make([]driver.Ptr, numGPU)
	gAs := make([]driver.Ptr, numGPU)
	gBs := make([]driver.Ptr, numGPU)
	cmdQs := make([]*driver.CommandQueue, numGPU)

	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)

		kStart := uint32(g) * kSlice
		aSlice := b.matrixA.columnSlice(kStart, kStart+kSlice)
		bSlice := b.matrixB.rowSlice(kStart, kStart+kSlice)

		gA := b.driver.AllocateMemory(
			b.context, uint64(aSlice.Width*aSlice.Height*4))
		gB := b.driver.AllocateMemory(
			b.context, uint64(bSlice.Width*bSlice.Height*4))
		gC := b.driver.AllocateMemory(b.context, uint64(b.N*b.M*4))

		b.driver.MemCopyH2D(b.context, gA, aSlice.Data)
		b.driver.MemCopyH2D(b.context, gB, bSlice.Data)

		gAs[g] = gA
		gBs[g] = gB
		partialC[g] = gC
		cmdQs[g] = b.driver.CreateCommandQueue(b.context)
	}

	// mmmKernel_local's HiddenGlobalOffsetY parameter does not work
	// correctly in this simulator (verified separately: even a single
	// workgroup-row launch with a nonzero Y offset silently produces
	// zeros), and a grid with more than one workgroup-row in Y produces
	// wrong results outright. So instead of one launch covering the full M
	// rows with a Y offset, every GPU issues one launch per rowBand-row
	// band, always with a single-workgroup-row grid and a zero offset,
	// retargeting MatrixA/MatrixC via plain pointer arithmetic instead.
	//
	// Bands are issued round-robin across GPUs -- one launch per GPU, then
	// drain every GPU's queue together, then the next band -- rather than
	// batching all of one GPU's bands before draining. That keeps genuine
	// cross-GPU overlap (several GPUs have a launch outstanding at once)
	// while ensuring no single queue ever accumulates more than one
	// undrained launch, which avoids a hang observed when GPUs had several
	// simultaneously-outstanding, undrained launches each.
	for rowStart := uint32(0); rowStart < b.M; rowStart += rowBand {
		for g, gpuID := range b.gpus {
			b.driver.SelectGPU(b.context, gpuID)

			aOffset := uint64(rowStart) * uint64(kSlice) * 4
			cOffset := uint64(rowStart) * uint64(b.N) * 4

			kernArgs := &matrixmultiplication.KernelArgs{
				MatrixA: gAs[g] + driver.Ptr(aOffset),
				MatrixB: gBs[g],
				MatrixC: partialC[g] + driver.Ptr(cOffset),
				WidthA:  kSlice,
				BlockA:  32 * 32 * 4,
			}
			b.driver.EnqueueLaunchKernel(
				cmdQs[g],
				b.kernel,
				[3]uint32{b.N / 4, rowBand / 4, 1},
				[3]uint16{tileSize, tileSize, 1},
				kernArgs,
			)
		}

		for _, q := range cmdQs {
			b.driver.DrainCommandQueue(q)
		}
	}

	b.allReduce(partialC)
	b.collectResult(partialC[0])
}

// validateDimensions checks the constraints mmmKernel_local's tiling
// requires: M, N, and each GPU's K slice must divide cleanly, or the
// kernel's internal loop counts silently truncate and produce wrong
// results rather than erroring.
func (b *Benchmark) validateDimensions(numGPU uint32) {
	if b.K%numGPU != 0 {
		log.Panicf("K (%d) must be divisible by the number of GPUs (%d)",
			b.K, numGPU)
	}

	kSlice := b.K / numGPU
	tile := uint32(4 * tileSize)

	if b.M%rowBand != 0 {
		log.Panicf("M (%d) must be a multiple of %d", b.M, uint32(rowBand))
	}

	if b.N%tile != 0 {
		log.Panicf("N (%d) must be a multiple of %d", b.N, tile)
	}

	if kSlice%tile != 0 {
		log.Panicf(
			"K/numGPU (%d) must be a multiple of %d", kSlice, tile)
	}
}

// allReduce sums the numGPU partial products in place. mccl's AllReduce
// computes the mean, not the sum, of its inputs, so the result is scaled
// back up by numGPU in collectResult.
func (b *Benchmark) allReduce(partialC []driver.Ptr) {
	numGPU := len(b.gpus)
	comms := mccl.CommInitAll(numGPU, b.driver, b.context, b.gpus)

	dataSize := int(b.M * b.N)
	bufSize := dataSize
	if bufSize > allReduceBufElements {
		bufSize = allReduceBufElements
	}

	bufs := make([]driver.Ptr, numGPU)
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		bufs[g] = b.driver.AllocateMemory(b.context, uint64(bufSize*4))
	}

	if b.Algorithm == AlgorithmNaive {
		mccl.AllReduceNaive(b.driver, comms, partialC, dataSize, bufs, bufSize)
	} else {
		mccl.AllReduceRing(b.driver, comms, partialC, dataSize, bufs, bufSize)
	}
}

func (b *Benchmark) collectResult(reducedC driver.Ptr) {
	b.matrixC = newMatrix(b.N, b.M)
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.driver.MemCopyD2H(b.context, b.matrixC.Data, reducedC)

	numGPU := float32(len(b.gpus))
	for i := range b.matrixC.Data {
		b.matrixC.Data[i] *= numGPU
	}
}

// Verify checks the GPU result against a CPU reference implementation.
func (b *Benchmark) Verify() {
	if !b.verify {
		return
	}

	for row := uint32(0); row < b.M; row++ {
		for col := uint32(0); col < b.N; col++ {
			var sum float32
			for k := uint32(0); k < b.K; k++ {
				sum += b.matrixA.Data[row*b.K+k] * b.matrixB.Data[k*b.N+col]
			}

			got := b.matrixC.Data[row*b.N+col]
			tolerance := float32(1e-2) * (float32(math.Abs(float64(sum))) + 1)
			if float32(math.Abs(float64(sum-got))) > tolerance {
				log.Panicf(
					"mismatch at [%d, %d]: expected %f, got %f",
					row, col, sum, got)
			}
		}
	}

	log.Print("Passed!")
}
