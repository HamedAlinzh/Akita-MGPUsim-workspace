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

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/kernels"
)

//go:embed kernels.hsaco
var hsacoBytes []byte

// tileSize is the local work-group edge length (8x8) that
// mmmKernel_local's tiling loop assumes; M, N, and each GPU's K slice must
// be a multiple of 4*tileSize for the kernel's internal loop counts to come
// out exact.
const tileSize = 8

// defaultAllReduceChunk caps the AllReduce scratch buffer at a fixed chunk
// size so memory use does not grow with M*N, matching the convention used by
// the DNN data-parallel trainer (benchmarks/dnn/gputraining).
const defaultAllReduceChunk = 65536

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

	// BandRows splits the M output rows into bands of this many rows, each
	// computed by its own GEMM launch and all-reduced on its own, so a
	// band's All-Reduce message is BandRows*N floats. 0 means one band
	// covering all of M. Must divide M and be a multiple of 4*tileSize.
	BandRows uint32

	// Overlap pipelines the bands: band r is all-reduced while band r+1's
	// GEMM is running. Without it, each band's GEMM finishes and its
	// All-Reduce completes before the next band starts, which is the
	// no-overlap control at the same message size.
	Overlap bool

	// AllReduceChunk is the All-Reduce scratch buffer size in elements; a
	// band larger than this is all-reduced in several chunks, each a full
	// round of ring steps. 0 means defaultAllReduceChunk.
	AllReduceChunk uint32

	// PerBandScratch gives every band its own All-Reduce scratch buffers
	// instead of one set reused by all bands.
	PerBandScratch bool

	kernel *insts.HsaCo

	comms []*mccl.Communicator
	bufs  [][]driver.Ptr // [scratch set][GPU]

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

	b.initAllReduce()

	bandRows := b.bandRows()
	launchBand := func(rowStart uint32) {
		for g, gpuID := range b.gpus {
			b.driver.SelectGPU(b.context, gpuID)

			aOffset := uint64(rowStart) * uint64(kSlice) * 4
			cOffset := uint64(rowStart) * uint64(b.N) * 4

			kernArgs := &matrixmultiplication.KernelArgs{
				MatrixA: gAs[g] + driver.Ptr(aOffset),
				MatrixB: gBs[g],
				MatrixC: partialC[g] + driver.Ptr(cOffset),
				WidthA:  kSlice,
				BlockA:  (4 * tileSize) * (4 * tileSize) * 4,
			}
			b.driver.EnqueueLaunchKernel(
				cmdQs[g],
				b.kernel,
				[3]uint32{b.N / 4, bandRows / 4, 1},
				[3]uint16{tileSize, tileSize, 1},
				kernArgs,
			)
		}
	}
	drain := func() {
		for _, q := range cmdQs {
			b.driver.DrainCommandQueue(q)
		}
	}

	// A band may span several workgroup rows. That relies on the patched
	// kernels.hsaco (see native/patch_gidy.py): upstream mmmKernel_local
	// indexes A by the local Y id, so every workgroup row past the first
	// read the wrong rows of A.
	//
	// With Overlap, band r's GEMM is left running on every GPU while the
	// host drives band r-1's All-Reduce on mccl's own command queues; the
	// CP's multiple dispatchers let the two kernels share the GPU. The
	// bands write disjoint rows of partialC, and band r-1's GEMM was drained
	// before its All-Reduce starts. Each GEMM queue still holds at most one
	// undrained launch at a time.
	//
	// The phase timers split the simulated time between the first launch
	// and the last All-Reduce into time the host spends waiting for GEMM
	// drains and time it spends inside All-Reduce calls. Without overlap
	// these are the GEMM and All-Reduce times; with overlap, the drain wait
	// is the GEMM time the All-Reduce did not hide.
	var gemmWait, allReduceTime sim.VTimeInSec
	timed := func(acc *sim.VTimeInSec, f func()) {
		start := b.driver.Engine.CurrentTime()
		f()
		*acc += b.driver.Engine.CurrentTime() - start
	}
	reduceBand := func(rowStart uint32) {
		timed(&allReduceTime, func() {
			b.allReduce(partialC, rowStart, bandRows)
		})
	}

	windowStart := b.driver.Engine.CurrentTime()
	for rowStart := uint32(0); rowStart < b.M; rowStart += bandRows {
		launchBand(rowStart)

		if !b.Overlap {
			timed(&gemmWait, drain)
			reduceBand(rowStart)
			continue
		}

		if rowStart > 0 {
			reduceBand(rowStart - bandRows)
		}
		timed(&gemmWait, drain)
	}

	if b.Overlap {
		reduceBand(b.M - bandRows)
	}
	window := b.driver.Engine.CurrentTime() - windowStart

	log.Printf("allreducegemm phases: window=%.9f gemm_wait=%.9f "+
		"allreduce=%.9f", window, gemmWait, allReduceTime)

	b.collectResult(partialC[0])
}

func (b *Benchmark) bandRows() uint32 {
	if b.BandRows == 0 {
		return b.M
	}
	return b.BandRows
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

	if b.M%tile != 0 {
		log.Panicf("M (%d) must be a multiple of %d", b.M, tile)
	}

	if b.N%tile != 0 {
		log.Panicf("N (%d) must be a multiple of %d", b.N, tile)
	}

	if kSlice%tile != 0 {
		log.Panicf(
			"K/numGPU (%d) must be a multiple of %d", kSlice, tile)
	}

	bandRows := b.bandRows()
	if bandRows%tile != 0 || b.M%bandRows != 0 {
		log.Panicf(
			"BandRows (%d) must be a multiple of %d and divide M (%d)",
			bandRows, tile, b.M)
	}
}

// initAllReduce creates the communicators and the per-GPU scratch buffers:
// one set shared by every band's All-Reduce, or one set per band with
// PerBandScratch.
func (b *Benchmark) initAllReduce() {
	numGPU := len(b.gpus)
	b.comms = mccl.CommInitAll(numGPU, b.driver, b.context, b.gpus)

	numSets := 1
	if b.PerBandScratch {
		numSets = int(b.M / b.bandRows())
	}

	b.bufs = make([][]driver.Ptr, numSets)
	for s := range b.bufs {
		b.bufs[s] = make([]driver.Ptr, numGPU)
		for g, gpuID := range b.gpus {
			b.driver.SelectGPU(b.context, gpuID)
			b.bufs[s][g] = b.driver.AllocateMemory(
				b.context, uint64(b.allReduceBufSize()*4))
		}
	}
}

func (b *Benchmark) allReduceBufSize() int {
	chunk := int(b.AllReduceChunk)
	if chunk == 0 {
		chunk = defaultAllReduceChunk
	}

	bandSize := int(b.bandRows() * b.N)
	if bandSize > chunk {
		return chunk
	}
	return bandSize
}

// allReduce sums rows [rowStart, rowStart+numRows) of the numGPU partial
// products in place. mccl's AllReduce computes the mean, not the sum, of
// its inputs, so the result is scaled back up by numGPU in collectResult.
func (b *Benchmark) allReduce(
	partialC []driver.Ptr,
	rowStart, numRows uint32,
) {
	offset := driver.Ptr(uint64(rowStart) * uint64(b.N) * 4)
	data := make([]driver.Ptr, len(partialC))
	for g := range partialC {
		data[g] = partialC[g] + offset
	}

	dataSize := int(numRows * b.N)
	bufSize := b.allReduceBufSize()
	bufs := b.bufs[0]
	if b.PerBandScratch {
		bufs = b.bufs[rowStart/b.bandRows()]
	}

	if b.Algorithm == AlgorithmNaive {
		mccl.AllReduceNaive(b.driver, b.comms, data, dataSize, bufs, bufSize)
	} else {
		mccl.AllReduceRing(b.driver, b.comms, data, dataSize, bufs, bufSize)
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

	m := mismatchSummary{}
	for row := uint32(0); row < b.M; row++ {
		for col := uint32(0); col < b.N; col++ {
			var sum float32
			for k := uint32(0); k < b.K; k++ {
				sum += b.matrixA.Data[row*b.K+k] * b.matrixB.Data[k*b.N+col]
			}

			got := b.matrixC.Data[row*b.N+col]
			tolerance := float32(1e-2) * (float32(math.Abs(float64(sum))) + 1)
			if float32(math.Abs(float64(sum-got))) > tolerance {
				m.add(b, row, col, sum, got)
			}
		}
	}

	if m.count > 0 {
		m.report(b)
		log.Panicf("mismatch at [%d, %d]: expected %f, got %f",
			m.firstRow, m.firstCol, m.firstWant, m.firstGot)
	}

	log.Print("Passed!")
}

// mismatchSummary describes where a failed verification went wrong: which
// bands and which per-GPU ring chunks within a band, and how far off.
type mismatchSummary struct {
	count               int
	firstRow, firstCol  uint32
	firstWant, firstGot float32
	minRatio, maxRatio  float64
	perBandChunk        map[[2]int]int
	rowsWithErrors      map[uint32]bool
}

func (m *mismatchSummary) add(b *Benchmark, row, col uint32, want, got float32) {
	ratio := float64(got) / float64(want)
	if m.count == 0 {
		m.firstRow, m.firstCol, m.firstWant, m.firstGot = row, col, want, got
		m.minRatio, m.maxRatio = ratio, ratio
		m.perBandChunk = make(map[[2]int]int)
		m.rowsWithErrors = make(map[uint32]bool)
	}
	m.count++
	m.minRatio = math.Min(m.minRatio, ratio)
	m.maxRatio = math.Max(m.maxRatio, ratio)
	m.rowsWithErrors[row] = true

	bandRows := b.bandRows()
	band := row / bandRows
	elem := (row-band*bandRows)*b.N + col
	numGPU := uint32(len(b.gpus))
	chunkElems := (bandRows*b.N + numGPU - 1) / numGPU
	m.perBandChunk[[2]int{int(band), int(elem / chunkElems)}]++
}

func (m *mismatchSummary) report(b *Benchmark) {
	log.Printf("verify: %d of %d elements wrong, in %d rows; got/expected "+
		"between %.3f and %.3f", m.count, b.M*b.N, len(m.rowsWithErrors),
		m.minRatio, m.maxRatio)
	for band := 0; band < int(b.M/b.bandRows()); band++ {
		for chunk := 0; chunk < len(b.gpus); chunk++ {
			if n := m.perBandChunk[[2]int{band, chunk}]; n > 0 {
				log.Printf("verify: band %d, ring chunk %d: %d wrong", band, chunk, n)
			}
		}
	}
}
