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
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/kernels"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
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
	// AlgorithmDirect uses mccl.AllReduceDirect (direct reduce-scatter +
	// all-gather, for fully connected fabrics).
	AlgorithmDirect Algorithm = "direct"
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

	// TileSize is the work-group edge length T; each work-group computes a
	// 4T x 4T output tile. 0 means tileSize.
	TileSize uint32

	// ReduceScatter replaces the mccl All-Reduce with a ring reduce-scatter
	// (one band, ring algorithm only); the result is assembled from each
	// element's owner. With Comm, set CommOptions.Collective instead.
	ReduceScatter bool

	// Comm, if set, reduces C with the communication engines instead of
	// mccl: the host arms them, launches one GEMM per GPU and waits. Each
	// finished tile group is reduced while the GEMM still runs (or after it,
	// in bulk mode). BandRows and Overlap do not apply.
	Comm        *commengine.Session
	CommOptions commengine.Options

	// GEMMWGCap and GEMMVALUScale are simulation hints for the GEMM kernels
	// only (insts.HsaCo.WGCap and VALUCyclesScale; 0 = platform default):
	// cap the GEMM's work-groups in flight per CU, and scale its vector ALU
	// cycles. The collectives' kernels are not affected.
	GEMMWGCap     int
	GEMMVALUScale float64

	// SmallRows, if set, computes the first SmallRows rows of C with a
	// second GEMM launch at SmallTileSize (default 4), with its own
	// SmallVALUScale, launched before the main one on its own queue so its
	// small tiles finish early. With Comm, the small part's tiles are sent
	// SmallTilesPerMsg at a time and the main part's one tile per message.
	SmallRows        uint32
	SmallTileSize    uint32
	SmallVALUScale   float64
	SmallTilesPerMsg int

	// SplitKWays, if at least 2, splits the K loop of the first and the last
	// SplitKRows rows of C across that many work-groups (the hybrid
	// schedule of Stream-K): each of these bands runs as SplitKWays
	// launches, one per K chunk, on queues created before (first band) and
	// after (last band) the main GEMM's over the middle rows. Chunk 0
	// writes C, the others their own partial buffers, which the engines add
	// (Comm only; CommOptions.SplitSend sends them unsummed).
	SplitKWays int
	SplitKRows uint32

	// SplitKPlan, if set, replaces SplitKWays and SplitKRows with a split
	// factor for every row of tiles, launched top to bottom: each split
	// row as its own K chunks, consecutive unsplit rows as one launch.
	SplitKPlan []int

	// SWBands, if set, runs the software-overlap baseline instead of the
	// engines: C is cut into bands of these many rows, every band's GEMM is
	// launched up front (split per SplitKPlan, which must be uniform within a
	// band), and the host waits for each band in turn, adds its split-K
	// partials with mccl.Accumulate and all-reduces it with mccl while later
	// bands compute. Needs ordered dispatch, under which the collectives'
	// kernels run beside the GEMM's.
	SWBands []uint32

	// FlashGroups, with Flash, runs the FlashOverlap-style software baseline
	// (arXiv 2504.19519) instead of the engines: the GEMM is launched as
	// with the engines (split per SplitKPlan), and the trackers count its
	// tiles in groups of these many rows, top to bottom. When a group's
	// last tile is in L2 on a GPU, its group signal is raised; the
	// communication kernels, enqueued before the GEMM and held by the CPs
	// until their signals, add the group's split-K partials and all-reduce
	// its rows with mccl's direct algorithm, on the CUs reserved for
	// communication. Needs ordered dispatch.
	FlashGroups []uint32
	Flash       *FlashSetup

	// Warmup runs the GEMM (with Comm or Flash) once, untimed and without
	// the engines, before the measured launch, so it starts with warm
	// caches and TLBs. WarmupSkipB reads a dummy buffer in place of B in
	// that run, so A and C are warm and B is not, as for a layer whose
	// weights stream from memory.
	Warmup      bool
	WarmupSkipB bool

	// AllGather, if set, runs the input side of a tensor-parallel layer
	// instead of the K-split GEMM (allgather.go): A's rows are gathered to
	// every GPU, which computes C = A x B_local. K is then every GPU's full
	// K, and B is K x pN. AllGatherSignals is the CPs' shared signal board,
	// needed with the engines (AllGatherRows, AllGatherBulk).
	AllGather        AllGatherMode
	AllGatherSignals *signal.Board
	gatherC          [][]float32

	// OnPartialC, if set, is called for each GPU once its partial C is
	// allocated and A and B are copied, before any GEMM launch.
	OnPartialC func(ctx *driver.Context, g, gpuID int, c driver.Ptr)

	kernel      *insts.HsaCo
	smallKernel *insts.HsaCo
	mainQueues  []*driver.CommandQueue

	comms []*mccl.Communicator
	bufs  [][]driver.Ptr // [scratch set][GPU]

	matrixA, matrixB, matrixC *matrix

	verify bool
}

// FlashSetup is what the FlashOverlap-style baseline needs from the
// platform: every GPU's tile tracker (in SelectGPU order), the CPs' shared
// signal board, and the signal delays (Local also for the group signals).
type FlashSetup struct {
	Trackers []*tiletracker.Tracker
	Signals  *signal.Board
	Delays   mccl.SignalDelays
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
	b.kernel.WGCap = b.GEMMWGCap
	b.kernel.VALUCyclesScale = b.GEMMVALUScale

	if b.SmallRows > 0 {
		// A separate code object, so its hints are its own.
		b.smallKernel = kernels.LoadProgramFromMemory(hsacoBytes, "mmmKernel_local")
		b.smallKernel.WGCap = b.GEMMWGCap
		b.smallKernel.VALUCyclesScale = b.SmallVALUScale
	}
}

func (b *Benchmark) initMem() {
	// A source of its own: since Go 1.24 rand.Seed does nothing, so the
	// global source gave different matrices every run.
	r := rand.New(rand.NewSource(0))

	b.matrixA = newMatrix(b.K, b.M)
	for i := range b.matrixA.Data {
		b.matrixA.Data[i] = r.Float32()
	}

	bCols := b.N
	if b.AllGather != "" {
		bCols *= uint32(len(b.gpus))
	}
	b.matrixB = newMatrix(bCols, b.K)
	for i := range b.matrixB.Data {
		b.matrixB.Data[i] = r.Float32()
	}
}

func (b *Benchmark) exec() {
	if b.AllGather != "" {
		b.execAllGather()
		return
	}
	numGPU := uint32(len(b.gpus))
	b.validateDimensions(numGPU)

	kSlice := b.K / numGPU
	partialC := make([]driver.Ptr, numGPU)
	gAs := make([]driver.Ptr, numGPU)
	gBs := make([]driver.Ptr, numGPU)
	cmdQs := make([]*driver.CommandQueue, numGPU)
	var smallQs []*driver.CommandQueue
	splits := make([]*splitK, numGPU)

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
		// The driver serves queues in creation order, so the small-tile
		// queue is created first for its launch to start first.
		if b.SmallRows > 0 {
			smallQs = append(smallQs, b.driver.CreateCommandQueue(b.context))
		}
		if b.splitKOn() {
			splits[g] = b.newSplitK(aSlice, bSlice, gB)
		}
		cmdQs[g] = b.driver.CreateCommandQueue(b.context)
		b.mainQueues = cmdQs
		if b.splitKOn() {
			splits[g].createLaterQueues(b)
		}

		if b.OnPartialC != nil {
			b.OnPartialC(b.context, g, gpuID, gC)
		}
	}

	if b.Comm == nil {
		b.initAllReduce()
	}

	bandRows := b.bandRows()
	// LocalPtr keeps the BlockA line in the form tune_gemm_tile.py expects.
	tileSize := driver.LocalPtr(b.tileSize())
	launchBand := func(rowStart uint32) {
		if b.SmallRows > 0 {
			b.launchSmallFirst(gAs, gBs, partialC, cmdQs, smallQs, kSlice)
			return
		}
		if b.splitKOn() {
			b.launchSplitK(gAs, gBs, partialC, cmdQs, splits, kSlice)
			return
		}
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
				[3]uint16{uint16(tileSize), uint16(tileSize), 1},
				kernArgs,
			)
		}
	}
	drain := func() {
		for _, q := range smallQs {
			b.driver.DrainCommandQueue(q)
		}
		for _, sk := range splits {
			if sk == nil {
				continue
			}
			for _, qs := range sk.queues {
				for _, q := range qs {
					b.driver.DrainCommandQueue(q)
				}
			}
		}
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

	if len(b.SWBands) > 0 {
		b.execSWOverlap(partialC, launchBand, splits, timed)
		return
	}

	warm := func() {
		if !b.Warmup {
			return
		}
		if b.WarmupSkipB {
			defer b.dummyB(gBs, splits, kSlice)()
		}
		launchBand(0)
		drain()
	}
	if len(b.FlashGroups) > 0 {
		warm()
		b.execFlash(partialC, launchBand, drain, splits, timed)
		return
	}

	if b.Comm != nil {
		if b.splitKOn() {
			b.CommOptions.Partials = make([][]tiletracker.Partial, numGPU)
			for g, sk := range splits {
				b.CommOptions.Partials[g] = sk.partials
			}
		}
		b.execFused(partialC, launchBand, drain, warm, timed)
		return
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

	if b.ReduceScatter {
		b.collectReduceScatter(partialC)
		return
	}
	b.collectResult(partialC[0])
}

// execSWOverlap launches every band's GEMM, then for each band in turn waits
// for it, sums its split-K partials into C and all-reduces its rows with
// mccl, on mccl's own queues, while later bands compute. The window runs
// from the launch to the last all-reduce.
func (b *Benchmark) execSWOverlap(
	partialC []driver.Ptr,
	launch func(rowStart uint32),
	splits []*splitK,
	timed func(acc *sim.VTimeInSec, f func()),
) {
	var gemmWait, presum, allReduceTime sim.VTimeInSec
	windowStart := b.driver.Engine.CurrentTime()
	launch(0)
	for i, band := range splits[0].bands {
		timed(&gemmWait, func() {
			for g, sk := range splits {
				qs := sk.queues[i]
				if qs == nil {
					qs = []*driver.CommandQueue{b.mainQueues[g]}
				}
				for _, q := range qs {
					b.driver.DrainCommandQueue(q)
				}
			}
		})
		offset := driver.Ptr(uint64(band.row) * uint64(b.N) * 4)
		for j := 1; j < band.s; j++ {
			dst := make([]driver.Ptr, len(splits))
			src := make([]driver.Ptr, len(splits))
			for g, sk := range splits {
				dst[g] = partialC[g] + offset
				src[g] = driver.Ptr(sk.bandPart[i][j-1].VBase)
			}
			timed(&presum, func() {
				mccl.Accumulate(b.driver, b.comms, dst, src, int(band.rows*b.N))
			})
		}
		timed(&allReduceTime, func() {
			b.allReduce(partialC, band.row, band.rows)
		})
	}
	window := b.driver.Engine.CurrentTime() - windowStart

	log.Printf("allreducegemm phases: window=%.9f gemm_wait=%.9f "+
		"allreduce=%.9f presum=%.9f", window, gemmWait, allReduceTime, presum)
	b.collectResult(partialC[0])
}

// execFlash runs the FlashOverlap-style baseline (FlashGroups): it arms the
// trackers to raise each GPU's group signals, enqueues every group's
// communication kernels (held by the CPs until their signals), waits until
// all of them are held, then launches the GEMM. The window runs from the
// launch to the last communication kernel completing.
func (b *Benchmark) execFlash(
	partialC []driver.Ptr,
	launch func(rowStart uint32),
	drain func(),
	splits []*splitK,
	timed func(acc *sim.VTimeInSec, f func()),
) {
	fs := b.Flash
	p := len(b.gpus)
	edge := 4 * b.tileSize()
	tilesX := b.N / edge
	groupOf := make([]int, b.M/edge) // by tile row
	var groupRow []uint32
	row := uint32(0)
	for g, rows := range b.FlashGroups {
		for r := row; r < row+rows; r += edge {
			groupOf[r/edge] = g
		}
		groupRow = append(groupRow, row)
		row += rows
	}
	grp := func(gpu, g int) string { return fmt.Sprintf("gpu%d/group%d", gpu, g) }

	for i, tr := range fs.Trackers {
		region := commengine.TrackRegion(b.driver, b.context, uint64(partialC[i]),
			b.M, b.N, edge, splits[i].partials)
		region.IgnoreLate = true
		tr.Arm(region)
		tr.ClearListeners()
		gpu, count := i, make([]int, len(b.FlashGroups))
		tr.AddLocalListener(func(ev tiletracker.TileEvent) {
			g := groupOf[ev.Row]
			count[g]++
			if count[g] == int(b.FlashGroups[g]/edge*tilesX) {
				fs.Signals.Raise(ev.Complete, grp(gpu, g), fs.Delays.Local)
			}
		})
	}

	// Every group's kernels: split-K pre-sums, then the direct all-reduce.
	numKernels := len(b.FlashGroups) * 3 * p * (p - 1)
	for _, band := range splits[0].bands {
		numKernels += p * (band.s - 1)
	}
	allHeld := fs.Signals.ExpectHeld(numKernels)
	var queues []*driver.CommandQueue
	for g, rows := range b.FlashGroups {
		ready := make([]string, p)
		data := make([]driver.Ptr, p)
		bufs := make([]driver.Ptr, p)
		size := int(rows * b.N)
		for i, gpuID := range b.gpus {
			ready[i] = grp(i, g)
			for k, band := range splits[i].bands {
				if band.s == 1 || groupOf[band.row/edge] != g {
					continue
				}
				for j := 1; j < band.s; j++ {
					name := fmt.Sprintf("%s/acc%d.%d", grp(i, g), band.row/edge, j)
					queues = append(queues, mccl.AccumulateOnSignal(b.driver,
						b.comms[i], partialC[i]+driver.Ptr(uint64(band.row)*uint64(b.N)*4),
						driver.Ptr(splits[i].bandPart[k][j-1].VBase), int(band.rows*b.N),
						[]string{ready[i]}, name, fs.Delays.Local))
					ready[i] = name
				}
			}
			data[i] = partialC[i] + driver.Ptr(uint64(groupRow[g])*uint64(b.N)*4)
			b.driver.SelectGPU(b.context, gpuID)
			bufs[i] = b.driver.AllocateMemory(b.context,
				uint64((p-1)*((size+p-1)/p)*4))
		}
		qs, _ := mccl.AllReduceDirectOnSignal(b.driver, b.comms, data, size, bufs,
			ready, fmt.Sprintf("group%d", g), fs.Delays)
		queues = append(queues, qs...)
	}
	b.driver.WaitFor(allHeld)
	b.driver.WaitIdle()

	var gemmWait sim.VTimeInSec
	windowStart := b.driver.Engine.CurrentTime()
	launch(0)
	timed(&gemmWait, drain)
	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}
	hostEnd := b.driver.Engine.CurrentTime() - windowStart
	for _, tr := range fs.Trackers {
		tr.Disarm()
	}

	end := windowStart
	recs := fs.Signals.Records()
	for _, r := range recs {
		end = max(end, r.Done)
	}
	firstSignal := sim.VTimeInSec(-1)
	for g := range b.FlashGroups {
		for i := range b.gpus {
			t, _ := fs.Signals.RaisedAt(grp(i, g))
			if firstSignal < 0 || t < firstSignal {
				firstSignal = t
			}
			log.Printf("allreducegemm flash group: group=%d gpu=%d rows=%d raised=%.9f",
				g, i, b.FlashGroups[g], t-windowStart)
		}
	}
	log.Printf("allreducegemm flash phases: window=%.9f gemm_wait=%.9f "+
		"host_end=%.9f first_signal=%.9f kernels=%d", end-windowStart, gemmWait,
		hostEnd, firstSignal-windowStart, len(recs))
	for _, r := range recs {
		log.Printf("allreducegemm flash kernel: name=%s released=%.9f done=%.9f",
			r.Name, r.Released-windowStart, r.Done-windowStart)
	}
	b.driver.WaitIdle()
	b.collectResult(partialC[0])
}

// dummyB points the GEMM's B (and the split chunks' B) at a fresh buffer of
// the same size on every GPU, and returns a function that restores them.
func (b *Benchmark) dummyB(gBs []driver.Ptr, splits []*splitK, kSlice uint32) func() {
	orig := append([]driver.Ptr(nil), gBs...)
	saved := make([]map[int][]driver.Ptr, len(splits))
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		dummy := b.driver.AllocateMemory(b.context, uint64(kSlice*b.N*4))
		gBs[g] = dummy
		if sk := splits[g]; sk != nil {
			saved[g] = sk.bChunks
			sk.bChunks = map[int][]driver.Ptr{}
			for s, chunks := range saved[g] {
				for _, c := range chunks {
					sk.bChunks[s] = append(sk.bChunks[s], dummy+(c-orig[g]))
				}
			}
		}
	}
	return func() {
		copy(gBs, orig)
		for g, sk := range splits {
			if sk != nil {
				sk.bChunks = saved[g]
			}
		}
	}
}

// collectReduceScatter assembles C from the GPU that holds each element
// after mccl.ReduceScatterRing, scaling the mean to the sum.
func (b *Benchmark) collectReduceScatter(partialC []driver.Ptr) {
	numGPU := len(b.gpus)
	copies := make([][]float32, numGPU)
	for g, gpuID := range b.gpus {
		copies[g] = make([]float32, b.M*b.N)
		b.driver.SelectGPU(b.context, gpuID)
		b.driver.MemCopyD2H(b.context, copies[g], partialC[g])
	}

	b.matrixC = newMatrix(b.N, b.M)
	size := int(b.M * b.N)
	for i := range b.matrixC.Data {
		owner := mccl.ReduceScatterRingOwner(i, size, b.allReduceBufSize(), numGPU)
		b.matrixC.Data[i] = copies[owner][i] * float32(numGPU)
	}
}

// execFused arms the engines, launches the GEMM on every GPU and waits until
// the engines have reduced all of C. The window runs from the launch to the
// last engine finishing; comm_tail is the part after the GEMM drained.
func (b *Benchmark) execFused(
	partialC []driver.Ptr,
	launch func(rowStart uint32),
	drain, warm func(),
	timed func(acc *sim.VTimeInSec, f func()),
) {
	if b.BandRows != 0 && b.BandRows != b.M || b.Overlap {
		log.Panic("the communication engine runs one GEMM over all of M; " +
			"do not set BandRows or Overlap")
	}

	warm()

	tileEdge := 4 * b.tileSize()
	if b.SmallRows > 0 {
		tileEdge = 4 * b.smallTileSize()
		b.CommOptions.Groups = b.smallFirstGroups(tileEdge)
	}
	b.Comm.Arm(b.context, partialC, b.M, b.N, tileEdge, b.CommOptions)

	var gemmWait sim.VTimeInSec
	windowStart := b.driver.Engine.CurrentTime()
	launch(0)
	timed(&gemmWait, drain)
	doneAt := b.Comm.Wait()
	window := max(doneAt, windowStart+gemmWait) - windowStart

	log.Printf("allreducegemm fused phases: window=%.9f gemm_wait=%.9f "+
		"comm_tail=%.9f first_ready=%.9f", window, gemmWait, window-gemmWait,
		b.Comm.FirstReady()-windowStart)
	b.logMessages()

	b.collectFused(partialC, tileEdge)
}

// logMessages prints message counts per direction and the first messages'
// timeline.
func (b *Benchmark) logMessages() {
	type dir struct{ from, to int }
	count := map[dir]int{}
	bytes := map[dir]int{}
	recs := b.Comm.Records()
	for _, r := range recs {
		d := dir{r.From, r.To}
		count[d]++
		bytes[d] += r.TrafficBytes
	}
	for d, n := range count {
		log.Printf("allreducegemm messages: from=%d to=%d count=%d "+
			"traffic_bytes=%d", d.from, d.to, n, bytes[d])
	}
	for i, r := range recs {
		if i >= 8 {
			break
		}
		log.Printf("allreducegemm msg: group=%d reduced=%v from=%d to=%d "+
			"queued=%.9f sent=%.9f received=%.9f",
			r.Group, r.Reduced, r.From, r.To, r.Queued, r.Sent, r.Received)
	}
}

// collectFused copies the reduced C to the host. After an all-reduce every
// GPU holds it; after a reduce-scatter each tile group is taken from its
// owner. The engines produce the sum, not the mean, so nothing is scaled.
func (b *Benchmark) collectFused(partialC []driver.Ptr, tileEdge uint32) {
	b.matrixC = newMatrix(b.N, b.M)
	if b.CommOptions.Collective == commengine.AllReduce {
		b.driver.SelectGPU(b.context, b.gpus[0])
		b.driver.MemCopyD2H(b.context, b.matrixC.Data, partialC[0])
		return
	}

	numGPU := len(b.gpus)
	copies := make([][]float32, numGPU)
	for g, gpuID := range b.gpus {
		copies[g] = make([]float32, b.M*b.N)
		b.driver.SelectGPU(b.context, gpuID)
		b.driver.MemCopyD2H(b.context, copies[g], partialC[g])
	}

	tilesX := b.N / tileEdge
	ownerOf := b.CommOptions.TileOwner(int((b.M/tileEdge)*tilesX), numGPU)
	for row := uint32(0); row < b.M; row++ {
		for col := uint32(0); col < b.N; col++ {
			tile := int((row/tileEdge)*tilesX + col/tileEdge)
			owner := ownerOf(tile)
			b.matrixC.Data[row*b.N+col] = copies[owner][row*b.N+col]
		}
	}
}

// splitBand is a band of rows of C computed by one launch (s = 1) or by s
// launches, one per K chunk.
type splitBand struct {
	row, rows uint32
	s         int
}

func (b *Benchmark) splitKOn() bool {
	return b.SplitKWays > 1 || len(b.SplitKPlan) > 0 || len(b.SWBands) > 0 ||
		len(b.FlashGroups) > 0
}

// splitBands lists the bands in launch order.
func (b *Benchmark) splitBands() []splitBand {
	edge := 4 * b.tileSize()
	if len(b.SWBands) > 0 {
		var bands []splitBand
		row := uint32(0)
		for _, rows := range b.SWBands {
			s := 1
			if len(b.SplitKPlan) > 0 {
				s = b.SplitKPlan[row/edge]
				for r := row; r < row+rows; r += edge {
					if b.SplitKPlan[r/edge] != s {
						log.Panicf("SplitKPlan must be uniform within each SWBands band")
					}
				}
			}
			bands = append(bands, splitBand{row, rows, s})
			row += rows
		}
		return bands
	}
	if len(b.SplitKPlan) == 0 && len(b.FlashGroups) > 0 {
		return []splitBand{{0, b.M, 1}}
	}
	if len(b.SplitKPlan) == 0 {
		r, s := b.SplitKRows, b.SplitKWays
		return []splitBand{{0, r, s}, {r, b.M - 2*r, 1}, {b.M - r, r, s}}
	}
	var bands []splitBand
	for i, s := range b.SplitKPlan {
		row := uint32(i) * edge
		if n := len(bands); s == 1 && n > 0 && bands[n-1].s == 1 {
			bands[n-1].rows += edge
			continue
		}
		bands = append(bands, splitBand{row, edge, s})
	}
	return bands
}

// splitK is one GPU's split-K state: A cut into K chunks for every split
// factor, the partial C buffers of chunks 1.. of every split band, and each
// band's queues (nil for the first unsplit band, which uses the main one).
type splitK struct {
	bands    []splitBand
	aChunks  map[int][]driver.Ptr // by s: M x K/s each
	bChunks  map[int][]driver.Ptr // by s: offsets into the GPU's B
	partials []tiletracker.Partial
	bandPart [][]tiletracker.Partial
	queues   [][]*driver.CommandQueue
}

// newSplitK copies A in K chunks to the selected GPU, allocates the partial
// buffers and creates the queues of the bands before the first unsplit one.
func (b *Benchmark) newSplitK(aSlice, bSlice *matrix, gB driver.Ptr) *splitK {
	sk := &splitK{bands: b.splitBands(),
		aChunks: map[int][]driver.Ptr{}, bChunks: map[int][]driver.Ptr{}}
	for _, band := range sk.bands {
		if band.s == 1 || sk.aChunks[band.s] != nil {
			continue
		}
		s := uint32(band.s)
		chunk := aSlice.Width / s
		for j := uint32(0); j < s; j++ {
			a := aSlice.columnSlice(j*chunk, (j+1)*chunk)
			gA := b.driver.AllocateMemory(b.context, uint64(len(a.Data)*4))
			b.driver.MemCopyH2D(b.context, gA, a.Data)
			sk.aChunks[band.s] = append(sk.aChunks[band.s], gA)
			sk.bChunks[band.s] = append(sk.bChunks[band.s],
				gB+driver.Ptr(uint64(j*chunk)*uint64(bSlice.Width)*4))
		}
	}
	for _, band := range sk.bands {
		var parts []tiletracker.Partial
		for j := 1; j < band.s; j++ {
			buf := b.driver.AllocateMemory(b.context, uint64(band.rows*b.N*4))
			parts = append(parts, tiletracker.Partial{
				VBase: uint64(buf), Row: band.row, Rows: band.rows})
		}
		sk.bandPart = append(sk.bandPart, parts)
		sk.partials = append(sk.partials, parts...)
	}
	sk.queues = make([][]*driver.CommandQueue, len(sk.bands))
	for i, band := range sk.bands {
		if band.s == 1 {
			break
		}
		sk.createQueues(b, i)
	}
	return sk
}

func (sk *splitK) createQueues(b *Benchmark, i int) {
	for j := 0; j < sk.bands[i].s; j++ {
		sk.queues[i] = append(sk.queues[i], b.driver.CreateCommandQueue(b.context))
	}
}

// createLaterQueues creates the queues of the bands after the first
// unsplit one, which uses the main queue.
func (sk *splitK) createLaterQueues(b *Benchmark) {
	main := -1
	for i, band := range sk.bands {
		if main < 0 && band.s == 1 {
			main = i
			continue
		}
		if main >= 0 {
			sk.createQueues(b, i)
		}
	}
}

// launchSplitK launches, on every GPU, every band's launches in band order.
// With ordered dispatch the CP gives earlier kernels first claim on free CU
// slots.
func (b *Benchmark) launchSplitK(
	gAs, gBs, partialC []driver.Ptr,
	cmdQs []*driver.CommandQueue, splits []*splitK, kSlice uint32,
) {
	ts := b.tileSize()
	edge := driver.LocalPtr(4 * ts)
	launch := func(q *driver.CommandQueue, a, bm, c driver.Ptr,
		widthA, rowStart, rows uint32) {
		b.driver.EnqueueLaunchKernel(q, b.kernel,
			[3]uint32{b.N / 4, rows / 4, 1},
			[3]uint16{uint16(ts), uint16(ts), 1},
			&matrixmultiplication.KernelArgs{
				MatrixA: a + driver.Ptr(uint64(rowStart)*uint64(widthA)*4),
				MatrixB: bm,
				MatrixC: c,
				WidthA:  widthA,
				BlockA:  edge * edge * 4,
			})
	}
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		sk := splits[g]
		for i, band := range sk.bands {
			cBand := partialC[g] + driver.Ptr(uint64(band.row)*uint64(b.N)*4)
			if band.s == 1 {
				q := cmdQs[g]
				if sk.queues[i] != nil {
					q = sk.queues[i][0]
				}
				launch(q, gAs[g], gBs[g], cBand, kSlice, band.row, band.rows)
				continue
			}
			s := uint32(band.s)
			for j := 0; j < band.s; j++ {
				c := cBand
				if j > 0 {
					c = driver.Ptr(sk.bandPart[i][j-1].VBase)
				}
				launch(sk.queues[i][j], sk.aChunks[band.s][j], sk.bChunks[band.s][j],
					c, kSlice/s, band.row, band.rows)
			}
		}
	}
}

func (b *Benchmark) smallTileSize() uint32 {
	if b.SmallTileSize == 0 {
		return 4
	}
	return b.SmallTileSize
}

// launchSmallFirst launches, on every GPU, the small-tile GEMM over rows
// [0, SmallRows) on its own queue, then the main GEMM over the rest.
func (b *Benchmark) launchSmallFirst(
	gAs, gBs, partialC []driver.Ptr,
	cmdQs, smallQs []*driver.CommandQueue,
	kSlice uint32,
) {
	launch := func(q *driver.CommandQueue, co *insts.HsaCo, g int,
		rowStart, rows, ts uint32) {
		edge := driver.LocalPtr(4 * ts)
		b.driver.EnqueueLaunchKernel(q, co,
			[3]uint32{b.N / 4, rows / 4, 1},
			[3]uint16{uint16(ts), uint16(ts), 1},
			&matrixmultiplication.KernelArgs{
				MatrixA: gAs[g] + driver.Ptr(uint64(rowStart)*uint64(kSlice)*4),
				MatrixB: gBs[g],
				MatrixC: partialC[g] + driver.Ptr(uint64(rowStart)*uint64(b.N)*4),
				WidthA:  kSlice,
				BlockA:  edge * edge * 4,
			})
	}
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		launch(smallQs[g], b.smallKernel, g, 0, b.SmallRows, b.smallTileSize())
		launch(cmdQs[g], b.kernel, g, b.SmallRows, b.M-b.SmallRows, b.tileSize())
	}
}

// smallFirstGroups groups the tracker's tiles (edge x edge, the small tile)
// into messages: SmallTilesPerMsg consecutive small tiles in the small rows,
// and the small tiles covering one main tile elsewhere.
func (b *Benchmark) smallFirstGroups(edge uint32) [][]int {
	tilesX := b.N / edge
	per := 4 * b.tileSize() / edge // small tiles per main-tile edge
	n := max(1, b.SmallTilesPerMsg)
	var groups [][]int
	smallTiles := int(b.SmallRows / edge * tilesX)
	for t := 0; t < smallTiles; t += n {
		var g []int
		for i := t; i < min(t+n, smallTiles); i++ {
			g = append(g, i)
		}
		groups = append(groups, g)
	}
	for r := b.SmallRows / edge; r < b.M/edge; r += per {
		for c := uint32(0); c < tilesX; c += per {
			var g []int
			for dr := uint32(0); dr < per; dr++ {
				for dc := uint32(0); dc < per; dc++ {
					g = append(g, int((r+dr)*tilesX+c+dc))
				}
			}
			groups = append(groups, g)
		}
	}
	return groups
}

func (b *Benchmark) tileSize() uint32 {
	if b.TileSize == 0 {
		return tileSize
	}
	return b.TileSize
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
	if len(b.SWBands) > 0 {
		tile := 4 * b.tileSize()
		total := uint32(0)
		bad := b.Comm != nil || b.SmallRows > 0 || b.SplitKWays > 1 ||
			b.BandRows != 0 || b.Overlap || b.ReduceScatter ||
			len(b.SplitKPlan) > 0 && uint32(len(b.SplitKPlan))*tile != b.M
		for _, rows := range b.SWBands {
			bad = bad || rows == 0 || rows%tile != 0
			total += rows
		}
		if bad || total != b.M {
			log.Panicf("SWBands needs mccl (no engines), bands of multiples of %d rows "+
				"adding up to M, and at most a per-row SplitKPlan", tile)
		}
		for _, s := range b.SplitKPlan {
			if (b.K/numGPU)%(uint32(s)*tile) != 0 {
				log.Panicf("K/GPU must be a multiple of %d x every split factor", tile)
			}
		}
	} else if len(b.FlashGroups) > 0 {
		tile := 4 * b.tileSize()
		total := uint32(0)
		bad := b.Flash == nil || len(b.Flash.Trackers) != int(numGPU) ||
			b.Comm != nil || b.SmallRows > 0 || b.SplitKWays > 1 ||
			b.BandRows != 0 || b.Overlap || b.ReduceScatter ||
			len(b.SplitKPlan) > 0 && uint32(len(b.SplitKPlan))*tile != b.M
		for _, rows := range b.FlashGroups {
			bad = bad || rows == 0 || rows%tile != 0
			total += rows
		}
		for _, s := range b.SplitKPlan {
			bad = bad || s < 1 || (b.K/numGPU)%(uint32(s)*tile) != 0
		}
		if bad || total != b.M {
			log.Panicf("FlashGroups needs Flash (no engines), groups of multiples "+
				"of %d rows adding up to M, and at most a per-row SplitKPlan", tile)
		}
	} else if b.splitKOn() {
		tile := 4 * b.tileSize()
		bad := b.Comm == nil || b.SmallRows > 0 ||
			b.CommOptions.Placement != commengine.OwnerEngine
		if len(b.SplitKPlan) == 0 {
			bad = bad || b.SplitKRows%tile != 0 || b.SplitKRows == 0 ||
				2*b.SplitKRows >= b.M
		} else {
			bad = bad || b.SplitKWays > 1 || uint32(len(b.SplitKPlan))*tile != b.M
		}
		if !bad {
			for _, band := range b.splitBands() {
				bad = bad || band.s < 1 || (b.K/numGPU)%(uint32(band.s)*tile) != 0
			}
		}
		if bad {
			log.Panicf("split-K needs the engines (owner placement), and either "+
				"bands of a multiple of %d rows below M/2 or one split factor per "+
				"row of tiles, with K/GPU a multiple of %d x the factor", tile, tile)
		}
	}
	if b.SmallRows > 0 && (b.SmallRows%(4*b.tileSize()) != 0 ||
		b.SmallRows >= b.M || b.bandRows() != b.M || b.Overlap ||
		b.N%(4*b.smallTileSize()) != 0) {
		log.Panicf("SmallRows (%d) must be a multiple of %d below M, "+
			"with one band", b.SmallRows, 4*b.tileSize())
	}
	if b.ReduceScatter && (b.Algorithm == AlgorithmNaive ||
		b.bandRows() != b.M || b.Overlap) {
		log.Panic("ReduceScatter needs the ring algorithm and one band")
	}

	if b.K%numGPU != 0 {
		log.Panicf("K (%d) must be divisible by the number of GPUs (%d)",
			b.K, numGPU)
	}

	kSlice := b.K / numGPU
	tile := 4 * b.tileSize()

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

	if b.ReduceScatter {
		mccl.ReduceScatterRing(b.driver, b.comms, data, dataSize, bufs, bufSize)
		return
	}

	switch b.Algorithm {
	case AlgorithmNaive:
		mccl.AllReduceNaive(b.driver, b.comms, data, dataSize, bufs, bufSize)
	case AlgorithmDirect:
		mccl.AllReduceDirect(b.driver, b.comms, data, dataSize, bufs, bufSize)
	default:
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
// ResultC returns the all-reduced C after Run, row-major M x N.
func (b *Benchmark) ResultC() []float32 {
	return b.matrixC.Data
}

func (b *Benchmark) Verify() {
	if !b.verify {
		return
	}
	if b.AllGather != "" {
		b.verifyAllGather()
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
