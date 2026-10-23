package allreducegemm

import (
	"fmt"
	"log"
	"math"
	"sort"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/benchmarks/amdappsdk/matrixmultiplication"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
)

// AllGatherMode selects how the rows of A reach every GPU when the benchmark
// runs the input side of a tensor-parallel layer (Benchmark.AllGather).
//
// Every GPU holds M/p rows of A (full K) and its own K x N block of the
// columns of B (B is K x pN), and computes C = A x B_local over all M rows.
type AllGatherMode string

const (
	// AllGatherRows: the engines push each GPU's rows of A to every peer,
	// one row of tiles per message, and the GEMM runs as one launch per row
	// of tiles: local rows first, then remote rows in expected arrival
	// order, each held at the CP until its row's arrival signal.
	AllGatherRows AllGatherMode = "rows"
	// AllGatherBulk: the same push, and one GEMM launch held until every
	// remote row has arrived (gather, then GEMM).
	AllGatherBulk AllGatherMode = "bulk"
	// AllGatherLocal: A is already on every GPU; one GEMM launch (the floor).
	AllGatherLocal AllGatherMode = "local"
	// AllGatherMCCL: mccl's direct all-gather, then one GEMM launch.
	AllGatherMCCL AllGatherMode = "mccl"
)

func (b *Benchmark) validateAllGather(edge uint32) {
	p := uint32(len(b.gpus))
	engines := b.AllGather == AllGatherRows || b.AllGather == AllGatherBulk
	bad := b.M%(edge*p) != 0 || b.N%edge != 0 || b.K%edge != 0 ||
		engines && (b.Comm == nil || b.AllGatherSignals == nil) ||
		b.splitKOn() || b.SmallRows > 0 || b.BandRows != 0 || b.Overlap ||
		b.Warmup || b.ReduceScatter
	switch b.AllGather {
	case AllGatherRows, AllGatherBulk, AllGatherLocal, AllGatherMCCL:
	default:
		bad = true
	}
	if bad {
		log.Panicf("AllGather %q needs M a multiple of %d x GPUs, N and K multiples "+
			"of %d, the engines and a signal board for rows/bulk, and no split-K, "+
			"small tiles, bands or warm-up", b.AllGather, edge, edge)
	}
}

// execAllGather runs the all-gather-before-GEMM layer and copies every GPU's
// C back. The window runs from the start of the gather (or the GEMM launch,
// for AllGatherLocal) to the last GEMM finishing and, with the engines, the
// last engine finishing.
func (b *Benchmark) execAllGather() {
	p := len(b.gpus)
	edge := 4 * b.tileSize()
	b.validateAllGather(edge)
	tileRows := int(b.M / edge)
	per := tileRows / p
	shard := per * int(edge*b.K) // elements of A each GPU holds

	as := make([]driver.Ptr, p)
	bs := make([]driver.Ptr, p)
	cs := make([]driver.Ptr, p)
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		as[g] = b.driver.AllocateMemory(b.context, uint64(b.M*b.K*4))
		bs[g] = b.driver.AllocateMemory(b.context, uint64(b.K*b.N*4))
		cs[g] = b.driver.AllocateMemory(b.context, uint64(b.M*b.N*4))
		if b.AllGather == AllGatherLocal {
			b.driver.MemCopyH2D(b.context, as[g], b.matrixA.Data)
		} else {
			b.driver.MemCopyH2D(b.context, as[g]+driver.Ptr(4*g*shard),
				b.matrixA.Data[g*shard:(g+1)*shard])
		}
		bLocal := b.matrixB.columnSlice(uint32(g)*b.N, uint32(g+1)*b.N)
		b.driver.MemCopyH2D(b.context, bs[g], bLocal.Data)
	}

	ts := b.tileSize()
	launch := func(g int, q *driver.CommandQueue, co *insts.HsaCo, row, rows uint32) {
		b.driver.SelectGPU(b.context, b.gpus[g])
		e := driver.LocalPtr(edge)
		b.driver.EnqueueLaunchKernel(q, co,
			[3]uint32{b.N / 4, rows / 4, 1},
			[3]uint16{uint16(ts), uint16(ts), 1},
			&matrixmultiplication.KernelArgs{
				MatrixA: as[g] + driver.Ptr(uint64(row)*uint64(b.K)*4),
				MatrixB: bs[g],
				MatrixC: cs[g] + driver.Ptr(uint64(row)*uint64(b.N)*4),
				WidthA:  b.K,
				BlockA:  e * e * 4,
			})
	}
	held := func(wait []string, name string) *insts.HsaCo {
		co := *b.kernel
		co.WaitSignals, co.Signal = wait, name
		return &co
	}
	newQueue := func(g int) *driver.CommandQueue {
		b.driver.SelectGPU(b.context, b.gpus[g])
		return b.driver.CreateCommandQueue(b.context)
	}

	var comms []*mccl.Communicator
	if b.AllGather == AllGatherMCCL {
		comms = mccl.CommInitAll(p, b.driver, b.context, b.gpus)
	}

	// Queues are created in launch order: the driver serves them in
	// creation order, and a queue runs one kernel at a time.
	var queues []*driver.CommandQueue
	var launches []func()
	for g := 0; g < p; g++ {
		g := g
		switch b.AllGather {
		case AllGatherLocal, AllGatherMCCL:
			q := newQueue(g)
			queues = append(queues, q)
			launches = append(launches, func() { launch(g, q, b.kernel, 0, b.M) })
		case AllGatherBulk:
			var wait []string
			for _, r := range b.remoteRows(g, per) {
				wait = append(wait, commengine.RowSignal(g, r))
			}
			q := newQueue(g)
			queues = append(queues, q)
			co := held(wait, fmt.Sprintf("gpu%d/gemm", g))
			launches = append(launches, func() { launch(g, q, co, 0, b.M) })
		case AllGatherRows:
			for r := g * per; r < (g+1)*per; r++ {
				q, row := newQueue(g), uint32(r)*edge
				queues = append(queues, q)
				launches = append(launches, func() { launch(g, q, b.kernel, row, edge) })
			}
			for _, r := range b.remoteRows(g, per) {
				q, row := newQueue(g), uint32(r)*edge
				queues = append(queues, q)
				co := held([]string{commengine.RowSignal(g, r)},
					fmt.Sprintf("gpu%d/gemm/row%d", g, r))
				launches = append(launches, func() { launch(g, q, co, row, edge) })
			}
		}
	}

	var gather, gemmWait sim.VTimeInSec
	b.driver.WaitIdle()
	windowStart := b.driver.Engine.CurrentTime()
	switch b.AllGather {
	case AllGatherMCCL:
		mccl.AllGatherDirect(b.driver, comms, as, shard)
		gather = b.driver.Engine.CurrentTime() - windowStart
	case AllGatherRows, AllGatherBulk:
		b.Comm.ArmPush(b.context, as, b.M, b.K, commengine.PushOptions{
			BandRows: edge, Signals: b.AllGatherSignals})
	}
	gemmStart := b.driver.Engine.CurrentTime()
	for _, l := range launches {
		l()
	}
	for _, q := range queues {
		b.driver.DrainCommandQueue(q)
	}
	gemmWait = b.driver.Engine.CurrentTime() - gemmStart
	end := windowStart + gather + gemmWait
	if b.Comm != nil && (b.AllGather == AllGatherRows || b.AllGather == AllGatherBulk) {
		pushDone := b.Comm.Wait()
		end = max(end, pushDone)
		b.logAllGather(windowStart, pushDone, per)
	}

	log.Printf("allreducegemm allgather phases: mode=%s window=%.9f gather=%.9f "+
		"gemm_wait=%.9f", b.AllGather, end-windowStart, gather, gemmWait)

	b.driver.WaitIdle()
	b.gatherC = make([][]float32, p)
	for g, gpuID := range b.gpus {
		b.gatherC[g] = make([]float32, b.M*b.N)
		b.driver.SelectGPU(b.context, gpuID)
		b.driver.MemCopyD2H(b.context, b.gatherC[g], cs[g])
	}
}

// remoteRows lists the rows of tiles GPU g receives, in expected arrival
// order: every peer sends its rows in order, all peers in parallel.
func (b *Benchmark) remoteRows(g, per int) []int {
	var rows []int
	for k := 0; k < per; k++ {
		for q := range b.gpus {
			if q != g {
				rows = append(rows, q*per+k)
			}
		}
	}
	return rows
}

// logAllGather prints, relative to the window start, every link's traffic,
// every row's arrival and every held launch's timeline.
func (b *Benchmark) logAllGather(start, pushDone sim.VTimeInSec, per int) {
	board := b.AllGatherSignals
	type link struct{ from, to int }
	type stat struct {
		msgs, bytes         int
		firstSent, lastRecv sim.VTimeInSec
	}
	links := map[link]*stat{}
	lastArrival := start
	for _, r := range b.Comm.Records() {
		l := link{r.From, r.To}
		s := links[l]
		if s == nil {
			s = &stat{firstSent: r.Sent}
			links[l] = s
		}
		s.msgs++
		s.bytes += r.TrafficBytes
		s.firstSent = min(s.firstSent, r.Sent)
		s.lastRecv = max(s.lastRecv, r.Received)
		raised, _ := board.RaisedAt(commengine.RowSignal(r.To, r.Group))
		lastArrival = max(lastArrival, raised)
		log.Printf("allreducegemm allgather row: gpu=%d row=%d from=%d queued=%.9f "+
			"sent=%.9f received=%.9f raised=%.9f", r.To, r.Group, r.From,
			r.Queued-start, r.Sent-start, r.Received-start, raised-start)
	}
	keys := make([]link, 0, len(links))
	for l := range links {
		keys = append(keys, l)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i].from < keys[j].from ||
			keys[i].from == keys[j].from && keys[i].to < keys[j].to
	})
	for _, l := range keys {
		s := links[l]
		log.Printf("allreducegemm allgather link: from=%d to=%d msgs=%d "+
			"traffic_bytes=%d first_sent=%.9f last_received=%.9f",
			l.from, l.to, s.msgs, s.bytes, s.firstSent-start, s.lastRecv-start)
	}

	recs := board.Records()
	for _, r := range recs {
		var g int
		fmt.Sscanf(r.Name, "gpu%d/", &g)
		waited := b.waitedFor(board, r.Name, g, per)
		log.Printf("allreducegemm allgather launch: name=%s held=%.9f signal=%.9f "+
			"released=%.9f done=%.9f", r.Name, r.Held-start, waited-start,
			r.Released-start, r.Done-start)
	}
	log.Printf("allreducegemm allgather push: done=%.9f last_arrival=%.9f held=%d",
		pushDone-start, lastArrival-start, len(recs))
}

// waitedFor returns when the last signal a held GEMM launch waited for was
// raised.
func (b *Benchmark) waitedFor(board *signal.Board, name string, g, per int) sim.VTimeInSec {
	var row int
	if _, err := fmt.Sscanf(name, fmt.Sprintf("gpu%d/gemm/row%%d", g), &row); err == nil {
		t, _ := board.RaisedAt(commengine.RowSignal(g, row))
		return t
	}
	var last sim.VTimeInSec
	for _, r := range b.remoteRows(g, per) {
		t, _ := board.RaisedAt(commengine.RowSignal(g, r))
		last = max(last, t)
	}
	return last
}

// verifyAllGather checks every GPU's C = A x B_local against the CPU.
func (b *Benchmark) verifyAllGather() {
	width := b.N * uint32(len(b.gpus))
	edge := 4 * b.tileSize()
	wrong := 0
	for g := range b.gpus {
		byRow := map[uint32]int{}
		for row := uint32(0); row < b.M; row++ {
			for col := uint32(0); col < b.N; col++ {
				var sum float32
				for k := uint32(0); k < b.K; k++ {
					sum += b.matrixA.Data[row*b.K+k] *
						b.matrixB.Data[k*width+uint32(g)*b.N+col]
				}
				got := b.gatherC[g][row*b.N+col]
				tolerance := float32(1e-2) * (float32(math.Abs(float64(sum))) + 1)
				if float32(math.Abs(float64(sum-got))) > tolerance {
					byRow[row/edge]++
				}
			}
		}
		for r, n := range byRow {
			log.Printf("verify: gpu %d, row of tiles %d: %d wrong", g, r, n)
			wrong += n
		}
	}
	if wrong > 0 {
		log.Panicf("verify: %d elements wrong", wrong)
	}
	log.Print("Passed!")
}
