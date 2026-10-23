package allreducegemm

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strings"

	"github.com/sarchlab/akita/v3/sim"
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
	// order, each held at the CP until its row's arrival signal. With
	// Benchmark.AllGatherPlan a remote row split s ways arrives as s K
	// chunks into chunk buffers, and runs as s launches (one per chunk, the
	// first writing C, the others partial buffers), each held until its
	// chunk; the engine adds the partials into C once the tracker reports
	// the row's tiles complete in all copies.
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
	if b.AllGatherPlan != nil {
		tileRows := int(b.M / edge)
		per := tileRows / int(p)
		bad = bad || b.AllGather != AllGatherRows || len(b.AllGatherPlan) != tileRows
		for i, s := range b.AllGatherPlan {
			bad = bad || s < 1 || i < per && s != 1 || (b.K/uint32(s))%edge != 0
		}
	}
	bad = bad || b.AllGatherNoSum && b.AllGatherPlan == nil
	if bad {
		log.Panicf("AllGather %q needs M a multiple of %d x GPUs, N and K multiples "+
			"of %d, the engines and a signal board for rows/bulk, and no split-K, "+
			"small tiles, bands or warm-up; a plan needs rows mode, a factor per "+
			"row of tiles with the local rows 1, and K/s a multiple of %d",
			b.AllGather, edge, edge, edge)
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

	launch := func(g int, q *driver.CommandQueue, co *insts.HsaCo, row, rows uint32) {
		b.driver.SelectGPU(b.context, b.gpus[g])
		b.launchRows(q, co, as[g], bs[g], cs[g]+driver.Ptr(uint64(row)*uint64(b.N)*4),
			b.K, row, rows)
	}
	b.agWait = map[string][]string{}
	held := func(wait []string, name string) *insts.HsaCo {
		co := *b.kernel
		co.WaitSignals, co.Signal = wait, name
		b.agWait[name] = wait
		return &co
	}
	split := b.allGatherSplits(per)
	chunks, sums := b.allocSplitRows(split, cs, edge)
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
				row, s := uint32(r)*edge, split[g][r]
				if s == 1 {
					q := newQueue(g)
					queues = append(queues, q)
					co := held([]string{commengine.RowSignal(g, r)},
						fmt.Sprintf("gpu%d/gemm/row%d", g, r))
					launches = append(launches, func() { launch(g, q, co, row, edge) })
					continue
				}
				kc := b.K / uint32(s)
				for c := 0; c < s; c++ {
					q := newQueue(g)
					queues = append(queues, q)
					co := held([]string{commengine.ChunkSignal(g, r, c)},
						fmt.Sprintf("gpu%d/gemm/row%d/chunk%d", g, r, c))
					a := chunks[g][s][c]
					bm := bs[g] + driver.Ptr(uint64(c)*uint64(kc)*uint64(b.N)*4)
					dst := cs[g] + driver.Ptr(uint64(row)*uint64(b.N)*4)
					if c > 0 {
						dst = driver.Ptr(sums[g][r].Srcs[c-1])
					}
					launches = append(launches, func() {
						b.driver.SelectGPU(b.context, b.gpus[g])
						b.launchRows(q, co, a, bm, dst, kc, row, edge)
					})
				}
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
		opt := commengine.PushOptions{BandRows: edge, Signals: b.AllGatherSignals}
		if b.AllGatherPlan != nil {
			opt.Parts = func(to, band int) int { return split[to][band] }
			opt.Chunks, opt.C, opt.CCols = chunks, cs, b.N
			if !b.AllGatherNoSum {
				opt.Sums = make([][]commengine.PushSum, p)
				for g := range sums {
					for _, sum := range sums[g] {
						opt.Sums[g] = append(opt.Sums[g], sum)
					}
					sort.Slice(opt.Sums[g], func(i, j int) bool {
						return opt.Sums[g][i].Row < opt.Sums[g][j].Row
					})
				}
			}
		}
		b.Comm.ArmPush(b.context, as, b.M, b.K, opt)
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

// allGatherSplits gives, by GPU, the split factor of every row of tiles:
// AllGatherPlan by position in that GPU's arrival order (local rows first),
// or 1 without a plan.
func (b *Benchmark) allGatherSplits(per int) []map[int]int {
	split := make([]map[int]int, len(b.gpus))
	for g := range split {
		split[g] = map[int]int{}
		order := append([]int(nil), b.localRows(g, per)...)
		order = append(order, b.remoteRows(g, per)...)
		for i, r := range order {
			split[g][r] = 1
			if b.AllGatherPlan != nil {
				split[g][r] = b.AllGatherPlan[i]
			}
		}
	}
	return split
}

func (b *Benchmark) localRows(g, per int) []int {
	rows := make([]int, per)
	for k := range rows {
		rows[k] = g*per + k
	}
	return rows
}

// allocSplitRows allocates, on every GPU, the K chunk buffers of A for each
// split factor its rows use (s buffers of M x K/s), and for every split row
// the partial buffers of its chunks 1.. (one row of tiles of C each); sums
// gives each split row's sum into C, by GPU and row.
func (b *Benchmark) allocSplitRows(
	split []map[int]int, cs []driver.Ptr, edge uint32,
) (chunks []map[int][]driver.Ptr, sums []map[int]commengine.PushSum) {
	chunks = make([]map[int][]driver.Ptr, len(b.gpus))
	sums = make([]map[int]commengine.PushSum, len(b.gpus))
	rowBytes := uint64(edge * b.N * 4)
	for g, gpuID := range b.gpus {
		b.driver.SelectGPU(b.context, gpuID)
		chunks[g], sums[g] = map[int][]driver.Ptr{}, map[int]commengine.PushSum{}
		rows := make([]int, 0, len(split[g]))
		for r := range split[g] {
			rows = append(rows, r)
		}
		sort.Ints(rows)
		for _, r := range rows {
			s := split[g][r]
			if s == 1 {
				continue
			}
			if chunks[g][s] == nil {
				for c := 0; c < s; c++ {
					chunks[g][s] = append(chunks[g][s], b.driver.AllocateMemory(b.context,
						uint64(b.M*b.K/uint32(s)*4)))
				}
			}
			sum := commengine.PushSum{Row: r,
				Dst: uint64(cs[g]) + uint64(r)*rowBytes, Bytes: int(rowBytes)}
			for c := 1; c < s; c++ {
				sum.Srcs = append(sum.Srcs, uint64(b.driver.AllocateMemory(b.context, rowBytes)))
			}
			sums[g][r] = sum
		}
	}
	return chunks, sums
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
		name := commengine.RowSignal(r.To, r.Group)
		if r.Parts > 1 {
			name = commengine.ChunkSignal(r.To, r.Group, r.Part)
		}
		raised, _ := board.RaisedAt(name)
		lastArrival = max(lastArrival, raised)
		log.Printf("allreducegemm allgather row: gpu=%d row=%d chunk=%d of=%d from=%d "+
			"queued=%.9f sent=%.9f received=%.9f raised=%.9f", r.To, r.Group, r.Part,
			max(1, r.Parts), r.From, r.Queued-start, r.Sent-start, r.Received-start,
			raised-start)
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
		var waited sim.VTimeInSec
		for _, w := range b.agWait[r.Name] {
			t, _ := board.RaisedAt(w)
			waited = max(waited, t)
		}
		log.Printf("allreducegemm allgather launch: name=%s held=%.9f signal=%.9f "+
			"released=%.9f done=%.9f", r.Name, r.Held-start, waited-start,
			r.Released-start, r.Done-start)
	}
	b.logRemoteRows(start, recs, per)
	log.Printf("allreducegemm allgather push: done=%.9f last_arrival=%.9f held=%d",
		pushDone-start, lastArrival-start, len(recs))
}

// logRemoteRows prints, for every remote row, when its first launch was
// released, when its last chunk (or the row) was in L2, and its partial
// sum's timeline (0 without one).
func (b *Benchmark) logRemoteRows(start sim.VTimeInSec, recs []signal.Record, per int) {
	board := b.AllGatherSignals
	sums := map[[2]int]commengine.SumRecord{}
	for _, s := range b.Comm.SumRecords() {
		sums[[2]int{s.GPU, s.Row}] = s
	}
	split := b.allGatherSplits(per)
	for g := range b.gpus {
		for _, r := range b.remoteRows(g, per) {
			s := split[g][r]
			prefix := fmt.Sprintf("gpu%d/gemm/row%d", g, r)
			first := sim.VTimeInSec(-1)
			for _, rec := range recs {
				if rec.Name == prefix || strings.HasPrefix(rec.Name, prefix+"/") {
					if first < 0 || rec.Released < first {
						first = rec.Released
					}
				}
			}
			var inL2 sim.VTimeInSec
			for c := 0; c < s; c++ {
				name := commengine.RowSignal(g, r)
				if s > 1 {
					name = commengine.ChunkSignal(g, r, c)
				}
				t, _ := board.RaisedAt(name)
				inL2 = max(inL2, t)
			}
			sum, ok := sums[[2]int{g, r}]
			if !ok {
				sum.Ready, sum.Done = start, start
			}
			log.Printf("allreducegemm allgather remote: gpu=%d row=%d s=%d "+
				"first_release=%.9f last_in_l2=%.9f sum_ready=%.9f sum_done=%.9f",
				g, r, s, first-start, inL2-start, sum.Ready-start, sum.Done-start)
		}
	}
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
