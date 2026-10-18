// Package commengine models a DMA-like communication engine that reduces the
// output tiles of a K-split GEMM across GPUs as soon as they are written,
// without using any compute unit.
//
// One engine sits in each GPU. It reads and writes local memory through the
// GPU's L2 caches (it plugs into the L1-to-L2 connection like an L1 cache)
// and talks to the other GPUs' engines over the GPU-to-GPU network. The tile
// tracker (timing/tiletracker) tells it when each output tile of C is
// complete in L2.
//
// Tiles are grouped into messages of TilesPerMsg consecutive tiles (row-major
// tile order). Group g is owned by GPU g mod numGPU. For each group:
//
//  1. Every non-owner reads its partial copy from L2 once all of the group's
//     tiles are complete there, and sends it to the owner.
//  2. The owner reads its own copy once complete, adds the partials from all
//     other GPUs, and writes the sum into its C.
//  3. For an all-reduce, the owner sends the sum back to every other GPU,
//     which writes it into its C (all-gather). A reduce-scatter stops after
//     step 2, leaving each group's sum only at its owner.
//
// Each message costs a fixed per-message overhead before it is handed to the
// network (command setup), serialized across the engine's messages but
// overlapped with the transmission of earlier ones. Merging tiles into one
// message saves this overhead.
//
// In bulk mode the engine releases groups only after every tile of the
// local GEMM is complete: the same engine and network, without overlap.
package commengine

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/mem/vm"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// Mode selects when the engine starts sending.
type Mode int

const (
	// ModeTile sends each group as soon as its tiles are complete.
	ModeTile Mode = iota
	// ModeBulk sends only after the whole local GEMM is complete.
	ModeBulk
)

// Placement selects where the reduction happens.
type Placement int

const (
	// OwnerEngine: the owner's engine reads its own copy from L2, adds the
	// partials in the engine, and writes the sum back.
	OwnerEngine Placement = iota
	// NearMemory: T3-style. Partials are written into the owner's C as
	// update stores, which the memory side adds in place (op-and-store),
	// in any order with the owner's own GEMM stores; the tile tracker
	// counts both and reports the tile reduced. The owner's engine only
	// reads the reduced tile to send it back (all-reduce).
	NearMemory
)

// Collective selects what the engines compute.
type Collective int

const (
	// AllReduce leaves the full sum in every GPU's C.
	AllReduce Collective = iota
	// ReduceScatter leaves each group's sum only in its owner's C.
	ReduceScatter
)

// physicalPID tags the engine's memory requests. Behind the address
// translators every request carries PID 0, so the L2 caches hold the GEMM's
// lines under PID 0, and the engine, which uses physical addresses too, must
// match it.
const physicalPID = vm.PID(0)

// msgHeaderBytes is the network header of a tile message.
const msgHeaderBytes = 16

// ConsecutiveGroups splits tiles 0..numTiles-1 into groups of n consecutive
// tiles.
func ConsecutiveGroups(numTiles, n int) [][]int {
	if numTiles%n != 0 {
		panic(fmt.Sprintf("commengine: %d tiles do not split into groups of %d",
			numTiles, n))
	}
	groups := make([][]int, numTiles/n)
	for g := range groups {
		for i := 0; i < n; i++ {
			groups[g] = append(groups[g], g*n+i)
		}
	}
	return groups
}

// Owner returns the GPU index that owns a group.
func Owner(group, numGPU int) int {
	return group % numGPU
}

// Config arms an engine for one GEMM's output.
type Config struct {
	GPU    int        // this engine's index among the participants
	NumGPU int        // number of participants
	Peers  []sim.Port // the port each participant receives on from us
	// Out gives, by participant, the port to send through (default
	// ToFabric).
	Out []sim.Port

	// C is a row-major matrix of 4-byte elements at virtual address VBase,
	// cut into TileRows x TileCols tiles. VToP translates its addresses.
	VBase              uint64
	Rows, Cols         uint32
	TileRows, TileCols uint32
	VToP               func(vAddr uint64) uint64

	TilesPerMsg int
	// Groups, if set, lists the tiles of every message group instead of
	// TilesPerMsg consecutive tiles; every tile must be in exactly one.
	Groups     [][]int
	Mode       Mode
	Collective Collective
	Placement  Placement

	// OnDone is called, inside the simulation, when this engine has
	// finished all its work.
	OnDone func(gpu int, now sim.VTimeInSec)
}

// TileMsg carries one group's partial or reduced data between engines.
type TileMsg struct {
	sim.MsgMeta
	Group   int
	Reduced bool
	From    int
	Data    []byte

	record *MsgRecord
	onSent func(now sim.VTimeInSec)
}

// Meta returns the message's meta data.
func (m *TileMsg) Meta() *sim.MsgMeta {
	return &m.MsgMeta
}

// MsgRecord is the timeline of one message.
type MsgRecord struct {
	Group        int
	Reduced      bool
	From, To     int
	PayloadBytes int
	TrafficBytes int
	// Queued: the data is read and the message waits for the engine.
	// Sent: the engine handed it to the network after its overhead.
	// Received: the destination engine took it from the network.
	Queued, Sent, Received sim.VTimeInSec
}

type segment struct {
	pAddr  uint64
	offset int // into the group buffer
	size   int
}

type memJob struct {
	write       bool
	update      bool
	segs        []segment
	data        []byte
	next        int
	outstanding int
	onDone      func(now sim.VTimeInSec, data []byte)
}

type reduceJob struct {
	group      int
	cyclesLeft int
}

// Engine is the per-GPU communication engine.
type Engine struct {
	*sim.TickingComponent

	ToMem    sim.Port
	ToFabric sim.Port

	// Links, if set, gives the port of a dedicated link to each other
	// engine; messages to that engine leave through it instead of ToFabric.
	Links map[*Engine]sim.Port

	memFinder           mem.LowModuleFinder
	msgOverhead         sim.VTimeInSec
	reduceCyclesPerLine int
	maxInflight         int

	armed     bool
	cfg       Config
	tileBytes int
	tilesX    uint32
	numTiles  int
	numGroups int
	groups    [][]int
	tileGroup []int

	localDone []int
	localAll  int
	released  []bool
	own       map[int][]byte
	partials  map[int]map[int][]byte
	reduced   map[int]bool
	nearDone  []int  // near memory: tiles reported reduced, per group
	gathered  []bool // near memory: reduced group read for the all-gather

	memJobs  []*memJob
	inflight map[string]*memJob
	offsets  map[string]int
	reduceQ  []*reduceJob

	sendQ       []*TileMsg
	sendStarted bool
	sendReadyAt sim.VTimeInSec

	ownedTotal, ownedDone       int
	nonOwnedTotal, nonOwnedDone int
	ownedPending                map[int]int
	finished                    bool

	records []*MsgRecord
}

// SetMemFinder sets how local physical addresses map to L2 banks.
func (e *Engine) SetMemFinder(f mem.LowModuleFinder) {
	e.memFinder = f
}

// Records returns the timeline of every message this engine sent.
func (e *Engine) Records() []*MsgRecord {
	return e.records
}

// Finished tells whether the engine has completed all its work.
func (e *Engine) Finished() bool {
	return e.finished
}

// Arm prepares the engine for one GEMM. It must be called before the GEMM's
// first tile completes.
func (e *Engine) Arm(cfg Config) {
	if cfg.TilesPerMsg < 1 && cfg.Groups == nil {
		panic("commengine: TilesPerMsg must be at least 1")
	}
	if cfg.Rows%cfg.TileRows != 0 || cfg.Cols%cfg.TileCols != 0 {
		panic("commengine: tiles must divide C")
	}

	e.cfg = cfg
	e.armed = true
	e.tileBytes = int(cfg.TileRows * cfg.TileCols * 4)
	e.tilesX = cfg.Cols / cfg.TileCols
	e.numTiles = int(cfg.Rows/cfg.TileRows) * int(e.tilesX)
	e.groups = cfg.Groups
	if e.groups == nil {
		e.groups = ConsecutiveGroups(e.numTiles, cfg.TilesPerMsg)
	}
	e.numGroups = len(e.groups)
	e.tileGroup = make([]int, e.numTiles)
	for i := range e.tileGroup {
		e.tileGroup[i] = -1
	}
	for g, tiles := range e.groups {
		for _, tile := range tiles {
			if e.tileGroup[tile] >= 0 {
				panic(fmt.Sprintf("commengine: tile %d in two groups", tile))
			}
			e.tileGroup[tile] = g
		}
	}
	for tile, g := range e.tileGroup {
		if g < 0 {
			panic(fmt.Sprintf("commengine: tile %d in no group", tile))
		}
	}

	e.localDone = make([]int, e.numGroups)
	e.localAll = 0
	e.released = make([]bool, e.numGroups)
	e.own = map[int][]byte{}
	e.partials = map[int]map[int][]byte{}
	e.reduced = map[int]bool{}
	e.nearDone = make([]int, e.numGroups)
	e.gathered = make([]bool, e.numGroups)
	e.memJobs = nil
	e.inflight = map[string]*memJob{}
	e.offsets = map[string]int{}
	e.reduceQ = nil
	e.sendQ = nil
	e.sendStarted = false
	e.ownedPending = map[int]int{}
	e.records = nil
	e.finished = false

	e.ownedTotal, e.nonOwnedTotal = 0, 0
	e.ownedDone, e.nonOwnedDone = 0, 0
	for g := 0; g < e.numGroups; g++ {
		if Owner(g, cfg.NumGPU) == cfg.GPU {
			e.ownedTotal++
		} else {
			e.nonOwnedTotal++
		}
	}
}

// OnTileLocal is the tile tracker's listener for a tile whose local GEMM
// output is complete in L2.
func (e *Engine) OnTileLocal(ev tiletracker.TileEvent) {
	if !e.armed {
		return
	}

	g := e.tileGroup[ev.Tile]
	e.localDone[g]++
	e.localAll++

	switch e.cfg.Mode {
	case ModeTile:
		if e.localDone[g] == len(e.groups[g]) {
			e.release(g)
		}
	case ModeBulk:
		if e.localAll == e.numTiles {
			for g := 0; g < e.numGroups; g++ {
				e.release(g)
				e.tryGather(ev.Complete, g)
			}
		}
	}

	e.TickLater(ev.Complete)
}

// OnTileComplete is the tile tracker's listener for a tile that has all its
// expected writes. With near-memory reduction that is an owned tile with
// every peer's update added; otherwise it is the same as OnTileLocal and is
// ignored.
func (e *Engine) OnTileComplete(ev tiletracker.TileEvent) {
	if !e.armed || e.cfg.Placement != NearMemory {
		return
	}

	g := e.tileGroup[ev.Tile]
	if Owner(g, e.cfg.NumGPU) != e.cfg.GPU {
		return
	}
	e.nearDone[g]++
	e.tryGather(ev.Complete, g)
	e.TickLater(ev.Complete)
}

// tryGather finishes a near-memory owned group once all its tiles are
// reduced (and, in bulk mode, the local GEMM is done): a reduce-scatter is
// then complete; an all-reduce reads the sum and sends it to every peer.
func (e *Engine) tryGather(now sim.VTimeInSec, g int) {
	if e.cfg.Placement != NearMemory || Owner(g, e.cfg.NumGPU) != e.cfg.GPU ||
		e.gathered[g] || e.nearDone[g] < len(e.groups[g]) ||
		e.cfg.Mode == ModeBulk && e.localAll < e.numTiles {
		return
	}
	e.gathered[g] = true

	if e.cfg.Collective == ReduceScatter {
		e.ownedDone++
		e.checkFinished(now)
		return
	}

	e.ownedPending[g] = e.cfg.NumGPU - 1
	e.enqueueMem(false, g, nil, func(now sim.VTimeInSec, sum []byte) {
		for p := 0; p < e.cfg.NumGPU; p++ {
			if p == e.cfg.GPU {
				continue
			}
			e.enqueueSend(now, g, true, p, append([]byte(nil), sum...),
				func(now sim.VTimeInSec) { e.ownedItemDone(now, g) })
		}
	})
}

func (e *Engine) release(g int) {
	if e.released[g] {
		return
	}
	e.released[g] = true

	if Owner(g, e.cfg.NumGPU) == e.cfg.GPU {
		if e.cfg.Placement == NearMemory {
			return // the memory side reduces; see tryGather
		}
		e.enqueueMem(false, g, nil, func(now sim.VTimeInSec, data []byte) {
			e.own[g] = data
			e.tryReduce(g)
		})
		return
	}

	e.enqueueMem(false, g, nil, func(now sim.VTimeInSec, data []byte) {
		owner := Owner(g, e.cfg.NumGPU)
		e.enqueueSend(now, g, false, owner, data, func(now sim.VTimeInSec) {
			if e.cfg.Collective == ReduceScatter {
				e.nonOwnedDone++
				e.checkFinished(now)
			}
		})
	})
}

func (e *Engine) tryReduce(g int) {
	if e.reduced[g] || e.own[g] == nil ||
		len(e.partials[g]) < e.cfg.NumGPU-1 {
		return
	}
	e.reduced[g] = true

	lines := (len(e.own[g]) + 63) / 64
	e.reduceQ = append(e.reduceQ, &reduceJob{
		group:      g,
		cyclesLeft: lines * e.reduceCyclesPerLine * (e.cfg.NumGPU - 1),
	})
}

func (e *Engine) finishReduce(now sim.VTimeInSec, g int) {
	sum := e.own[g]
	for p := 0; p < e.cfg.NumGPU; p++ {
		if p == e.cfg.GPU {
			continue
		}
		addFloat32s(sum, e.partials[g][p])
	}
	delete(e.partials, g)

	pending := 1
	if e.cfg.Collective == AllReduce {
		pending += e.cfg.NumGPU - 1
	}
	e.ownedPending[g] = pending

	e.enqueueMem(true, g, sum, func(now sim.VTimeInSec, _ []byte) {
		e.ownedItemDone(now, g)
	})

	if e.cfg.Collective != AllReduce {
		return
	}
	for p := 0; p < e.cfg.NumGPU; p++ {
		if p == e.cfg.GPU {
			continue
		}
		e.enqueueSend(now, g, true, p, append([]byte(nil), sum...),
			func(now sim.VTimeInSec) { e.ownedItemDone(now, g) })
	}
}

func (e *Engine) ownedItemDone(now sim.VTimeInSec, g int) {
	e.ownedPending[g]--
	if e.ownedPending[g] == 0 {
		e.ownedDone++
		e.checkFinished(now)
	}
}

func (e *Engine) checkFinished(now sim.VTimeInSec) {
	if e.finished || e.ownedDone < e.ownedTotal ||
		e.nonOwnedDone < e.nonOwnedTotal {
		return
	}
	e.finished = true
	if e.cfg.OnDone != nil {
		e.cfg.OnDone(e.cfg.GPU, now)
	}
}

func addFloat32s(dst, src []byte) {
	for i := 0; i+4 <= len(dst); i += 4 {
		a := math.Float32frombits(binary.LittleEndian.Uint32(dst[i:]))
		b := math.Float32frombits(binary.LittleEndian.Uint32(src[i:]))
		binary.LittleEndian.PutUint32(dst[i:], math.Float32bits(a+b))
	}
}

// groupSegments lists a group's bytes as line-sized physical segments, in
// group-buffer order: tile by tile, row by row.
func (e *Engine) groupSegments(g int) []segment {
	cfg := &e.cfg
	rowBytes := int(cfg.TileCols * 4)
	var segs []segment
	for i, tile := range e.groups[g] {
		tr := uint32(tile) / e.tilesX
		tc := uint32(tile) % e.tilesX
		for r := uint32(0); r < cfg.TileRows; r++ {
			v := cfg.VBase + uint64(((tr*cfg.TileRows+r)*cfg.Cols+tc*cfg.TileCols)*4)
			for off := 0; off < rowBytes; {
				addr := v + uint64(off)
				lineEnd := addr/64*64 + 64
				n := min(rowBytes-off, int(lineEnd-addr))
				segs = append(segs, segment{
					pAddr:  cfg.VToP(addr),
					offset: i*e.tileBytes + int(r)*rowBytes + off,
					size:   n,
				})
				off += n
			}
		}
	}
	return segs
}

func (e *Engine) enqueueMem(
	write bool, g int, data []byte,
	onDone func(now sim.VTimeInSec, data []byte),
) {
	job := &memJob{write: write, segs: e.groupSegments(g), data: data, onDone: onDone}
	if !write {
		job.data = make([]byte, e.tileBytes*len(e.groups[g]))
	}
	e.memJobs = append(e.memJobs, job)
}

// enqueueUpdate writes a peer's partial into this GPU's C as update stores,
// which the memory side adds to what the tile holds.
func (e *Engine) enqueueUpdate(g int, data []byte) {
	job := &memJob{write: true, update: true, segs: e.groupSegments(g), data: data,
		onDone: func(sim.VTimeInSec, []byte) {}}
	e.memJobs = append(e.memJobs, job)
}

func (e *Engine) enqueueSend(
	now sim.VTimeInSec, g int, reduced bool, to int, data []byte,
	onSent func(now sim.VTimeInSec),
) {
	rec := &MsgRecord{
		Group: g, Reduced: reduced, From: e.cfg.GPU, To: to,
		PayloadBytes: len(data), TrafficBytes: len(data) + msgHeaderBytes,
		Queued: now,
	}
	e.records = append(e.records, rec)
	msg := &TileMsg{Group: g, Reduced: reduced, From: e.cfg.GPU, Data: data,
		record: rec, onSent: onSent}
	msg.ID = sim.GetIDGenerator().Generate()
	msg.Src = e.outPort(to)
	msg.Dst = e.cfg.Peers[to]
	msg.TrafficBytes = rec.TrafficBytes
	e.sendQ = append(e.sendQ, msg)
}

// Tick advances the engine by one cycle.
func (e *Engine) Tick(now sim.VTimeInSec) bool {
	if !e.armed {
		return false
	}

	madeProgress := false
	madeProgress = e.recvFabric(now) || madeProgress
	madeProgress = e.recvMem(now) || madeProgress
	madeProgress = e.doReduce(now) || madeProgress
	madeProgress = e.issueMem(now) || madeProgress
	madeProgress = e.send(now) || madeProgress
	return madeProgress
}

func (e *Engine) outPort(to int) sim.Port {
	if to < len(e.cfg.Out) && e.cfg.Out[to] != nil {
		return e.cfg.Out[to]
	}
	return e.ToFabric
}

func (e *Engine) recvFabric(now sim.VTimeInSec) bool {
	madeProgress := e.recvFrom(now, e.ToFabric)
	for _, p := range e.Links {
		madeProgress = e.recvFrom(now, p) || madeProgress
	}
	return madeProgress
}

func (e *Engine) recvFrom(now sim.VTimeInSec, port sim.Port) bool {
	item := port.Retrieve(now)
	if item == nil {
		return false
	}

	msg := item.(*TileMsg)
	msg.record.Received = now

	if !msg.Reduced && e.cfg.Placement == NearMemory {
		e.enqueueUpdate(msg.Group, msg.Data)
		return true
	}

	if !msg.Reduced {
		if e.partials[msg.Group] == nil {
			e.partials[msg.Group] = map[int][]byte{}
		}
		e.partials[msg.Group][msg.From] = msg.Data
		e.tryReduce(msg.Group)
		return true
	}

	e.enqueueMem(true, msg.Group, msg.Data, func(now sim.VTimeInSec, _ []byte) {
		e.nonOwnedDone++
		e.checkFinished(now)
	})
	return true
}

func (e *Engine) recvMem(now sim.VTimeInSec) bool {
	item := e.ToMem.Retrieve(now)
	if item == nil {
		return false
	}

	var reqID string
	switch rsp := item.(type) {
	case *mem.DataReadyRsp:
		reqID = rsp.RespondTo
		job := e.inflight[reqID]
		copy(job.data[e.offsets[reqID]:], rsp.Data)
	case *mem.WriteDoneRsp:
		reqID = rsp.RespondTo
	default:
		panic(fmt.Sprintf("commengine: unexpected %T from memory", item))
	}

	job := e.inflight[reqID]
	delete(e.inflight, reqID)
	delete(e.offsets, reqID)
	job.outstanding--
	if job.outstanding == 0 && job.next == len(job.segs) {
		job.onDone(now, job.data)
	}
	return true
}

func (e *Engine) issueMem(now sim.VTimeInSec) bool {
	if len(e.memJobs) == 0 || len(e.inflight) >= e.maxInflight {
		return false
	}

	job := e.memJobs[0]
	seg := job.segs[job.next]
	dst := e.memFinder.Find(seg.pAddr)

	var req sim.Msg
	if job.write {
		w := mem.WriteReqBuilder{}.
			WithSendTime(now).
			WithSrc(e.ToMem).
			WithDst(dst).
			WithAddress(seg.pAddr).
			WithPID(physicalPID).
			WithData(job.data[seg.offset : seg.offset+seg.size]).
			Build()
		if job.update {
			w.Info = tiletracker.Update{}
		}
		req = w
	} else {
		req = mem.ReadReqBuilder{}.
			WithSendTime(now).
			WithSrc(e.ToMem).
			WithDst(dst).
			WithAddress(seg.pAddr).
			WithPID(physicalPID).
			WithByteSize(uint64(seg.size)).
			Build()
	}

	if err := e.ToMem.Send(req); err != nil {
		return false
	}

	e.inflight[req.Meta().ID] = job
	e.offsets[req.Meta().ID] = seg.offset
	job.outstanding++
	job.next++
	if job.next == len(job.segs) {
		e.memJobs = e.memJobs[1:]
	}
	return true
}

func (e *Engine) doReduce(now sim.VTimeInSec) bool {
	if len(e.reduceQ) == 0 {
		return false
	}

	job := e.reduceQ[0]
	job.cyclesLeft--
	if job.cyclesLeft <= 0 {
		e.reduceQ = e.reduceQ[1:]
		e.finishReduce(now, job.group)
	}
	return true
}

// send hands the head message to the network once the engine has spent the
// per-message overhead on it. The overhead of the next message starts when
// the previous one is handed over, so it overlaps that one's transmission.
func (e *Engine) send(now sim.VTimeInSec) bool {
	if len(e.sendQ) == 0 {
		return false
	}

	if !e.sendStarted {
		e.sendStarted = true
		e.sendReadyAt = now + e.msgOverhead
	}
	if now < e.sendReadyAt {
		return true // busy with the message's setup
	}

	msg := e.sendQ[0]
	msg.SendTime = now
	if err := msg.Src.Send(msg); err != nil {
		return false
	}

	e.sendQ = e.sendQ[1:]
	e.sendStarted = false
	msg.record.Sent = now
	msg.onSent(now)
	return true
}
