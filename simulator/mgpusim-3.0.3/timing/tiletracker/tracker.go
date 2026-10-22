// Package tiletracker watches the stores a kernel makes into an output
// matrix at the L2 caches and reports each output tile complete once every
// byte of it has been written and acknowledged by L2.
//
// It is a set of Akita hooks on the L2 top ports, so it changes neither the
// kernel nor the caches. A WriteReq arriving at an L2 top port is recorded by
// request ID; the bytes it covers count toward their tiles only when L2
// sends the matching WriteDoneRsp, which happens after L2 has written the
// data into its storage. A read of the tile that reaches L2 after the
// completion therefore sees the final values.
//
// The tracker relies on the producer writing every byte of the region
// exactly once, as mmmKernel_local does, or once per split-K partial (see
// Region.Partials). Writing a byte twice, or storing
// into a tile after it completed, is recorded as an error.
package tiletracker

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
)

// Region describes a row-major matrix of 4-byte elements, cut into tiles.
type Region struct {
	// VBase is the virtual address of element (0, 0).
	VBase uint64
	// Rows and Cols are the matrix dimensions in elements; Cols is also the
	// row pitch.
	Rows, Cols uint32
	// TileRows and TileCols are the tile dimensions in elements. They must
	// divide Rows and Cols.
	TileRows, TileCols uint32

	// PageMap maps the physical base address of every page the region
	// touches to that page's virtual base address. L2 sees physical
	// addresses.
	PageMap      map[uint64]uint64
	Log2PageSize uint64

	// Expected, if set, gives how many times every byte of a tile is
	// written before the tile is complete (default 1). With near-memory
	// reduction the owner's tiles expect one GEMM store plus one update per
	// other GPU.
	Expected func(tile int) int
	// Accumulate, if set, marks tiles whose stores add to what the tile
	// holds instead of overwriting it (op-and-store), in any order.
	Accumulate func(tile int) bool

	// Partials lists split-K partial buffers. A tile covered by k of them
	// is written k+1 times by the local GEMM (once into the region, once
	// into each partial) and is complete only after all k+1 are
	// acknowledged. Not combined with Expected or Accumulate.
	Partials []Partial
}

// Partial is a row-major buffer of Rows x Region.Cols elements at VBase
// whose element (r, c) is a split-K partial of region element (Row+r, c).
// Row and Rows are multiples of Region.TileRows.
type Partial struct {
	VBase     uint64
	Row, Rows uint32
}

// Covers tells whether the partial holds part of tile row tileRow of tiles
// tileRows high.
func (p Partial) Covers(tileRow, tileRows uint32) bool {
	r := tileRow * tileRows
	return r >= p.Row && r < p.Row+p.Rows
}

const elemBytes = 4

// Update tags a WriteReq (in its Info) as an update from a peer GPU: it
// counts toward its tile even when its source port is ignored, and it is
// not part of the local GEMM's output.
type Update struct{}

func isUpdate(w *mem.WriteReq) bool {
	_, ok := w.Info.(Update)
	return ok
}

// TileEvent reports that a tile completed.
type TileEvent struct {
	Tile     int
	Row, Col uint32 // tile coordinates, in tiles
	// FirstStore and LastStore are when L2 received the first and last
	// store into the tile, and Complete is when L2 sent the acknowledgement
	// that covered its last byte.
	FirstStore, LastStore, Complete sim.VTimeInSec
	NumStores                       int
}

type segment struct {
	tile     int
	tileByte int
	value    byte
	update   bool
}

type tileState struct {
	written    []uint8
	data       []byte
	sum        []float32 // running sums of an accumulating tile
	expected   int
	local      int // writes per byte from the local GEMM
	accumulate bool
	count      int
	localCount int
	localDone  bool
	numStores  int
	firstStore sim.VTimeInSec
	lastStore  sim.VTimeInSec
	started    bool
	complete   bool
	pending    int // stores received but not yet acknowledged
}

// Tracker is the per-GPU tile tracker.
type Tracker struct {
	mu sync.Mutex

	armed    bool
	region   Region
	tileSize int // bytes
	tilesX   uint32
	tiles    []tileState
	pending  map[string][]segment

	events []TileEvent
	errors []string

	// OnComplete, if set, is called when a tile completes, with the tile's
	// bytes as L2 holds them.
	OnComplete func(ev TileEvent, data []byte)

	listeners      []func(ev TileEvent)
	localListeners []func(ev TileEvent)
	ignored        map[sim.Port]bool
}

// AddLocalListener registers a function called when all of a tile's bytes
// from the local GEMM (stores that are not updates) are acknowledged. For a
// tile that expects one write per byte this is when it completes.
func (t *Tracker) AddLocalListener(f func(ev TileEvent)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.localListeners = append(t.localListeners, f)
}

// AddListener registers a function called, inside the simulation, every
// time a tile completes.
func (t *Tracker) AddListener(f func(ev TileEvent)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.listeners = append(t.listeners, f)
}

// ClearListeners removes every listener.
func (t *Tracker) ClearListeners() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.listeners = nil
	t.localListeners = nil
}

// IgnoreSrc makes the tracker ignore stores sent from the given port, such
// as a communication engine writing reduced tiles back into the region.
func (t *Tracker) IgnoreSrc(p sim.Port) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.ignored == nil {
		t.ignored = map[sim.Port]bool{}
	}
	t.ignored[p] = true
}

// New creates an unarmed tracker.
func New() *Tracker {
	return &Tracker{}
}

// Attach installs the tracker's hook on the given ports, which should be
// the top ports of every L2 bank of one GPU.
func (t *Tracker) Attach(ports ...sim.Port) {
	for _, p := range ports {
		p.AcceptHook(t)
	}
}

// Arm starts tracking a region, discarding any previous state. Stores that
// L2 received before Arm are ignored, even if acknowledged after it.
func (t *Tracker) Arm(r Region) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if r.TileRows == 0 || r.TileCols == 0 ||
		r.Rows%r.TileRows != 0 || r.Cols%r.TileCols != 0 {
		panic(fmt.Sprintf("tiletracker: tiles %dx%d do not divide %dx%d",
			r.TileRows, r.TileCols, r.Rows, r.Cols))
	}
	for _, p := range r.Partials {
		if r.Expected != nil || r.Accumulate != nil ||
			p.Row%r.TileRows != 0 || p.Rows%r.TileRows != 0 ||
			p.Rows == 0 || p.Row+p.Rows > r.Rows {
			panic("tiletracker: partials must cover whole tile rows of the " +
				"region, without Expected or Accumulate")
		}
	}

	t.armed = true
	t.region = r
	t.tileSize = int(r.TileRows * r.TileCols * elemBytes)
	t.tilesX = r.Cols / r.TileCols
	numTiles := int((r.Rows / r.TileRows) * t.tilesX)
	t.tiles = make([]tileState, numTiles)
	for i := range t.tiles {
		ts := &t.tiles[i]
		ts.written = make([]uint8, t.tileSize)
		ts.data = make([]byte, t.tileSize)
		ts.local = 1
		for _, p := range r.Partials {
			if p.Covers(uint32(i)/t.tilesX, r.TileRows) {
				ts.local++
			}
		}
		ts.expected = ts.local
		if r.Expected != nil {
			ts.expected = r.Expected(i)
		}
		if r.Accumulate != nil && r.Accumulate(i) {
			ts.accumulate = true
			ts.sum = make([]float32, t.tileSize/elemBytes)
		}
	}
	t.pending = make(map[string][]segment)
	t.events = nil
	t.errors = nil
}

// Disarm stops tracking. Results stay readable.
func (t *Tracker) Disarm() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.armed = false
}

// NumTiles returns the number of tiles in the armed region.
func (t *Tracker) NumTiles() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.tiles)
}

// Events returns the completions in the order they happened.
func (t *Tracker) Events() []TileEvent {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]TileEvent(nil), t.events...)
}

// Errors returns every protocol violation seen.
func (t *Tracker) Errors() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.errors...)
}

// Incomplete returns the indices of tiles that have not completed, and how
// many stores are received but unacknowledged.
func (t *Tracker) Incomplete() (tiles []int, pendingStores int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for i := range t.tiles {
		if !t.tiles[i].complete {
			tiles = append(tiles, i)
		}
	}
	return tiles, len(t.pending)
}

// Func implements sim.Hook.
func (t *Tracker) Func(ctx sim.HookCtx) {
	switch ctx.Pos {
	case sim.HookPosPortMsgRecvd:
		if w, ok := ctx.Item.(*mem.WriteReq); ok {
			t.storeReceived(w)
		}
	case sim.HookPosPortMsgSend:
		if rsp, ok := ctx.Item.(*mem.WriteDoneRsp); ok {
			t.storeAcked(rsp)
		}
	}
}

func (t *Tracker) virtualAddr(pAddr uint64) (uint64, bool) {
	pageMask := uint64(1)<<t.region.Log2PageSize - 1
	vPage, ok := t.region.PageMap[pAddr&^pageMask]
	if !ok {
		return 0, false
	}
	return vPage | pAddr&pageMask, true
}

// locate maps a virtual byte address, in the region or one of its partials,
// to its tile and byte within the tile.
func (t *Tracker) locate(vAddr uint64) (tile, tileByte int, ok bool) {
	r := &t.region
	if tile, tileByte, ok = t.locateIn(vAddr, r.VBase, 0, r.Rows); ok {
		return tile, tileByte, true
	}
	for _, p := range r.Partials {
		if tile, tileByte, ok = t.locateIn(vAddr, p.VBase, p.Row, p.Rows); ok {
			return tile, tileByte, true
		}
	}
	return 0, 0, false
}

// locateIn locates vAddr in a buffer of rows region rows, starting at region
// row row0, at base.
func (t *Tracker) locateIn(
	vAddr, base uint64, row0, rows uint32,
) (tile, tileByte int, ok bool) {
	r := &t.region
	if vAddr < base {
		return 0, 0, false
	}
	off := vAddr - base
	elem := off / elemBytes
	if elem >= uint64(rows)*uint64(r.Cols) {
		return 0, 0, false
	}
	row := row0 + uint32(elem/uint64(r.Cols))
	col := uint32(elem % uint64(r.Cols))
	tile = int((row/r.TileRows)*t.tilesX + col/r.TileCols)
	tileByte = int(((row%r.TileRows)*r.TileCols+col%r.TileCols)*elemBytes) +
		int(off%elemBytes)
	return tile, tileByte, true
}

func (t *Tracker) storeReceived(w *mem.WriteReq) {
	t.mu.Lock()
	defer t.mu.Unlock()

	update := isUpdate(w)
	if !t.armed || t.ignored[w.Src] && !update {
		return
	}

	var segs []segment
	for i := 0; i < len(w.Data); i++ {
		if w.DirtyMask != nil && !w.DirtyMask[i] {
			continue
		}
		vAddr, ok := t.virtualAddr(w.Address + uint64(i))
		if !ok {
			continue
		}
		tile, tileByte, ok := t.locate(vAddr)
		if !ok {
			continue
		}
		if t.tiles[tile].accumulate {
			t.accumulate(w, i, tile, tileByte)
		}
		segs = append(segs, segment{tile: tile, tileByte: tileByte,
			value: w.Data[i], update: update})
	}
	if len(segs) == 0 {
		return
	}

	now := w.RecvTime
	var tiles []int
	for _, s := range segs {
		tiles = appendUnique(tiles, s.tile)
	}
	for _, tile := range tiles {
		ts := &t.tiles[tile]
		if ts.complete {
			t.errors = append(t.errors, fmt.Sprintf(
				"%.12f: store %s into tile %d after it completed",
				now, w.ID, tile))
		}
		if !ts.started {
			ts.started = true
			ts.firstStore = now
		}
		ts.lastStore = now
		ts.pending++
		ts.numStores++
	}
	t.pending[w.ID] = segs
}

// accumulate turns the store of a whole element, starting at byte i of the
// request, into an add: the tile's running sum gains the stored value, and the
// request carries the new sum to L2. L2 handles writes to a line in arrival
// order, so the last write leaves the full sum whatever the order of the
// GEMM's store and the peers' updates. This stands in for an op-and-store
// ALU in front of the L2 bank.
func (t *Tracker) accumulate(w *mem.WriteReq, i, tile, tileByte int) {
	if tileByte%elemBytes != 0 {
		return // a later byte of an element already handled
	}
	if i+elemBytes > len(w.Data) || w.DirtyMask != nil &&
		!(w.DirtyMask[i+1] && w.DirtyMask[i+2] && w.DirtyMask[i+3]) {
		panic("tiletracker: accumulating store does not cover whole elements")
	}
	ts := &t.tiles[tile]
	e := tileByte / elemBytes
	ts.sum[e] += math.Float32frombits(binary.LittleEndian.Uint32(w.Data[i:]))
	binary.LittleEndian.PutUint32(w.Data[i:], math.Float32bits(ts.sum[e]))
}

func (t *Tracker) storeAcked(rsp *mem.WriteDoneRsp) {
	t.mu.Lock()
	defer t.mu.Unlock()

	segs, ok := t.pending[rsp.RespondTo]
	if !ok {
		return
	}
	delete(t.pending, rsp.RespondTo)

	now := rsp.SendTime
	var touched []int // tiles of this store, in first-byte order
	for _, s := range segs {
		touched = appendUnique(touched, s.tile)
	}
	for _, s := range segs {
		ts := &t.tiles[s.tile]
		if int(ts.written[s.tileByte]) >= ts.expected {
			t.errors = append(t.errors, fmt.Sprintf(
				"%.12f: byte %d of tile %d written more than %d times",
				now, s.tileByte, s.tile, ts.expected))
			continue
		}
		ts.written[s.tileByte]++
		ts.data[s.tileByte] = s.value
		ts.count++
		if !s.update {
			ts.localCount++
		}
	}

	for _, tile := range touched {
		ts := &t.tiles[tile]
		ts.pending--
		if ts.localCount == t.tileSize*ts.local && !ts.localDone {
			ts.localDone = true
			ev := t.event(tile, now)
			for _, f := range t.localListeners {
				f(ev)
			}
		}
		if ts.count < t.tileSize*ts.expected || ts.complete {
			continue
		}
		if ts.pending != 0 {
			t.errors = append(t.errors, fmt.Sprintf(
				"%.12f: tile %d full with %d stores unacknowledged",
				now, tile, ts.pending))
		}
		ts.complete = true
		ev := t.event(tile, now)
		t.events = append(t.events, ev)
		if t.OnComplete != nil {
			t.OnComplete(ev, append([]byte(nil), ts.data...))
		}
		for _, f := range t.listeners {
			f(ev)
		}
	}
}

func (t *Tracker) event(tile int, now sim.VTimeInSec) TileEvent {
	ts := &t.tiles[tile]
	return TileEvent{
		Tile:       tile,
		Row:        uint32(tile) / t.tilesX,
		Col:        uint32(tile) % t.tilesX,
		FirstStore: ts.firstStore,
		LastStore:  ts.lastStore,
		Complete:   now,
		NumStores:  ts.numStores,
	}
}

func appendUnique(list []int, v int) []int {
	for _, x := range list {
		if x == v {
			return list
		}
	}
	return append(list, v)
}
