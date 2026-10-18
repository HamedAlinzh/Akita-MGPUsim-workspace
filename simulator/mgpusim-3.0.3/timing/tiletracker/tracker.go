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
// exactly once, as mmmKernel_local does. Writing a byte twice, or storing
// into a tile after it completed, is recorded as an error.
package tiletracker

import (
	"fmt"
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
}

const elemBytes = 4

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
}

type tileState struct {
	written    []bool
	data       []byte
	count      int
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

	t.armed = true
	t.region = r
	t.tileSize = int(r.TileRows * r.TileCols * elemBytes)
	t.tilesX = r.Cols / r.TileCols
	numTiles := int((r.Rows / r.TileRows) * t.tilesX)
	t.tiles = make([]tileState, numTiles)
	for i := range t.tiles {
		t.tiles[i].written = make([]bool, t.tileSize)
		t.tiles[i].data = make([]byte, t.tileSize)
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

// locate maps a virtual byte address to its tile and byte within the tile.
func (t *Tracker) locate(vAddr uint64) (tile, tileByte int, ok bool) {
	r := &t.region
	if vAddr < r.VBase {
		return 0, 0, false
	}
	off := vAddr - r.VBase
	elem := off / elemBytes
	if elem >= uint64(r.Rows)*uint64(r.Cols) {
		return 0, 0, false
	}
	row := uint32(elem / uint64(r.Cols))
	col := uint32(elem % uint64(r.Cols))
	tile = int((row/r.TileRows)*t.tilesX + col/r.TileCols)
	tileByte = int(((row%r.TileRows)*r.TileCols+col%r.TileCols)*elemBytes) +
		int(off%elemBytes)
	return tile, tileByte, true
}

func (t *Tracker) storeReceived(w *mem.WriteReq) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.armed {
		return
	}

	var segs []segment
	for i := range w.Data {
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
		segs = append(segs, segment{tile: tile, tileByte: tileByte, value: w.Data[i]})
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
		if ts.written[s.tileByte] {
			t.errors = append(t.errors, fmt.Sprintf(
				"%.12f: byte %d of tile %d written twice",
				now, s.tileByte, s.tile))
			continue
		}
		ts.written[s.tileByte] = true
		ts.data[s.tileByte] = s.value
		ts.count++
	}

	for _, tile := range touched {
		ts := &t.tiles[tile]
		ts.pending--
		if ts.count < t.tileSize || ts.complete {
			continue
		}
		if ts.pending != 0 {
			t.errors = append(t.errors, fmt.Sprintf(
				"%.12f: tile %d full with %d stores unacknowledged",
				now, tile, ts.pending))
		}
		ts.complete = true
		ev := TileEvent{
			Tile:       tile,
			Row:        uint32(tile) / t.tilesX,
			Col:        uint32(tile) % t.tilesX,
			FirstStore: ts.firstStore,
			LastStore:  ts.lastStore,
			Complete:   now,
			NumStores:  ts.numStores,
		}
		t.events = append(t.events, ev)
		if t.OnComplete != nil {
			t.OnComplete(ev, append([]byte(nil), ts.data...))
		}
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
