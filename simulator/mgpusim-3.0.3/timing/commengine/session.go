package commengine

import (
	"sync"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// Participant is one GPU's tracker and engine.
type Participant struct {
	Tracker *tiletracker.Tracker
	Engine  *Engine
}

// Options configures one fused collective.
type Options struct {
	TilesPerMsg int
	// Groups, if set, replaces TilesPerMsg with explicit tile groups.
	Groups     [][]int
	Mode       Mode
	Collective Collective
	Placement  Placement
	// Partials, if set, gives every GPU's split-K partial buffers of C (by
	// GPU, then buffer); see Config.Partials and Config.SplitSend.
	Partials  [][]tiletracker.Partial
	SplitSend bool
}

// Session drives the engines of a set of GPUs from the host: arm them before
// the GEMM launch, then wait until every engine has finished.
type Session struct {
	driver *driver.Driver
	parts  []Participant

	mu         sync.Mutex
	numDone    int
	done       chan struct{}
	doneTime   sim.VTimeInSec
	firstReady sim.VTimeInSec
}

// NewSession creates a session over the participants, in GPU order.
func NewSession(d *driver.Driver, parts []Participant) *Session {
	return &Session{driver: d, parts: parts}
}

// NumGPU returns the number of participants.
func (s *Session) NumGPU() int {
	return len(s.parts)
}

// Arm programs every GPU's tracker and engine for a row-major C of rows x
// cols floats at cs[i] on GPU i, cut into tileEdge x tileEdge tiles.
func (s *Session) Arm(
	ctx *driver.Context, cs []driver.Ptr,
	rows, cols, tileEdge uint32, opt Options,
) {
	s.mu.Lock()
	s.numDone = 0
	s.firstReady = -1
	s.done = make(chan struct{})
	s.mu.Unlock()

	for i, p := range s.parts {
		peers, out := s.ports(i)

		var partials []tiletracker.Partial
		if opt.Partials != nil {
			partials = opt.Partials[i]
		}
		region := s.region(ctx, uint64(cs[i]), rows, cols, tileEdge, partials)
		if opt.Placement == NearMemory {
			gpu, numGPU := i, len(s.parts)
			groupOf := opt.groupIndex(int((rows / tileEdge) * (cols / tileEdge)))
			owned := func(tile int) bool { return Owner(groupOf[tile], numGPU) == gpu }
			region.Accumulate = owned
			region.Expected = func(tile int) int {
				if owned(tile) {
					return numGPU
				}
				return 1
			}
		}
		p.Tracker.Arm(region)
		p.Tracker.ClearListeners()
		p.Tracker.AddLocalListener(p.Engine.OnTileLocal)
		p.Tracker.AddLocalListener(s.tileReady)
		p.Tracker.AddListener(p.Engine.OnTileComplete)
		p.Tracker.IgnoreSrc(p.Engine.ToMem)

		p.Engine.Arm(Config{
			GPU: i, NumGPU: len(s.parts), Peers: peers, Out: out,
			VBase: uint64(cs[i]), Rows: rows, Cols: cols,
			TileRows: tileEdge, TileCols: tileEdge,
			VToP:        translator(region.PageMap, region.Log2PageSize),
			TilesPerMsg: opt.TilesPerMsg,
			Groups:      opt.Groups,
			Mode:        opt.Mode,
			Collective:  opt.Collective,
			Placement:   opt.Placement,
			Partials:    partials,
			SplitSend:   opt.SplitSend,
			OnDone:      s.engineDone,
		})
	}
}

// PushOptions configures an all-gather of a matrix's rows with the engines
// in push mode (see PushConfig).
type PushOptions struct {
	BandRows uint32
	// Order lists a GPU's sends given the bands it owns (nil: DirectOrder).
	Order   func(gpu, numGPU int, own []int) []PushSend
	Signals *signal.Board
	// Parts gives the K chunks band is sent to participant to in (nil: 1),
	// and Chunks[to][s] are to's s chunk buffers (rows x cols/s each).
	Parts  func(to, band int) int
	Chunks []map[int][]driver.Ptr
	// Sums[i] are GPU i's partial sums (PushSum) of the rows of C[i], a
	// row-major matrix of rows x CCols floats that GPU i's tracker watches
	// in BandRows x BandRows tiles, with each sum's sources as partials.
	Sums  [][]PushSum
	C     []driver.Ptr
	CCols uint32
}

// ArmPush programs every GPU's engine to all-gather a row-major matrix of
// rows x cols floats at as[i] on GPU i, where GPU i owns the i-th of numGPU
// equal blocks of bands, and starts the engines at the current simulated
// time. Call it while the simulation is idle (Driver.WaitIdle).
func (s *Session) ArmPush(
	ctx *driver.Context, as []driver.Ptr, rows, cols uint32, opt PushOptions,
) {
	n := len(s.parts)
	bands := int(rows / opt.BandRows)
	if opt.BandRows == 0 || rows%opt.BandRows != 0 || bands%n != 0 {
		panic("commengine: push bands must split evenly among the GPUs")
	}
	per := bands / n
	order := opt.Order
	if order == nil {
		order = DirectOrder
	}
	parts := opt.Parts
	if parts == nil {
		parts = func(int, int) int { return 1 }
	}

	s.mu.Lock()
	s.numDone = 0
	s.firstReady = -1
	s.done = make(chan struct{})
	s.mu.Unlock()

	now := s.driver.Engine.CurrentTime()
	log2 := s.driver.Log2PageSize
	for i, p := range s.parts {
		peers, out := s.ports(i)
		own := make([]int, per)
		for k := range own {
			own[k] = i*per + k
		}
		sends := order(i, n, own)
		for k := range sends {
			sends[k].Parts = parts(sends[k].To, sends[k].Band)
		}
		receives := 0
		for b := 0; b < bands; b++ {
			if b/per != i {
				receives += parts(i, b)
			}
		}

		pages := map[uint64]uint64{}
		mapPages(s.driver, ctx, pages, uint64(as[i]), uint64(rows)*uint64(cols)*4)
		chunks := map[int][]uint64{}
		if opt.Chunks != nil {
			for k, bufs := range opt.Chunks[i] {
				for _, b := range bufs {
					chunks[k] = append(chunks[k], uint64(b))
					mapPages(s.driver, ctx, pages, uint64(b), uint64(rows*cols/uint32(k))*4)
				}
			}
		}
		var sums []PushSum
		if opt.Sums != nil {
			sums = opt.Sums[i]
		}
		if len(sums) > 0 {
			var partials []tiletracker.Partial
			for _, sum := range sums {
				for _, src := range sum.Srcs {
					partials = append(partials, tiletracker.Partial{VBase: src,
						Row: uint32(sum.Row) * opt.BandRows, Rows: opt.BandRows})
				}
			}
			region := TrackRegion(s.driver, ctx, uint64(opt.C[i]), rows, opt.CCols,
				opt.BandRows, partials)
			region.IgnoreLate = true
			for pp, v := range region.PageMap {
				pages[pp] = v
			}
			p.Tracker.Arm(region)
			p.Tracker.ClearListeners()
			p.Tracker.AddLocalListener(p.Engine.OnRowTile)
			p.Tracker.IgnoreSrc(p.Engine.ToMem)
		}

		p.Engine.Arm(Config{
			GPU: i, NumGPU: n, Peers: peers, Out: out,
			VBase: uint64(as[i]), Rows: rows, Cols: cols,
			VToP:   translator(pages, log2),
			OnDone: s.engineDone,
			Push: &PushConfig{BandRows: opt.BandRows, Sends: sends,
				Receives: receives, Signals: opt.Signals, Chunks: chunks,
				Sums: sums, RowTiles: int(opt.CCols / opt.BandRows)},
		})
		p.Engine.TickLater(now)
	}
}

// SumRecords returns every partial sum of the last push.
func (s *Session) SumRecords() []SumRecord {
	var out []SumRecord
	for _, p := range s.parts {
		out = append(out, p.Engine.SumRecords()...)
	}
	return out
}

// ports gives participant i's destination port at each peer and the port
// it sends through: with dedicated links, its link to that peer and the
// peer's end of it; otherwise both are ToFabric.
func (s *Session) ports(i int) (peers, out []sim.Port) {
	p := s.parts[i]
	peers = make([]sim.Port, len(s.parts))
	out = make([]sim.Port, len(s.parts))
	for j, q := range s.parts {
		peers[j], out[j] = q.Engine.ToFabric, p.Engine.ToFabric
		if l, ok := p.Engine.Links[q.Engine]; ok {
			out[j], peers[j] = l, q.Engine.Links[p.Engine]
		}
	}
	return peers, out
}

// translator maps virtual to physical addresses with a physical-to-virtual
// page map.
func translator(pageMap map[uint64]uint64, log2PageSize uint64) func(uint64) uint64 {
	vToP := invert(pageMap)
	mask := uint64(1)<<log2PageSize - 1
	return func(v uint64) uint64 {
		pPage, ok := vToP[v&^mask]
		if !ok {
			panic("commengine: address outside the armed region")
		}
		return pPage | v&mask
	}
}

// groupIndex maps every tile to its group.
func (o Options) groupIndex(numTiles int) []int {
	groups := o.Groups
	if groups == nil {
		groups = ConsecutiveGroups(numTiles, o.TilesPerMsg)
	}
	index := make([]int, numTiles)
	for g, tiles := range groups {
		for _, tile := range tiles {
			index[tile] = g
		}
	}
	return index
}

// TileOwner returns the GPU index that owns a tile's group, for a C of
// numTiles tiles.
func (o Options) TileOwner(numTiles, numGPU int) func(tile int) int {
	index := o.groupIndex(numTiles)
	return func(tile int) int { return Owner(index[tile], numGPU) }
}

func (s *Session) region(
	ctx *driver.Context, c uint64, rows, cols, tileEdge uint32,
	partials []tiletracker.Partial,
) tiletracker.Region {
	return TrackRegion(s.driver, ctx, c, rows, cols, tileEdge, partials)
}

// TrackRegion returns the tracker region of a row-major C of rows x cols
// floats at c, cut into tileEdge x tileEdge tiles, with its split-K partial
// buffers and the physical-to-virtual map of every page they touch.
func TrackRegion(
	d *driver.Driver, ctx *driver.Context, c uint64, rows, cols, tileEdge uint32,
	partials []tiletracker.Partial,
) tiletracker.Region {
	pages := map[uint64]uint64{}
	mapPages(d, ctx, pages, c, uint64(rows)*uint64(cols)*4)
	for _, p := range partials {
		mapPages(d, ctx, pages, p.VBase, uint64(p.Rows)*uint64(cols)*4)
	}
	return tiletracker.Region{
		VBase: c, Rows: rows, Cols: cols,
		TileRows: tileEdge, TileCols: tileEdge,
		PageMap: pages, Log2PageSize: d.Log2PageSize,
		Partials: partials,
	}
}

// mapPages adds the physical-to-virtual map of every page of the size bytes
// at base to pages.
func mapPages(d *driver.Driver, ctx *driver.Context, pages map[uint64]uint64, base, size uint64) {
	pageSize := uint64(1) << d.Log2PageSize
	for v := base &^ (pageSize - 1); v < base+size; v += pageSize {
		p, ok := d.PhysicalAddress(ctx, v)
		if !ok {
			panic("commengine: buffer is not mapped")
		}
		pages[p] = v
	}
}

func invert(m map[uint64]uint64) map[uint64]uint64 {
	out := make(map[uint64]uint64, len(m))
	for k, v := range m {
		out[v] = k
	}
	return out
}

func (s *Session) tileReady(ev tiletracker.TileEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.firstReady < 0 {
		s.firstReady = ev.Complete
	}
}

// FirstReady returns when the first tile of the last collective was
// complete in L2 on any GPU.
func (s *Session) FirstReady() sim.VTimeInSec {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.firstReady
}

func (s *Session) engineDone(_ int, now sim.VTimeInSec) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.numDone++
	if s.numDone == len(s.parts) {
		s.doneTime = now
		close(s.done)
	}
}

// Wait blocks the host until every engine has finished, and returns the
// simulated time when the last one did.
func (s *Session) Wait() sim.VTimeInSec {
	s.driver.WaitFor(s.done)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, p := range s.parts {
		p.Tracker.Disarm()
	}
	return s.doneTime
}

// Records returns every message of the last collective, by sending GPU.
func (s *Session) Records() []*MsgRecord {
	var out []*MsgRecord
	for _, p := range s.parts {
		out = append(out, p.Engine.Records()...)
	}
	return out
}
