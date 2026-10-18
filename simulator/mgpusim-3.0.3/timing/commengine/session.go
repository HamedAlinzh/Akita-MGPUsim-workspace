package commengine

import (
	"sync"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/driver"
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
	Mode        Mode
	Collective  Collective
	Placement   Placement
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

	peers := make([]sim.Port, len(s.parts))
	for i, p := range s.parts {
		peers[i] = p.Engine.ToFabric
	}

	for i, p := range s.parts {
		region := s.region(ctx, uint64(cs[i]), rows, cols, tileEdge)
		if opt.Placement == NearMemory {
			gpu, numGPU, n := i, len(s.parts), opt.TilesPerMsg
			owned := func(tile int) bool { return Owner(tile/n, numGPU) == gpu }
			region.Accumulate = owned
			region.Expected = func(tile int) int {
				if owned(tile) {
					return numGPU
				}
				return 1
			}
		}
		vToP := invert(region.PageMap)
		mask := uint64(1)<<region.Log2PageSize - 1

		p.Tracker.Arm(region)
		p.Tracker.ClearListeners()
		p.Tracker.AddLocalListener(p.Engine.OnTileLocal)
		p.Tracker.AddLocalListener(s.tileReady)
		p.Tracker.AddListener(p.Engine.OnTileComplete)
		p.Tracker.IgnoreSrc(p.Engine.ToMem)

		p.Engine.Arm(Config{
			GPU: i, NumGPU: len(s.parts), Peers: peers,
			VBase: uint64(cs[i]), Rows: rows, Cols: cols,
			TileRows: tileEdge, TileCols: tileEdge,
			VToP: func(v uint64) uint64 {
				pPage, ok := vToP[v&^mask]
				if !ok {
					panic("commengine: address outside the armed C")
				}
				return pPage | v&mask
			},
			TilesPerMsg: opt.TilesPerMsg,
			Mode:        opt.Mode,
			Collective:  opt.Collective,
			Placement:   opt.Placement,
			OnDone:      s.engineDone,
		})
	}
}

func (s *Session) region(
	ctx *driver.Context, c uint64, rows, cols, tileEdge uint32,
) tiletracker.Region {
	log2 := s.driver.Log2PageSize
	pageSize := uint64(1) << log2
	pages := map[uint64]uint64{}
	for v := c &^ (pageSize - 1); v < c+uint64(rows)*uint64(cols)*4; v += pageSize {
		p, ok := s.driver.PhysicalAddress(ctx, v)
		if !ok {
			panic("commengine: C is not mapped")
		}
		pages[p] = v
	}
	return tiletracker.Region{
		VBase: c, Rows: rows, Cols: cols,
		TileRows: tileEdge, TileCols: tileEdge,
		PageMap: pages, Log2PageSize: log2,
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
