package commengine

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/sarchlab/akita/v3/mem/idealmemcontroller"
	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// The tests connect each engine to an ideal memory holding a 64x64 C of
// four 32x32 tiles at address 0, and all engines to one direct connection
// as the fabric. Tile completions are injected as events.

const (
	testRows, testCols, testEdge = 64, 64, 32
	testTiles                    = 4
)

type callback func()

func (c callback) Handle(sim.Event) error { c(); return nil }

type rig struct {
	engine  sim.Engine
	engines []*Engine
	mems    []*idealmemcontroller.Comp
	done    []sim.VTimeInSec
}

func newRig(t *testing.T, numGPU int, overhead sim.VTimeInSec) *rig {
	t.Helper()
	r := &rig{engine: sim.NewSerialEngine(), done: make([]sim.VTimeInSec, numGPU)}
	fabric := sim.NewDirectConnection("Fabric", r.engine, 1*sim.GHz)
	for i := 0; i < numGPU; i++ {
		e := MakeBuilder().WithEngine(r.engine).WithMsgOverhead(overhead).
			Build(fmt.Sprintf("Engine%d", i))
		m := idealmemcontroller.MakeBuilder().WithEngine(r.engine).
			WithNewStorage(1 * mem.MB).Build(fmt.Sprintf("Mem%d", i))
		conn := sim.NewDirectConnection(fmt.Sprintf("Conn%d", i), r.engine, 1*sim.GHz)
		conn.PlugIn(e.ToMem, 64)
		conn.PlugIn(m.GetPortByName("Top"), 64)
		e.SetMemFinder(&mem.SingleLowModuleFinder{LowModule: m.GetPortByName("Top")})
		fabric.PlugIn(e.ToFabric, 16)

		// GPU i's partial for element (row, col) of tile t is (i+1)*(t+1).
		for row := 0; row < testRows; row++ {
			for col := 0; col < testCols; col++ {
				tile := row/testEdge*(testCols/testEdge) + col/testEdge
				writeFloat(t, m, row, col, float32((i+1)*(tile+1)))
			}
		}
		r.engines = append(r.engines, e)
		r.mems = append(r.mems, m)
	}
	return r
}

func addr(row, col int) uint64 { return uint64((row*testCols + col) * 4) }

func writeFloat(t *testing.T, m *idealmemcontroller.Comp, row, col int, v float32) {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	if err := m.Storage.Write(addr(row, col), b); err != nil {
		t.Fatal(err)
	}
}

func readFloat(t *testing.T, m *idealmemcontroller.Comp, row, col int) float32 {
	b, err := m.Storage.Read(addr(row, col), 4)
	if err != nil {
		t.Fatal(err)
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(b))
}

// run arms every engine and completes tile t on every GPU at
// completeAt(t), then runs the simulation to the end.
func (r *rig) run(
	t *testing.T, tilesPerMsg int, mode Mode, coll Collective,
	completeAt func(tile int) sim.VTimeInSec,
) {
	t.Helper()
	peers := make([]sim.Port, len(r.engines))
	for i, e := range r.engines {
		peers[i] = e.ToFabric
	}
	for i, e := range r.engines {
		e.Arm(Config{
			GPU: i, NumGPU: len(r.engines), Peers: peers,
			Rows: testRows, Cols: testCols, TileRows: testEdge, TileCols: testEdge,
			VToP:        func(v uint64) uint64 { return v },
			TilesPerMsg: tilesPerMsg, Mode: mode, Collective: coll,
			OnDone: func(gpu int, now sim.VTimeInSec) {
				if r.done[gpu] != 0 {
					t.Errorf("engine %d finished twice", gpu)
				}
				r.done[gpu] = now
			},
		})
		for tile := 0; tile < testTiles; tile++ {
			e, tile := e, tile
			at := completeAt(tile)
			r.engine.Schedule(sim.NewEventBase(at, callback(func() {
				e.OnTileLocal(tiletracker.TileEvent{Tile: tile, Complete: at})
			})))
		}
	}
	if err := r.engine.Run(); err != nil {
		t.Fatal(err)
	}
	for i, d := range r.done {
		if d == 0 {
			t.Errorf("engine %d never finished", i)
		}
	}
}

func staggered(tile int) sim.VTimeInSec { return sim.VTimeInSec(tile+1) * 10e-6 }

// expectC checks every GPU's C: the full sum (tile+1) * sum(i+1) where
// summed, the GPU's own partial elsewhere.
func (r *rig) expectC(t *testing.T, summed func(gpu, tile int) bool) {
	t.Helper()
	n := len(r.engines)
	total := n * (n + 1) / 2
	for i, m := range r.mems {
		for row := 0; row < testRows; row += 7 {
			for col := 0; col < testCols; col += 5 {
				tile := row/testEdge*(testCols/testEdge) + col/testEdge
				want := float32((i + 1) * (tile + 1))
				if summed(i, tile) {
					want = float32(total * (tile + 1))
				}
				if got := readFloat(t, m, row, col); got != want {
					t.Fatalf("GPU %d (%d, %d), tile %d: got %v, want %v",
						i, row, col, tile, got, want)
				}
			}
		}
	}
}

// messagesPerDirection counts messages by (from, to).
func (r *rig) messagesPerDirection() map[[2]int]int {
	out := map[[2]int]int{}
	for _, e := range r.engines {
		for _, rec := range e.Records() {
			out[[2]int{rec.From, rec.To}]++
			if rec.Received == 0 {
				panic("message never received")
			}
		}
	}
	return out
}

func TestAllReduceTwoGPUs(t *testing.T) {
	for _, n := range []int{1, 2, 4} {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			r := newRig(t, 2, 0)
			r.run(t, n, ModeTile, AllReduce, staggered)
			r.expectC(t, func(int, int) bool { return true })
			for d, c := range r.messagesPerDirection() {
				if want := testTiles / n; c != want {
					t.Errorf("%v: %d messages, want tiles/N = %d", d, c, want)
				}
			}
		})
	}
}

func TestAllReduceFourGPUs(t *testing.T) {
	r := newRig(t, 4, 0)
	r.run(t, 1, ModeTile, AllReduce, staggered)
	r.expectC(t, func(int, int) bool { return true })
}

func TestReduceScatterKeepsSumOnlyAtOwner(t *testing.T) {
	r := newRig(t, 2, 0)
	r.run(t, 1, ModeTile, ReduceScatter, staggered)
	r.expectC(t, func(gpu, tile int) bool { return Owner(tile, 2) == gpu })
	for d, c := range r.messagesPerDirection() {
		if want := testTiles / 2; c != want {
			t.Errorf("%v: %d messages, want %d", d, c, want)
		}
	}
}

func TestTileModeSendsBeforeTheGEMMEnds(t *testing.T) {
	r := newRig(t, 2, 0)
	r.run(t, 1, ModeTile, AllReduce, staggered)
	last := staggered(testTiles - 1)
	if first := r.engines[0].Records()[0].Sent; first >= last {
		t.Errorf("first message sent at %g, not before the last tile at %g",
			first, last)
	}
}

func TestBulkModeWaitsForTheWholeGEMM(t *testing.T) {
	r := newRig(t, 2, 0)
	r.run(t, 1, ModeBulk, AllReduce, staggered)
	r.expectC(t, func(int, int) bool { return true })
	last := staggered(testTiles - 1)
	for _, e := range r.engines {
		for _, rec := range e.Records() {
			if rec.Queued < last {
				t.Errorf("message queued at %g, before the last tile at %g",
					rec.Queued, last)
			}
		}
	}
}

func TestMsgOverheadIsSerializedPerEngine(t *testing.T) {
	const overhead = 5e-6
	r := newRig(t, 2, overhead)
	// All tiles finish at once, so messages queue up behind each other.
	r.run(t, 1, ModeTile, AllReduce, func(int) sim.VTimeInSec { return 1e-6 })
	r.expectC(t, func(int, int) bool { return true })
	for _, e := range r.engines {
		recs := e.Records()
		for i, rec := range recs {
			if rec.Sent-rec.Queued < overhead*0.999 {
				t.Errorf("message %d sent %g after queued, less than the overhead",
					i, rec.Sent-rec.Queued)
			}
			if i > 0 && rec.Sent-recs[i-1].Sent < overhead*0.999 {
				t.Errorf("messages %d and %d sent %g apart, less than the overhead",
					i-1, i, rec.Sent-recs[i-1].Sent)
			}
		}
	}
}
