package commengine

import (
	"encoding/binary"
	"fmt"
	"math"
	"testing"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// runPush all-gathers the rig's 64x64 matrix in bands of bandRows rows, GPU
// i owning band i, with the direct order.
func (r *rig) runPush(t *testing.T, bandRows uint32, board *signal.Board) {
	t.Helper()
	n := len(r.engines)
	peers := make([]sim.Port, n)
	for i, e := range r.engines {
		peers[i] = e.ToFabric
	}
	for i, e := range r.engines {
		e.Arm(Config{
			GPU: i, NumGPU: n, Peers: peers, Rows: testRows, Cols: testCols,
			VToP: func(v uint64) uint64 { return v },
			OnDone: func(gpu int, now sim.VTimeInSec) {
				if r.done[gpu] != 0 {
					t.Errorf("engine %d finished twice", gpu)
				}
				r.done[gpu] = now
			},
			Push: &PushConfig{BandRows: bandRows, Sends: DirectOrder(i, n, []int{i}),
				Receives: n - 1, Signals: board},
		})
		e.TickLater(0)
	}
	if err := r.engine.Run(); err != nil {
		t.Fatal(err)
	}
}

// Every GPU ends with every band as its owner wrote it, and each band's
// signal is raised on the receiver after the message arrived, before the
// engine finished.
func TestPushAllGathersRows(t *testing.T) {
	for _, n := range []int{2, 4} {
		t.Run(fmt.Sprintf("%dGPUs", n), func(t *testing.T) {
			r := newRig(t, n, 1e-6)
			board := signal.NewBoard(r.engine)
			bandRows := testRows / n
			r.runPush(t, uint32(bandRows), board)

			for i, m := range r.mems {
				for row := 0; row < testRows; row += 3 {
					for col := 0; col < testCols; col += 5 {
						tile := row/testEdge*(testCols/testEdge) + col/testEdge
						owner := row / bandRows
						want := float32((owner + 1) * (tile + 1))
						if got := readFloat(t, m, row, col); got != want {
							t.Fatalf("GPU %d (%d, %d): got %v, want %v", i, row, col, got, want)
						}
					}
				}
			}

			for d, c := range r.messagesPerDirection() {
				if c != 1 {
					t.Errorf("%v: %d messages, want 1", d, c)
				}
			}
			for _, e := range r.engines {
				for _, rec := range e.Records() {
					raised, ok := board.RaisedAt(RowSignal(rec.To, rec.Group))
					if !ok || raised <= rec.Received || raised > r.done[rec.To] {
						t.Errorf("band %d at GPU %d: received %g, raised %v at %g, done %g",
							rec.Group, rec.To, rec.Received, ok, raised, r.done[rec.To])
					}
				}
			}
			for i := range r.engines {
				if _, ok := board.RaisedAt(RowSignal(i, i)); ok {
					t.Errorf("GPU %d raised a signal for its own band", i)
				}
			}
		})
	}
}

// The engine sends in list order, reading each band once.
func TestPushFollowsTheSendOrder(t *testing.T) {
	r := newRig(t, 2, 1e-6)
	peers := []sim.Port{r.engines[0].ToFabric, r.engines[1].ToFabric}
	sends := []PushSend{{Band: 1, To: 1}, {Band: 0, To: 1}}
	for i, e := range r.engines {
		cfg := Config{GPU: i, NumGPU: 2, Peers: peers, Rows: testRows, Cols: testCols,
			VToP:   func(v uint64) uint64 { return v },
			OnDone: func(gpu int, now sim.VTimeInSec) { r.done[gpu] = now },
			Push:   &PushConfig{BandRows: 16, Receives: 2}}
		if i == 0 {
			cfg.Push.Sends, cfg.Push.Receives = sends, 0
		}
		e.Arm(cfg)
		e.TickLater(0)
	}
	if err := r.engine.Run(); err != nil {
		t.Fatal(err)
	}
	recs := r.engines[0].Records()
	if len(recs) != 2 || recs[0].Group != 1 || recs[1].Group != 0 ||
		recs[0].Sent >= recs[1].Sent {
		t.Fatalf("records %+v %+v", *recs[0], *recs[1])
	}
	if r.done[0] == 0 || r.done[1] == 0 {
		t.Fatalf("done %v", r.done)
	}
}

// A band sent in 2 K chunks lands in the receiver's chunk buffers, one
// message and one chunk signal each, in K order.
func TestPushSendsKChunks(t *testing.T) {
	r := newRig(t, 2, 1e-6)
	board := signal.NewBoard(r.engine)
	peers := []sim.Port{r.engines[0].ToFabric, r.engines[1].ToFabric}
	chunks := map[int][]uint64{2: {0x10000, 0x12000}} // 64 rows x 32 cols each
	for i, e := range r.engines {
		e.Arm(Config{GPU: i, NumGPU: 2, Peers: peers, Rows: testRows, Cols: testCols,
			VToP:   func(v uint64) uint64 { return v },
			OnDone: func(gpu int, now sim.VTimeInSec) { r.done[gpu] = now },
			Push: &PushConfig{BandRows: 32, Receives: 2, Signals: board, Chunks: chunks,
				Sends: []PushSend{{Band: i, To: 1 - i, Parts: 2}}}})
		e.TickLater(0)
	}
	if err := r.engine.Run(); err != nil {
		t.Fatal(err)
	}
	for i, m := range r.mems {
		owner := 1 - i
		for row := owner * 32; row < owner*32+32; row += 3 {
			for col := 0; col < testCols; col += 5 {
				tile := row/testEdge*(testCols/testEdge) + col/testEdge
				c := col / 32
				b, err := m.Storage.Read(chunks[2][c]+uint64((row*32+col%32)*4), 4)
				if err != nil {
					t.Fatal(err)
				}
				got := math.Float32frombits(binary.LittleEndian.Uint32(b))
				if want := float32((owner + 1) * (tile + 1)); got != want {
					t.Fatalf("GPU %d chunk %d (%d, %d): got %v, want %v", i, c, row, col, got, want)
				}
			}
		}
	}
	for i, e := range r.engines {
		recs := e.Records()
		if len(recs) != 2 || recs[0].Part != 0 || recs[1].Part != 1 || recs[0].Sent >= recs[1].Sent {
			t.Fatalf("GPU %d records: %d", i, len(recs))
		}
		for c := 0; c < 2; c++ {
			if _, ok := board.RaisedAt(ChunkSignal(1-i, i, c)); !ok {
				t.Errorf("chunk %d of band %d not signalled on GPU %d", c, i, 1-i)
			}
		}
	}
	if r.done[0] == 0 || r.done[1] == 0 {
		t.Fatalf("done %v", r.done)
	}
}

// A row's partial sum starts once all its tiles are complete, adds the
// partial buffers into the row and finishes the engine after it.
func TestPushSumsPartials(t *testing.T) {
	r := newRig(t, 2, 0)
	e := r.engines[0]
	const dst, src1, src2, n = 0x20000, 0x21000, 0x22000, 1024
	for k := 0; k < n/4; k++ {
		for j, base := range []uint64{dst, src1, src2} {
			b := make([]byte, 4)
			binary.LittleEndian.PutUint32(b, math.Float32bits(float32(j+1)))
			if err := r.mems[0].Storage.Write(base+uint64(4*k), b); err != nil {
				t.Fatal(err)
			}
		}
	}
	e.Arm(Config{GPU: 0, NumGPU: 2, Peers: []sim.Port{e.ToFabric, r.engines[1].ToFabric},
		Rows: testRows, Cols: testCols, VToP: func(v uint64) uint64 { return v },
		OnDone: func(gpu int, now sim.VTimeInSec) { r.done[gpu] = now },
		Push: &PushConfig{BandRows: 32, RowTiles: 2,
			Sums: []PushSum{{Row: 1, Dst: dst, Srcs: []uint64{src1, src2}, Bytes: n}}}})
	for i, at := range []sim.VTimeInSec{5e-6, 9e-6, 7e-6} {
		ev := tiletracker.TileEvent{Row: uint32(min(i, 1)), Complete: at}
		r.engine.Schedule(sim.NewEventBase(at, callback(func() { e.OnRowTile(ev) })))
	}
	if err := r.engine.Run(); err != nil {
		t.Fatal(err)
	}
	for k := 0; k < n/4; k += 7 {
		b, _ := r.mems[0].Storage.Read(dst+uint64(4*k), 4)
		if got := math.Float32frombits(binary.LittleEndian.Uint32(b)); got != 6 {
			t.Fatalf("element %d: got %v, want 6", k, got)
		}
	}
	sums := e.SumRecords()
	if len(sums) != 1 || sums[0].Ready != 9e-6 || sums[0].Done <= sums[0].Ready ||
		r.done[0] != sums[0].Done {
		t.Fatalf("sums %+v, done %g", sums, r.done[0])
	}
}
