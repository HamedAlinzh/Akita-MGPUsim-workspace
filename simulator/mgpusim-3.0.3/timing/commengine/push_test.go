package commengine

import (
	"fmt"
	"testing"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
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
