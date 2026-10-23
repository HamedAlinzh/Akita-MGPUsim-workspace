package allreducegemm_test

import (
	"fmt"
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
)

// TestAllGather gathers A (128x128, K 128) before the GEMM in every mode and
// verifies each GPU's C = A x B_local; with the engines every remote row's
// launch is held and released only after its row has arrived.
func TestAllGather(t *testing.T) {
	for _, c := range []struct {
		mode allreducegemm.AllGatherMode
		gpus int
	}{
		{allreducegemm.AllGatherRows, 2},
		{allreducegemm.AllGatherRows, 4},
		{allreducegemm.AllGatherBulk, 2},
		{allreducegemm.AllGatherLocal, 2},
		{allreducegemm.AllGatherMCCL, 2},
	} {
		t.Run(fmt.Sprintf("%s/%dgpus", c.mode, c.gpus), func(t *testing.T) {
			engines := c.mode == allreducegemm.AllGatherRows ||
				c.mode == allreducegemm.AllGatherBulk
			platform := runner.MakeR9NanoBuilder().
				WithNumGPU(c.gpus).
				WithNumCUPerGPU(4).
				WithNumDispatchers(16).
				WithOrderedDispatch(true).
				WithPeerMesh(c.gpus > 2).
				WithCommEngine(engines, 0).
				Build()
			d := platform.Driver
			d.Run()
			defer d.Terminate()

			b := allreducegemm.NewBenchmark(d)
			b.M, b.K, b.N = 128, 128, 128
			b.AllGather = c.mode
			var ids []int
			var parts []commengine.Participant
			board := signal.NewBoard(platform.Engine)
			for i, gpu := range platform.GPUs {
				ids = append(ids, i+1)
				parts = append(parts, commengine.Participant{
					Tracker: gpu.TileTracker, Engine: gpu.CommEngine})
				gpu.CommandProcessor.Signals = board
				board.AddWaker(gpu.CommandProcessor)
			}
			if engines {
				b.Comm = commengine.NewSession(d, parts)
				b.AllGatherSignals = board
			}
			b.SelectGPU(ids)
			b.EnableVerification()
			b.Run()
			b.Verify()

			recs := board.Records()
			want := 0
			switch c.mode {
			case allreducegemm.AllGatherRows:
				want = 4 * (c.gpus - 1) // each GPU's remote rows of the 4
			case allreducegemm.AllGatherBulk:
				want = c.gpus
			}
			if len(recs) != want {
				t.Fatalf("%d held launches, want %d", len(recs), want)
			}
			for _, r := range recs {
				var g, row int
				if _, err := fmt.Sscanf(r.Name, "gpu%d/gemm/row%d", &g, &row); err != nil {
					continue
				}
				arrived, ok := board.RaisedAt(commengine.RowSignal(g, row))
				if !ok || r.Released < arrived || r.Released < r.Held || r.Done <= r.Released {
					t.Errorf("%s: arrived %g, held %g, released %g, done %g",
						r.Name, arrived, r.Held, r.Released, r.Done)
				}
			}
		})
	}
}
