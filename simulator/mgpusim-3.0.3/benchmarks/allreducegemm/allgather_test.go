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
// (or K chunk's) launch is held and released only after its data has
// arrived, and every split row's partials are summed.
func TestAllGather(t *testing.T) {
	for _, c := range []struct {
		mode allreducegemm.AllGatherMode
		gpus int
		plan []int
	}{
		{allreducegemm.AllGatherRows, 2, nil},
		{allreducegemm.AllGatherRows, 4, nil},
		{allreducegemm.AllGatherRows, 2, []int{1, 1, 4, 2}},
		{allreducegemm.AllGatherRows, 4, []int{1, 2, 4, 1}},
		{allreducegemm.AllGatherBulk, 2, nil},
		{allreducegemm.AllGatherLocal, 2, nil},
		{allreducegemm.AllGatherMCCL, 2, nil},
	} {
		t.Run(fmt.Sprintf("%s/%dgpus/plan%v", c.mode, c.gpus, c.plan), func(t *testing.T) {
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
			b.AllGather, b.AllGatherPlan = c.mode, c.plan
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
				// one launch per remote row, or per chunk of a split one
				per := 4 / c.gpus
				for i := per; i < 4; i++ {
					s := 1
					if c.plan != nil {
						s = c.plan[i]
					}
					want += c.gpus * s
				}
			case allreducegemm.AllGatherBulk:
				want = c.gpus
			}
			if len(recs) != want {
				t.Fatalf("%d held launches, want %d", len(recs), want)
			}
			for _, r := range recs {
				var g, row, chunk int
				wait := ""
				if _, err := fmt.Sscanf(r.Name, "gpu%d/gemm/row%d/chunk%d", &g, &row,
					&chunk); err == nil {
					wait = commengine.ChunkSignal(g, row, chunk)
				} else if _, err := fmt.Sscanf(r.Name, "gpu%d/gemm/row%d", &g, &row); err == nil {
					wait = commengine.RowSignal(g, row)
				} else {
					continue
				}
				arrived, ok := board.RaisedAt(wait)
				if !ok || r.Released < arrived || r.Released < r.Held || r.Done <= r.Released {
					t.Errorf("%s: arrived %g, held %g, released %g, done %g",
						r.Name, arrived, r.Held, r.Released, r.Done)
				}
			}

			wantSums := 0
			for i, s := range c.plan {
				if s > 1 && i >= 4/c.gpus {
					wantSums += c.gpus
				}
			}
			if engines && len(b.Comm.SumRecords()) != wantSums {
				t.Errorf("%d partial sums, want %d", len(b.Comm.SumRecords()), wantSums)
			}
			for _, p := range parts {
				if errs := p.Tracker.Errors(); len(errs) > 0 {
					t.Fatalf("tracker errors: %v", errs[:min(3, len(errs))])
				}
			}
		})
	}
}
