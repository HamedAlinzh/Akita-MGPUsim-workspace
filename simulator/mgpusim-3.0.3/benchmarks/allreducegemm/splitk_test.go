package allreducegemm_test

import (
	"fmt"
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
)

// runSplitK all-reduces a 128x128 C (K/GPU 128) on 2 GPUs with the engines,
// its first and last 32-row bands split s ways, and verifies it.
func runSplitK(t *testing.T, s int, splitSend bool, plan ...int) {
	t.Helper()

	platform := runner.MakeR9NanoBuilder().
		WithNumGPU(2).
		WithNumCUPerGPU(4).
		WithNumDispatchers(16).
		WithOrderedDispatch(true).
		WithCommEngine(true, 0).
		Build()
	d := platform.Driver
	d.Run()
	defer d.Terminate()

	var parts []commengine.Participant
	for _, gpu := range platform.GPUs {
		parts = append(parts, commengine.Participant{
			Tracker: gpu.TileTracker, Engine: gpu.CommEngine})
	}

	b := allreducegemm.NewBenchmark(d)
	b.M, b.K, b.N = 128, 256, 128
	b.SplitKWays, b.SplitKRows = s, 32
	if plan != nil {
		b.SplitKWays, b.SplitKRows, b.SplitKPlan = 0, 0, plan
	}
	b.Comm = commengine.NewSession(d, parts)
	b.CommOptions = commengine.Options{
		TilesPerMsg: 1, Mode: commengine.ModeTile, SplitSend: splitSend}
	b.SelectGPU([]int{1, 2})
	b.EnableVerification()
	b.Run()
	b.Verify()

	for _, p := range parts {
		if errs := p.Tracker.Errors(); len(errs) > 0 {
			t.Fatalf("tracker errors: %v", errs[:min(3, len(errs))])
		}
	}

	// Without pre-sum, each split tile a GPU does not own costs s messages.
	want := 8 // per direction: the 8 of 16 tiles the peer owns
	if splitSend && plan == nil {
		want += 4 * (s - 1) // 2 of them in each split band
	}
	count := map[[2]int]int{}
	for _, r := range b.Comm.Records() {
		if !r.Reduced {
			count[[2]int{r.From, r.To}]++
		}
	}
	for dir, n := range count {
		if n != want {
			t.Errorf("%v: %d partial messages, want %d", dir, n, want)
		}
	}
}

func TestSplitK(t *testing.T) {
	for _, c := range []struct {
		s         int
		splitSend bool
	}{{2, false}, {2, true}, {4, false}} {
		t.Run(fmt.Sprintf("s=%d,splitSend=%v", c.s, c.splitSend), func(t *testing.T) {
			runSplitK(t, c.s, c.splitSend)
		})
	}
	t.Run("plan=4,1,1,2", func(t *testing.T) { runSplitK(t, 0, false, 4, 1, 1, 2) })
}

// TestSWOverlap runs the software-overlap baseline on 2 GPUs (128x128, K/GPU
// 128): s = 1 bands and a split plan, verified against the CPU.
func TestSWOverlap(t *testing.T) {
	for _, c := range []struct {
		name  string
		bands []uint32
		plan  []int
	}{
		{"s=1", []uint32{32, 32, 64}, nil},
		{"plan=4,1,2,2", []uint32{32, 32, 32, 32}, []int{4, 1, 2, 2}},
	} {
		t.Run(c.name, func(t *testing.T) {
			platform := runner.MakeR9NanoBuilder().
				WithNumGPU(2).
				WithNumCUPerGPU(4).
				WithNumDispatchers(16).
				WithOrderedDispatch(true).
				Build()
			d := platform.Driver
			d.Run()
			defer d.Terminate()

			b := allreducegemm.NewBenchmark(d)
			b.M, b.K, b.N = 128, 256, 128
			b.SWBands, b.SplitKPlan = c.bands, c.plan
			b.SelectGPU([]int{1, 2})
			b.EnableVerification()
			b.Run()
			b.Verify()
		})
	}
}
