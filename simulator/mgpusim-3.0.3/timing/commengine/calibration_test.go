package commengine_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// calibrate measures the engine on the real 2-GPU platform without a GEMM:
// tiles of a 32-row C are declared complete at once on both GPUs, and the
// engines reduce-scatter them (tile t is owned by GPU t mod 2, so each GPU
// sends every other tile, one 4 KiB message each, to its peer).
func calibrate(
	t *testing.T, bwGBps, latencyNS, overheadNS float64, tiles int,
) []*commengine.MsgRecord {
	t.Helper()

	platform := runner.MakeR9NanoBuilder().
		WithNumGPU(2).
		WithLinkBandwidth(bwGBps*1e9).
		WithLinkLatency(latencyNS).
		WithCommEngine(true, overheadNS).
		Build()
	d := platform.Driver
	d.Run()
	defer d.Terminate()

	ctx := d.Init()
	const edge = 32
	cols := uint32(edge * tiles)
	var cs []driver.Ptr
	var parts []commengine.Participant
	for i, gpu := range platform.GPUs {
		d.SelectGPU(ctx, i+1)
		cs = append(cs, d.AllocateMemory(ctx, uint64(edge*cols*4)))
		parts = append(parts, commengine.Participant{
			Tracker: gpu.TileTracker, Engine: gpu.CommEngine})
	}

	s := commengine.NewSession(d, parts)
	s.Arm(ctx, cs, edge, cols, edge, commengine.Options{
		TilesPerMsg: 1, Mode: commengine.ModeTile,
		Collective: commengine.ReduceScatter,
	})

	d.Engine.Pause()
	now := d.Engine.CurrentTime()
	for _, p := range parts {
		for tile := 0; tile < tiles; tile++ {
			e, ev := p.Engine, tiletracker.TileEvent{Tile: tile, Complete: now}
			d.Engine.Schedule(sim.NewEventBase(now, handler(func() {
				e.OnTileLocal(ev)
			})))
		}
	}
	d.Engine.Continue()
	s.Wait()

	return s.Records()
}

type handler func()

func (h handler) Handle(sim.Event) error { h(); return nil }

// The three representative points' knobs (results/step0_ratio/points.csv).
var calibrationPoints = []struct {
	name                          string
	bwGBps, latencyNS, overheadNS float64
}{
	{"f~0.8", 0.1807, 1904, 9163},
	{"f~0.5", 0.6325, 544, 2618},
	{"f~0.25", 1.1504, 299, 1440},
}

// TestCalibration reports, per point, one message's latency against
// -msg-overhead + -link-latency + its transfer time, and a stream's payload
// rate as a share of -link-bw. With CALIBRATION_OUT set it writes a CSV.
func TestCalibration(t *testing.T) {
	if testing.Short() {
		t.Skip("slow")
	}

	var csv []string
	csv = append(csv, "point,link_bw_gbps,link_latency_ns,msg_overhead_ns,"+
		"single_latency_ns,stream_ns_per_msg,expected_latency_ns,"+
		"latency_excess_ns,stream_bw_share,expected_bw_share")
	for _, p := range calibrationPoints {
		// One message per direction: 2 tiles, GPU 1 sends tile 0, GPU 0
		// sends tile 1.
		single := calibrate(t, p.bwGBps, p.latencyNS, p.overheadNS, 2)
		lat := float64(single[0].Received-single[0].Queued) * 1e9

		// A stream: 128 tiles, 64 messages each way, all ready at once.
		stream := calibrate(t, p.bwGBps, p.latencyNS, p.overheadNS, 128)
		var first, last sim.VTimeInSec = 1, 0
		payload := 0
		n := 0
		for _, r := range stream {
			if r.From != 0 {
				continue
			}
			n++
			payload += r.PayloadBytes
			first = min(first, r.Sent)
			last = max(last, r.Received)
		}
		perMsg := float64(last-first) * 1e9 / float64(n)
		share := float64(payload) / (float64(last-first) * p.bwGBps * 1e9)

		transfer := float64(single[0].TrafficBytes) / (p.bwGBps * 0.8)
		expected := p.overheadNS + p.latencyNS + transfer
		expectedShare := 0.8 * 4096 / 4112

		t.Logf("%s: one message %.0f ns vs overhead+latency+transfer %.0f "+
			"(excess %.0f ns, %.1f%%); stream %.0f ns per message, %.3f of "+
			"-link-bw (Akita's 25%% encoding overhead alone gives %.3f)",
			p.name, lat, expected, lat-expected, (lat-expected)/expected*100,
			perMsg, share, expectedShare)
		csv = append(csv, fmt.Sprintf("%s,%g,%g,%g,%.0f,%.0f,%.0f,%.0f,%.4f,%.4f",
			p.name, p.bwGBps, p.latencyNS, p.overheadNS, lat, perMsg, expected,
			lat-expected, share, expectedShare))
	}

	if path := os.Getenv("CALIBRATION_OUT"); path != "" {
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		for _, line := range csv {
			fmt.Fprintln(f, line)
		}
	}
}
