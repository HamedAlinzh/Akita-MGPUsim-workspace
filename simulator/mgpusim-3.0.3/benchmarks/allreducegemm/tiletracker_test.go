package allreducegemm_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/allreducegemm"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// trackTiles runs a 1-GPU GEMM with the tile tracker armed on C and checks
// that every output tile is reported complete exactly once, after the last
// acknowledged store into it, holding the final values.
func trackTiles(t *testing.T, m, k, n, tileSize uint32) {
	t.Helper()
	trackTilesOnCUs(t, 64, m, k, n, tileSize)
}

func trackTilesOnCUs(t *testing.T, numCU int, m, k, n, tileSize uint32) {
	t.Helper()
	trackTilesWith(t, false, numCU, m, k, n, tileSize)
}

func trackTilesWith(
	t *testing.T, ldsOccupancy bool, numCU int, m, k, n, tileSize uint32,
) {
	t.Helper()

	platform := runner.MakeR9NanoBuilder().
		WithNumGPU(1).
		WithNumCUPerGPU(numCU).
		WithLDSOccupancy(ldsOccupancy).
		Build()
	d := platform.Driver
	d.Run()
	defer d.Terminate()

	tracker := platform.GPUs[0].TileTracker
	tileEdge := 4 * tileSize
	snapshots := map[int][]byte{}
	var launch float64
	tracker.OnComplete = func(ev tiletracker.TileEvent, data []byte) {
		snapshots[ev.Tile] = data
	}

	b := allreducegemm.NewBenchmark(d)
	b.M, b.K, b.N = m, k, n
	b.TileSize = tileSize
	b.SelectGPU([]int{1})
	b.EnableVerification()
	b.OnPartialC = func(ctx *driver.Context, _, _ int, c driver.Ptr) {
		tracker.Arm(region(t, d, ctx, uint64(c), m, n, tileEdge))
		launch = float64(d.Engine.CurrentTime())
	}

	b.Run()
	tracker.Disarm()
	b.Verify()

	for _, e := range tracker.Errors() {
		t.Error(e)
	}

	events := tracker.Events()
	numTiles := tracker.NumTiles()
	if want := int((m / tileEdge) * (n / tileEdge)); numTiles != want {
		t.Fatalf("tracker has %d tiles, want %d", numTiles, want)
	}

	times := map[int]int{}
	for _, ev := range events {
		times[ev.Tile]++
		if ev.Complete <= ev.LastStore {
			t.Errorf("tile %d completed at %g, not after its last store at %g",
				ev.Tile, ev.Complete, ev.LastStore)
		}
	}
	for tile := 0; tile < numTiles; tile++ {
		if times[tile] != 1 {
			t.Errorf("tile %d reported complete %d times", tile, times[tile])
		}
	}
	if missing, pending := tracker.Incomplete(); len(missing) > 0 || pending > 0 {
		t.Errorf("incomplete tiles %v, %d stores unacknowledged", missing, pending)
	}

	// The bytes L2 held when the tile completed must be the final result.
	c := b.ResultC()
	tilesX := n / tileEdge
	for tile, data := range snapshots {
		r0 := uint32(tile) / tilesX * tileEdge
		c0 := uint32(tile) % tilesX * tileEdge
		for i := uint32(0); i < tileEdge*tileEdge; i++ {
			got := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
			want := c[(r0+i/tileEdge)*n+c0+i%tileEdge]
			if got != want {
				t.Errorf("tile %d element %d: %v at completion, %v final",
					tile, i, got, want)
				break
			}
		}
	}

	report(t, events, launch, numCU, m, k, n, tileSize)
}

// region describes C to the tracker, with its pages' physical addresses.
func region(
	t *testing.T, d *driver.Driver, ctx *driver.Context,
	c uint64, m, n, tileEdge uint32,
) tiletracker.Region {
	t.Helper()
	pageSize := uint64(1) << d.Log2PageSize
	pages := map[uint64]uint64{}
	for v := c &^ (pageSize - 1); v < c+uint64(m)*uint64(n)*4; v += pageSize {
		p, ok := d.PhysicalAddress(ctx, v)
		if !ok {
			t.Fatalf("page %#x of C is not mapped", v)
		}
		pages[p] = v
	}
	return tiletracker.Region{
		VBase: c, Rows: m, Cols: n, TileRows: tileEdge, TileCols: tileEdge,
		PageMap: pages, Log2PageSize: d.Log2PageSize,
	}
}

// report logs a summary and, if TILETRACKER_OUT names a directory, writes
// every completion to a CSV there.
func report(
	t *testing.T, events []tiletracker.TileEvent, launch float64,
	numCU int, m, k, n, tileSize uint32,
) {
	t.Helper()
	if len(events) == 0 {
		return
	}
	us := func(v float64) float64 { return (v - launch) * 1e6 }
	first := float64(events[0].Complete)
	last := float64(events[len(events)-1].Complete)
	minStores, maxStores := events[0].NumStores, events[0].NumStores
	var maxLag float64
	for _, ev := range events {
		minStores = min(minStores, ev.NumStores)
		maxStores = max(maxStores, ev.NumStores)
		maxLag = max(maxLag, float64(ev.Complete-ev.LastStore))
	}
	t.Logf("%d CUs, M=%d N=%d K=%d tileSize %d: %d tiles, each complete once; "+
		"completions %.3f-%.3f us after the hook; %d-%d L2 stores per tile; "+
		"last store to completion up to %.1f ns",
		numCU, m, n, k, tileSize, len(events), us(first), us(last),
		minStores, maxStores, maxLag*1e9)

	dir := os.Getenv("TILETRACKER_OUT")
	if dir == "" {
		return
	}
	path := filepath.Join(dir,
		fmt.Sprintf("tiles_cu%d_M%d_N%d_K%d_t%d.csv", numCU, m, n, k, tileSize))
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	fmt.Fprintln(f, "order,tile,tile_row,tile_col,"+
		"first_store_us,last_store_us,complete_us,l2_stores")
	for i, ev := range events {
		fmt.Fprintf(f, "%d,%d,%d,%d,%.6f,%.6f,%.6f,%d\n",
			i, ev.Tile, ev.Row, ev.Col, us(float64(ev.FirstStore)),
			us(float64(ev.LastStore)), us(float64(ev.Complete)), ev.NumStores)
	}
}

// The spike's shape, M=N=128, K=256: one wave of work-groups at every
// tile size.
func TestTileTrackerTile4(t *testing.T)  { trackTiles(t, 128, 256, 128, 4) }
func TestTileTrackerTile8(t *testing.T)  { trackTiles(t, 128, 256, 128, 8) }
func TestTileTrackerTile16(t *testing.T) { trackTiles(t, 128, 256, 128, 16) }

// 256 work-groups for 64 CUs. A CU holds several work-groups at once, so
// they still run concurrently.
func TestTileTrackerManyWorkGroups(t *testing.T) { trackTiles(t, 256, 256, 256, 4) }

// One CU, so work-groups run in sequential waves and tiles complete at
// clearly different times.
func TestTileTrackerSequentialWaves(t *testing.T) {
	trackTilesOnCUs(t, 1, 128, 256, 128, 16)
	trackTilesOnCUs(t, 1, 128, 256, 128, 8)
}

// The simulated shapes proposed for the representative points (Step 0
// follow-up), with LDS counted in occupancy: K per GPU 256, tileSize 8,
// 4 CUs of 12 resident work-groups each.
func TestTileTrackerRepresentative(t *testing.T) {
	trackTilesWith(t, true, 4, 384, 256, 256, 8) // 96 tiles, 2.0 waves
	trackTilesWith(t, true, 4, 512, 256, 320, 8) // 160 tiles, 3.3 waves
}

// Checks that 4 CUs is not a toy: 16 CUs with 4x the tiles is also 2.0
// waves and should complete tiles with a similar spread.
func TestTileTrackerScaleCheck(t *testing.T) {
	trackTilesWith(t, true, 16, 768, 256, 512, 8) // 384 tiles, 2.0 waves
}
