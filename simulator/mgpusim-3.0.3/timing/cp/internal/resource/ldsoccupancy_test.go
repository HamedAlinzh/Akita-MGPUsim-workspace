package resource

import (
	"os"
	"testing"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/kernels"
)

// timingCU has the dispatch resources of timing/cu.ComputeUnit.
type timingCU struct{}

func (timingCU) DispatchingPort() sim.Port { return nil }
func (timingCU) WfPoolSizes() []int        { return []int{10, 10, 10, 10} }
func (timingCU) VRegCounts() []int         { return []int{16384, 16384, 16384, 16384} }
func (timingCU) SRegCount() int            { return 3200 }
func (timingCU) LDSBytes() int             { return 64 * 1024 }

// residentWGs returns how many work-groups of numWf wavefronts and ldsBytes
// of packet LDS one CU holds at once.
func residentWGs(co *insts.HsaCo, numWf int, ldsBytes uint32, countLDS bool) int {
	pool := NewCUResourcePool()
	pool.CountDynamicLDS = countLDS
	pool.RegisterCU(timingCU{})
	cu := pool.GetCU(0)

	packet := &kernels.HsaKernelDispatchPacket{GroupSegmentSize: ldsBytes}
	n := 0
	for ; n < 1000; n++ {
		wg := kernels.NewWorkGroup()
		wg.CodeObject = co
		wg.Packet = packet
		for i := 0; i < numWf; i++ {
			wg.Wavefronts = append(wg.Wavefronts, kernels.NewWavefront())
		}
		if _, ok := cu.ReserveResourceForWG(wg); !ok {
			break
		}
	}
	return n
}

func TestLDSOccupancyLimitsWorkGroupsAt16KB(t *testing.T) {
	co := insts.NewHsaCo()
	co.WFSgprCount = 16
	co.WIVgprCount = 4

	// tileSize 16: 256 work-items (4 wavefronts) and 64x64 floats of LDS.
	const lds = 64 * 64 * 4
	if got := residentWGs(co, 4, lds, true); got != 4 {
		t.Errorf("with LDS counted, a 64 KB CU holds %d work-groups, want 4", got)
	}
	if got := residentWGs(co, 4, lds, false); got <= 4 {
		t.Errorf("without LDS counted, a CU holds %d work-groups; "+
			"the test no longer shows the difference", got)
	}
}

// TestGEMMResidency reports, for the patched GEMM kernel, how many
// work-groups one CU holds at each tile size.
func TestGEMMResidency(t *testing.T) {
	data, err := os.ReadFile("../../../../benchmarks/allreducegemm/kernels.hsaco")
	if err != nil {
		t.Skip(err)
	}
	co := kernels.LoadProgramFromMemory(data, "mmmKernel_local")
	t.Logf("mmmKernel_local: static LDS %d B, %d SGPR, %d VGPR",
		co.WGGroupSegmentByteSize, co.WFSgprCount, co.WIVgprCount)

	for _, tile := range []uint32{4, 8, 16} {
		numWf := int((tile*tile + 63) / 64)
		lds := co.WGGroupSegmentByteSize + (4*tile)*(4*tile)*4
		off := residentWGs(co, numWf, lds, false)
		on := residentWGs(co, numWf, lds, true)
		t.Logf("tileSize %2d: %d wavefronts, %5d B LDS per work-group; "+
			"resident per CU: %d without LDS counted, %d with",
			tile, numWf, lds, off, on)
		if on > off {
			t.Errorf("counting LDS raised residency at tileSize %d", tile)
		}
	}
}

func shapedCU(cap, ramp int) CUResource {
	pool := NewCUResourcePool()
	pool.WGCap, pool.WGRampStart = cap, ramp
	pool.RegisterCU(timingCU{})
	return pool.GetCU(0)
}

func oneWaveWG() *kernels.WorkGroup {
	co := insts.NewHsaCo()
	co.WFSgprCount, co.WIVgprCount = 16, 4
	wg := kernels.NewWorkGroup()
	wg.CodeObject = co
	wg.Packet = &kernels.HsaKernelDispatchPacket{}
	wg.Wavefronts = append(wg.Wavefronts, kernels.NewWavefront())
	return wg
}

func TestWGCapLimitsWorkGroupsInFlight(t *testing.T) {
	cu := shapedCU(2, 0)
	a, b := oneWaveWG(), oneWaveWG()
	for _, wg := range []*kernels.WorkGroup{a, b} {
		if _, ok := cu.ReserveResourceForWG(wg); !ok {
			t.Fatal("a work-group under the cap was refused")
		}
	}
	if _, ok := cu.ReserveResourceForWG(oneWaveWG()); ok {
		t.Fatal("a third work-group fit under a cap of 2")
	}
	cu.FreeResourcesForWG(a)
	if _, ok := cu.ReserveResourceForWG(oneWaveWG()); !ok {
		t.Fatal("no room after a work-group finished")
	}
}

func TestWGRampGrowsWithCompletions(t *testing.T) {
	cu := shapedCU(0, 1)
	first := oneWaveWG()
	if _, ok := cu.ReserveResourceForWG(first); !ok {
		t.Fatal("the first work-group was refused")
	}
	if _, ok := cu.ReserveResourceForWG(oneWaveWG()); ok {
		t.Fatal("a second work-group fit before any completed")
	}
	cu.FreeResourcesForWG(first) // the cap is now 2
	for i := 0; i < 2; i++ {
		if _, ok := cu.ReserveResourceForWG(oneWaveWG()); !ok {
			t.Fatalf("work-group %d refused after the ramp grew", i)
		}
	}
	if _, ok := cu.ReserveResourceForWG(oneWaveWG()); ok {
		t.Fatal("a third work-group fit under a ramped cap of 2")
	}
}
