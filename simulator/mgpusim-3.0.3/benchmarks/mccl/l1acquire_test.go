package mccl

import (
	"testing"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

// staleL1Reads reproduces the hazard behind the 4-GPU overlap failures at
// a tiny size, on one GPU. L1 vector caches are per CU and not coherent,
// so a kernel can only rely on fresh data if L1 is invalidated when it is
// dispatched. The dispatcher hands work-groups to CUs in a rotation that
// continues across kernels, so the scenario steers the rotation:
//
//  1. push data -> scratch, 16 work-groups: these CUs cache data's lines.
//  2. push twos -> data, 4 work-groups: the next CUs overwrite data.
//  3. an empty kernel whose work-groups bring the rotation back to CU 0.
//  4. push data -> out, 16 work-groups: each line is read again by the CU
//     that cached it in step 1.
//
// It returns how many elements of out are not 2.
func staleL1Reads(t *testing.T, acquire bool) int {
	t.Helper()

	platform := runner.MakeR9NanoBuilder().
		WithNumGPU(1).
		WithL1AcquireOnLaunch(acquire).
		Build()
	d := platform.Driver
	d.Run()
	defer d.Terminate()

	ctx := d.Init()
	d.SelectGPU(ctx, 1)

	const n = 4096
	const numThread = 1024
	ones := make([]float32, n)
	twos := make([]float32, n)
	for i := range ones {
		ones[i] = 1
		twos[i] = 2
	}

	alloc := func() driver.Ptr { return d.AllocateMemory(ctx, n*4) }
	data, twosBuf, scratch, out := alloc(), alloc(), alloc(), alloc()
	d.MemCopyH2D(ctx, data, ones)
	d.MemCopyH2D(ctx, twosBuf, twos)

	q := d.CreateCommandQueue(ctx)
	push := func(src, dst driver.Ptr, size, threads uint32, wgSize uint16) {
		d.EnqueueLaunchKernel(q, coPush,
			[3]uint32{threads, 1, 1}, [3]uint16{wgSize, 1, 1},
			&pushKernelArgs{Src: src, Dst: dst, Size: size, NumThread: threads})
		d.DrainCommandQueue(q)
	}
	numCU := uint32(len(platform.GPUs[0].CUs))
	push(data, scratch, n, numThread, 64)    // 16 work-groups
	push(twosBuf, data, n, numThread, 256)   // 4 work-groups
	filler := numCU - 16 - 4                 // work-groups left in the rotation
	push(scratch, scratch, 0, filler*64, 64) // Size 0: every thread exits
	push(data, out, n, numThread, 64)

	got := make([]float32, n)
	d.MemCopyD2H(ctx, got, out)

	wrong := 0
	for _, v := range got {
		if v != 2 {
			wrong++
		}
	}
	return wrong
}

func TestKernelReadsAreStaleWithoutL1Acquire(t *testing.T) {
	// Guards the test below: if this scenario stopped exercising stale L1
	// lines, the acquire test would pass for the wrong reason.
	wrong := staleL1Reads(t, false)
	if wrong == 0 {
		t.Fatal("expected stale reads without an L1 acquire; " +
			"the scenario no longer exercises the hazard")
	}
	t.Logf("%d of 4096 elements stale without the L1 acquire", wrong)
}

func TestL1AcquireOnLaunchGivesFreshReads(t *testing.T) {
	if wrong := staleL1Reads(t, true); wrong != 0 {
		t.Fatalf("%d of 4096 elements stale with the L1 acquire on", wrong)
	}
}
