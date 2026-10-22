package mccl_test

import (
	"fmt"
	"testing"

	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

// TestAllReduceDirect checks every element of AllReduceDirect against the
// CPU mean, on 2 and 4 GPUs, with the buffer smaller than and equal to the
// data.
func TestAllReduceDirect(t *testing.T) {
	for _, c := range []struct{ gpus, size, buf int }{
		{2, 1029, 256}, {4, 1029, 256}, {4, 1029, 1029}, {2, 64, 64},
	} {
		t.Run(fmt.Sprintf("gpus=%d,size=%d,buf=%d", c.gpus, c.size, c.buf), func(t *testing.T) {
			platform := runner.MakeR9NanoBuilder().WithNumGPU(c.gpus).Build()
			d := platform.Driver
			d.Run()
			defer d.Terminate()
			ctx := d.Init()

			var ids []int
			datas := make([]driver.Ptr, c.gpus)
			bufs := make([]driver.Ptr, c.gpus)
			want := make([]float32, c.size)
			for i := 0; i < c.gpus; i++ {
				vals := make([]float32, c.size)
				for j := range vals {
					vals[j] = float32((i+1)*(j%7+1)) + float32(j)/16
					want[j] += vals[j] / float32(c.gpus)
				}
				d.SelectGPU(ctx, i+1)
				datas[i] = d.AllocateMemory(ctx, uint64(c.size*4))
				d.MemCopyH2D(ctx, datas[i], vals)
				bufs[i] = d.AllocateMemory(ctx, uint64(c.buf*4))
				ids = append(ids, i+1)
			}

			comms := mccl.CommInitAll(c.gpus, d, ctx, ids)
			mccl.AllReduceDirect(d, comms, datas, c.size, bufs, c.buf)

			for i := 0; i < c.gpus; i++ {
				got := make([]float32, c.size)
				d.SelectGPU(ctx, i+1)
				d.MemCopyD2H(ctx, got, datas[i])
				for j := range got {
					diff := got[j] - want[j]
					if diff > 1e-3*want[j] || -diff > 1e-3*want[j] {
						t.Fatalf("GPU %d element %d: got %v, want %v", i, j, got[j], want[j])
					}
				}
			}
		})
	}
}
