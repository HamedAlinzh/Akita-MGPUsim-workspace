package mccl

import (
	"github.com/sarchlab/mgpusim/v3/driver"
)

// AllGatherDirect gathers equal shards: GPU i holds elements
// [i x shardSize, (i+1) x shardSize) of data[i] and pushes them to the same
// place in every peer's data, one push kernel per peer on its own queue (the
// all-gather half of AllReduceDirect), then waits for all of them.
func AllGatherDirect(
	d *driver.Driver,
	comms []*Communicator,
	data []driver.Ptr,
	shardSize int,
) {
	p := len(comms)
	numThread := collectiveThreads()
	var qs []*driver.CommandQueue
	for i := 0; i < p; i++ {
		d.SelectGPU(comms[i].Ctx, comms[i].GPUID)
		off := driver.Ptr(4 * i * shardSize)
		for j := 0; j < p; j++ {
			if j == i {
				continue
			}
			q := d.CreateCommandQueue(comms[i].Ctx)
			qs = append(qs, q)
			d.EnqueueLaunchKernel(q, coPush,
				[3]uint32{uint32(numThread), 1, 1}, [3]uint16{64, 1, 1},
				&pushKernelArgs{Src: data[i] + off, Dst: data[j] + off,
					Size: uint32(shardSize), NumThread: uint32(numThread)})
		}
	}
	for _, q := range qs {
		d.DrainCommandQueue(q)
	}
}
