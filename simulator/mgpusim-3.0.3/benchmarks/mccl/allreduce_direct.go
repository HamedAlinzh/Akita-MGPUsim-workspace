package mccl

import (
	"log"

	"github.com/sarchlab/mgpusim/v3/driver"
)

// AllReduceDirect performs AllReduce average operation as a direct
// reduce-scatter and all-gather, the fair software collective on a fully
// connected fabric (the ring uses only neighbour links). In each pass of up
// to sizePerBuf elements, GPU j owns share j. Every GPU pushes each peer that
// peer's share once, into the peer's buffer; each owner adds the p-1
// received shares into its own and divides by p; then every owner pushes its
// share to every peer. Every message crosses one link, and pushes to
// different peers run on their own queues, so they overlap.
//
// bufs must hold (p-1) x ceil(sizePerBuf/p) elements on every GPU, which
// sizePerBuf elements do for any sizePerBuf >= (p-1)^2.
func AllReduceDirect(
	d *driver.Driver,
	comms []*Communicator,
	data []driver.Ptr,
	dataSize int,
	bufs []driver.Ptr,
	sizePerBuf int,
) {
	p := len(comms)
	if p < 2 {
		return
	}
	numThread := collectiveThreads()

	// qs[i][j]: GPU i's queue for pushes to peer j; qs[i][i] for its reduces.
	qs := make([][]*driver.CommandQueue, p)
	for i := range qs {
		d.SelectGPU(comms[i].Ctx, comms[i].GPUID)
		for j := 0; j < p; j++ {
			qs[i] = append(qs[i], d.CreateCommandQueue(comms[i].Ctx))
		}
	}
	drain := func() {
		for i := range qs {
			for _, q := range qs[i] {
				d.DrainCommandQueue(q)
			}
		}
	}
	push := func(i, j int, src, dst driver.Ptr, n int) {
		d.SelectGPU(comms[i].Ctx, comms[i].GPUID)
		d.EnqueueLaunchKernel(qs[i][j], coPush,
			[3]uint32{uint32(numThread), 1, 1}, [3]uint16{64, 1, 1},
			&pushKernelArgs{Src: src, Dst: dst, Size: uint32(n),
				NumThread: uint32(numThread)})
	}
	// slot is where sender i's share lands in receiver j's buffer.
	slot := func(i, j int) int {
		if i < j {
			return i
		}
		return i - 1
	}

	for offset := 0; offset < dataSize; offset += sizePerBuf {
		cur := min(sizePerBuf, dataSize-offset)
		each := (cur + p - 1) / p
		if (p-1)*each > sizePerBuf {
			log.Panicf("mccl: AllReduceDirect needs sizePerBuf >= (p-1)^2, "+
				"got %d for %d GPUs", sizePerBuf, p)
		}
		share := func(j int) int { return max(0, min(each, cur-j*each)) }
		at := func(ptr driver.Ptr, elem int) driver.Ptr {
			return ptr + driver.Ptr(4*elem)
		}

		// Reduce-scatter: each GPU sends each peer that peer's share.
		for i := 0; i < p; i++ {
			for j := 0; j < p; j++ {
				if j == i || share(j) == 0 {
					continue
				}
				push(i, j, at(data[i], offset+j*each), at(bufs[j], slot(i, j)*each),
					share(j))
			}
		}
		drain()

		// Each owner adds the received shares; the last add divides by p.
		for j := 0; j < p; j++ {
			if share(j) == 0 {
				continue
			}
			d.SelectGPU(comms[j].Ctx, comms[j].GPUID)
			for k := 0; k < p-1; k++ {
				last := uint32(0)
				if k == p-2 {
					last = 1
				}
				d.EnqueueLaunchKernel(qs[j][j], coReduce,
					[3]uint32{uint32(numThread), 1, 1}, [3]uint16{64, 1, 1},
					&allReduceReduceKernelArgs{
						Buf: at(bufs[j], k*each), Store: at(data[j], offset+j*each),
						Size: uint32(share(j)), NumThread: uint32(numThread),
						GPUNum: uint32(p), Last: last,
					})
			}
		}
		drain()

		// All-gather: each owner sends its share to every peer.
		for j := 0; j < p; j++ {
			for i := 0; i < p; i++ {
				if i == j || share(j) == 0 {
					continue
				}
				push(j, i, at(data[j], offset+j*each), at(data[i], offset+j*each),
					share(j))
			}
		}
		drain()
	}
}

// Accumulate adds src[i] into dst[i] on every communicator's GPU, n elements
// each, with the all-reduce's reduce kernel, and waits for all of them. It
// lets a software pipeline sum a GPU's split-K partials before an
// all-reduce.
func Accumulate(d *driver.Driver, comms []*Communicator, dst, src []driver.Ptr, n int) {
	qs := createCommandQueues(d, comms)
	numThread := collectiveThreads()
	for i, c := range comms {
		d.SelectGPU(c.Ctx, c.GPUID)
		d.EnqueueLaunchKernel(qs[i], coReduce,
			[3]uint32{uint32(numThread), 1, 1}, [3]uint16{64, 1, 1},
			&allReduceReduceKernelArgs{Buf: src[i], Store: dst[i], Size: uint32(n),
				NumThread: uint32(numThread), GPUNum: 1})
	}
	for _, q := range qs {
		d.DrainCommandQueue(q)
	}
}
