package mccl

import "github.com/sarchlab/mgpusim/v3/driver"

// AllReduceNaive performs an AllReduce average operation using a centralized
// gather-reduce-broadcast strategy: every non-root GPU pushes its data
// directly to a root GPU (rank 0), the root sequentially reduces (sums, then
// averages on the last step) the incoming data locally, and finally pushes
// the result directly back to every other GPU. Unlike AllReduceRing, there
// are no intermediate hops, so the root GPU's link bandwidth becomes the
// bottleneck as the GPU count grows -- this is the naive baseline that Ring
// All-Reduce is designed to improve on.
//
// Each push/reduce step uses driver.LaunchKernel (a fresh command queue
// created and drained per call) rather than a long-lived, reused queue.
// AllReduceRing's helpers reuse persistent per-GPU queues across many
// dispatches, which is safe for Ring's phase-batched, dependency-free
// enqueue pattern; AllReduceNaive's tight, sequential read-after-write
// chain (push into root's buffer, immediately reduce it, immediately
// broadcast the result) does not tolerate that pattern's completion timing,
// so it sticks to the one-queue-per-operation idiom the driver itself uses
// for synchronous, ordered execution.
func AllReduceNaive(
	d *driver.Driver,
	comms []*Communicator,
	data []driver.Ptr,
	dataSize int,
	bufs []driver.Ptr,
	sizePerBuf int,
) {
	const root = 0

	numGPU := len(comms)
	bigSteps := (dataSize-1)/sizePerBuf + 1
	numThread := 1024

	for j := 0; j < bigSteps; j++ {
		dataOffset := j * sizePerBuf
		currChunkSize := min(sizePerBuf, dataSize-dataOffset)

		naiveGatherReduce(
			d, comms, data, bufs, root, numGPU, numThread,
			uint64(currChunkSize), uint64(dataOffset),
		)

		naiveBroadcast(
			d, comms, data, root, numGPU, numThread,
			uint64(currChunkSize), uint64(dataOffset),
		)
	}
}

// naiveGatherReduce sequentially gathers every non-root GPU's chunk into the
// root's scratch buffer and reduces it into the root's data, averaging on
// the last source so that the final value equals the mean of all numGPU
// original chunks (including the root's own, untouched value).
func naiveGatherReduce(
	d *driver.Driver,
	comms []*Communicator,
	data, bufs []driver.Ptr,
	root, numGPU, numThread int,
	chunkSize, offset uint64,
) {
	rootBuf := bufs[root]
	rootStore := data[root] + driver.Ptr(4*offset)

	nonRootIndex := 0
	numNonRoot := numGPU - 1

	for i := 0; i < numGPU; i++ {
		if i == root {
			continue
		}

		src := data[i] + driver.Ptr(4*offset)

		d.SelectGPU(comms[i].Ctx, comms[i].GPUID)
		d.LaunchKernel(
			comms[i].Ctx,
			coPush,
			[3]uint32{uint32(numThread), 1, 1},
			[3]uint16{64, 1, 1},
			&pushKernelArgs{
				Src:       src,
				Dst:       rootBuf,
				Size:      uint32(chunkSize),
				NumThread: uint32(numThread),
			},
		)

		var lastReduce uint32
		if nonRootIndex == numNonRoot-1 {
			lastReduce = 1
		}
		nonRootIndex++

		d.SelectGPU(comms[root].Ctx, comms[root].GPUID)
		d.LaunchKernel(
			comms[root].Ctx,
			coReduce,
			[3]uint32{uint32(numThread), 1, 1},
			[3]uint16{64, 1, 1},
			&allReduceReduceKernelArgs{
				Buf:       rootBuf,
				Store:     rootStore,
				Size:      uint32(chunkSize),
				NumThread: uint32(numThread),
				GPUNum:    uint32(numGPU),
				Last:      lastReduce,
			},
		)
	}
}

// naiveBroadcast pushes the reduced chunk directly from the root to every
// other GPU, without any intermediate hops.
func naiveBroadcast(
	d *driver.Driver,
	comms []*Communicator,
	data []driver.Ptr,
	root, numGPU, numThread int,
	chunkSize, offset uint64,
) {
	src := data[root] + driver.Ptr(4*offset)

	// The reduce kernel's write to src (root's own data) can still be
	// settling through root's memory hierarchy when the first broadcast
	// kernel tries to read it, since DrainCommandQueue only guarantees the
	// reduce kernel finished dispatching, not that its write is externally
	// visible yet. A host round trip forces it to fully commit before any
	// push reads it remotely.
	d.SelectGPU(comms[root].Ctx, comms[root].GPUID)
	settled := make([]float32, chunkSize)
	d.MemCopyD2H(comms[root].Ctx, settled, src)

	for i := 0; i < numGPU; i++ {
		if i == root {
			continue
		}

		dst := data[i] + driver.Ptr(4*offset)

		d.SelectGPU(comms[root].Ctx, comms[root].GPUID)
		d.LaunchKernel(
			comms[root].Ctx,
			coPush,
			[3]uint32{uint32(numThread), 1, 1},
			[3]uint16{64, 1, 1},
			&pushKernelArgs{
				Src:       src,
				Dst:       dst,
				Size:      uint32(chunkSize),
				NumThread: uint32(numThread),
			},
		)
	}
}
