package mccl

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
)

// SignalDelays are the seconds between a kernel completing and its signal
// being raised: Local for a consumer on the same GPU, Peer on another GPU.
type SignalDelays struct {
	Local, Peer float64
}

// onSignal returns a copy of co that the CP holds until every signal in wait
// is raised, raises signal delay seconds after it completes, and runs on the
// CUs reserved for communication.
func onSignal(co *insts.HsaCo, wait []string, signal string, delay float64) *insts.HsaCo {
	c := *co
	c.WaitSignals, c.Signal, c.SignalDelay, c.CommCUs = wait, signal, delay, true
	return &c
}

// AllReduceDirectOnSignal enqueues, without waiting, AllReduceDirect's data
// movement for dataSize elements at data[i], as kernels that the command
// processors hold until their inputs are ready instead of host drains: GPU
// i's pushes start once ready[i] is raised; the owner of each share adds the
// p-1 received shares one after another, each as soon as it has arrived (and
// ready[owner]); the last add divides by p and releases the owner's pushes of
// the reduced share to every peer. Signals are named tag/push/i-j,
// tag/red/j/k and tag/gather/j-i. bufs[j] must hold (p-1) x ceil(dataSize/p)
// elements and is not reused before the call's kernels finish. Every kernel
// gets its own queue, returned; done lists the gathers' signals.
func AllReduceDirectOnSignal(
	d *driver.Driver,
	comms []*Communicator,
	data []driver.Ptr,
	dataSize int,
	bufs []driver.Ptr,
	ready []string,
	tag string,
	delay SignalDelays,
) (queues []*driver.CommandQueue, done []string) {
	p := len(comms)
	numThread := uint32(collectiveThreads())
	each := (dataSize + p - 1) / p
	share := func(j int) int { return max(0, min(each, dataSize-j*each)) }
	at := func(ptr driver.Ptr, elem int) driver.Ptr {
		return ptr + driver.Ptr(4*elem)
	}
	slot := func(i, j int) int {
		if i < j {
			return i
		}
		return i - 1
	}
	launch := func(gpu int, co *insts.HsaCo, args interface{}) {
		d.SelectGPU(comms[gpu].Ctx, comms[gpu].GPUID)
		q := d.CreateCommandQueue(comms[gpu].Ctx)
		d.EnqueueLaunchKernel(q, co, [3]uint32{numThread, 1, 1},
			[3]uint16{64, 1, 1}, args)
		queues = append(queues, q)
	}
	push := func(i, j int) string { return fmt.Sprintf("%s/push/%d-%d", tag, i, j) }
	red := func(j, k int) string { return fmt.Sprintf("%s/red/%d/%d", tag, j, k) }

	// Reduce-scatter: each GPU sends each peer that peer's share.
	for i := 0; i < p; i++ {
		for j := 0; j < p; j++ {
			if j == i || share(j) == 0 {
				continue
			}
			launch(i, onSignal(coPush, []string{ready[i]}, push(i, j), delay.Peer),
				&pushKernelArgs{Src: at(data[i], j*each),
					Dst: at(bufs[j], slot(i, j)*each), Size: uint32(share(j)),
					NumThread: numThread})
		}
	}

	// Each owner adds the received shares in slot order.
	for j := 0; j < p; j++ {
		if share(j) == 0 {
			continue
		}
		for k := 0; k < p-1; k++ {
			sender := k
			if k >= j {
				sender = k + 1
			}
			wait := []string{ready[j], push(sender, j)}
			if k > 0 {
				wait = append(wait, red(j, k-1))
			}
			last := uint32(0)
			if k == p-2 {
				last = 1
			}
			launch(j, onSignal(coReduce, wait, red(j, k), delay.Local),
				&allReduceReduceKernelArgs{Buf: at(bufs[j], k*each),
					Store: at(data[j], j*each), Size: uint32(share(j)),
					NumThread: numThread, GPUNum: uint32(p), Last: last})
		}
	}

	// All-gather: each owner sends its reduced share to every peer.
	for j := 0; j < p; j++ {
		if share(j) == 0 {
			continue
		}
		for i := 0; i < p; i++ {
			if i == j {
				continue
			}
			name := fmt.Sprintf("%s/gather/%d-%d", tag, j, i)
			launch(j, onSignal(coPush, []string{red(j, p-2)}, name, delay.Local),
				&pushKernelArgs{Src: at(data[j], j*each), Dst: at(data[i], j*each),
					Size: uint32(share(j)), NumThread: numThread})
			done = append(done, name)
		}
	}
	return queues, done
}

// AccumulateOnSignal enqueues, without waiting, Accumulate's add of n
// elements of src into dst on c's GPU, held until every signal in wait is
// raised; it raises signal delay seconds after it completes.
func AccumulateOnSignal(
	d *driver.Driver, c *Communicator, dst, src driver.Ptr, n int,
	wait []string, signal string, delay float64,
) *driver.CommandQueue {
	numThread := uint32(collectiveThreads())
	d.SelectGPU(c.Ctx, c.GPUID)
	q := d.CreateCommandQueue(c.Ctx)
	d.EnqueueLaunchKernel(q, onSignal(coReduce, wait, signal, delay),
		[3]uint32{numThread, 1, 1}, [3]uint16{64, 1, 1},
		&allReduceReduceKernelArgs{Buf: src, Store: dst, Size: uint32(n),
			NumThread: numThread, GPUNum: 1})
	return q
}
