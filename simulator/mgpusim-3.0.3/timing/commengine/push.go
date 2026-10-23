package commengine

import (
	"fmt"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
)

// PushConfig arms an engine to all-gather the rows of a row-major matrix
// (Config.VBase, Rows x Cols 4-byte elements, the same address on every
// GPU's copy), cut into bands of BandRows rows, one band per message.
//
// The engine reads each band it sends once from local memory, then sends
// in the order of Sends; there is no reduction. A received band is written
// into this GPU's copy, and once L2 has acknowledged its last write the
// engine raises RowSignal(GPU, band) on Signals. The engine is done when
// every send is on the network and every expected band is written.
type PushConfig struct {
	BandRows uint32
	Sends    []PushSend
	Receives int
	Signals  *signal.Board
}

// PushSend is one message of a push: band Band to participant To.
type PushSend struct {
	Band, To int
}

// DirectOrder sends every band in own, in order, to every other participant,
// nearest index first. With per-peer queues (Engine.PerPeerQueues) on a
// per-pair mesh, all peers are served in parallel.
func DirectOrder(gpu, numGPU int, own []int) []PushSend {
	var sends []PushSend
	for _, band := range own {
		for d := 1; d < numGPU; d++ {
			sends = append(sends, PushSend{Band: band, To: (gpu + d) % numGPU})
		}
	}
	return sends
}

// RowSignal names the signal raised once band row of the pushed matrix has
// been written on participant gpu.
func RowSignal(gpu, row int) string {
	return fmt.Sprintf("gpu%d/a/row%d", gpu, row)
}

type pushState struct {
	data      map[int][]byte // bands read for sending
	next      int            // next entry of Sends
	sendsLeft int
	recvsLeft int
}

func (e *Engine) armPush() {
	pc := e.cfg.Push
	if pc.BandRows == 0 || e.cfg.Rows%pc.BandRows != 0 {
		panic("commengine: push bands must divide the matrix")
	}
	e.push = &pushState{data: map[int][]byte{},
		sendsLeft: len(pc.Sends), recvsLeft: pc.Receives}

	read := map[int]bool{}
	for _, s := range pc.Sends {
		if read[s.Band] {
			continue
		}
		read[s.Band] = true
		band := s.Band
		e.memJobs = append(e.memJobs, &memJob{
			segs: e.bandSegments(band), data: make([]byte, e.bandBytes()),
			onDone: func(now sim.VTimeInSec, data []byte) {
				e.push.data[band] = data
				e.pushSends(now)
			},
		})
	}
}

func (e *Engine) bandBytes() int {
	return int(e.cfg.Push.BandRows * e.cfg.Cols * 4)
}

// bandSegments lists a band's bytes as line-sized physical segments.
func (e *Engine) bandSegments(band int) []segment {
	n := e.bandBytes()
	base := e.cfg.VBase + uint64(band*n)
	var segs []segment
	for off := 0; off < n; {
		addr := base + uint64(off)
		size := min(n-off, int(addr/64*64+64-addr))
		segs = append(segs, segment{pAddr: e.cfg.VToP(addr), offset: off, size: size})
		off += size
	}
	return segs
}

// pushSends queues the sends whose bands have been read, in list order.
func (e *Engine) pushSends(now sim.VTimeInSec) {
	sends := e.cfg.Push.Sends
	for e.push.next < len(sends) {
		s := sends[e.push.next]
		data, ok := e.push.data[s.Band]
		if !ok {
			return
		}
		e.push.next++
		e.enqueueSend(now, s.Band, false, s.To, 0, data, func(now sim.VTimeInSec) {
			e.push.sendsLeft--
			e.pushCheckDone(now)
		})
	}
}

// pushRecv writes a received band into this GPU's copy and raises its
// signal once the writes are acknowledged.
func (e *Engine) pushRecv(msg *TileMsg) {
	band := msg.Group
	e.memJobs = append(e.memJobs, &memJob{
		write: true, segs: e.bandSegments(band), data: msg.Data,
		onDone: func(now sim.VTimeInSec, _ []byte) {
			if b := e.cfg.Push.Signals; b != nil {
				b.Raise(now, RowSignal(e.cfg.GPU, band), 0)
			}
			e.push.recvsLeft--
			e.pushCheckDone(now)
		},
	})
}

func (e *Engine) pushCheckDone(now sim.VTimeInSec) {
	if e.finished || e.push.sendsLeft > 0 || e.push.recvsLeft > 0 {
		return
	}
	e.finished = true
	if e.cfg.OnDone != nil {
		e.cfg.OnDone(e.cfg.GPU, now)
	}
}
