package commengine

import (
	"fmt"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/timing/signal"
	"github.com/sarchlab/mgpusim/v3/timing/tiletracker"
)

// PushConfig arms an engine to all-gather the rows of a row-major matrix
// (Config.VBase, Rows x Cols 4-byte elements, the same address on every
// GPU's copy), cut into bands of BandRows rows.
//
// The engine reads each band it sends once from local memory, then sends
// in the order of Sends; there is no reduction. A send with Parts = 1 is one
// message, written into the receiver's copy of the matrix; once L2 has
// acknowledged its last write the receiver raises RowSignal(GPU, band) on
// Signals. A send with Parts = s > 1 is s messages in K order, chunk c
// holding columns [c Cols/s, (c+1) Cols/s) of the band's rows; the receiver
// writes it into its chunk buffer Chunks[s][c] (a Rows x Cols/s matrix, at
// the band's rows) and raises ChunkSignal(GPU, band, c). Receives counts the
// messages this engine expects.
//
// Sums are the partial sums of rows a split GEMM computed in pieces (see
// PushSum), started once the tile tracker reports every tile of the row
// complete in all its copies (OnRowTile). The engine is done when every
// send is on the network, every expected message is written and every sum
// is written.
type PushConfig struct {
	BandRows uint32
	Sends    []PushSend
	Receives int
	Signals  *signal.Board
	Chunks   map[int][]uint64

	Sums     []PushSum
	RowTiles int // tiles per row of the summed matrix
}

// PushSend is one send of a push: band Band to participant To, in Parts K
// chunks (0 means 1).
type PushSend struct {
	Band, To, Parts int
}

// PushSum adds the partial buffers Srcs into Dst, Bytes each, once every
// tile of row Row of the tracked matrix is complete. Dst and Srcs are
// virtual addresses on this GPU.
type PushSum struct {
	Row   int
	Dst   uint64
	Srcs  []uint64
	Bytes int
}

// SumRecord is one partial sum's timeline: Ready when the row's last tile
// was complete, Done when L2 acknowledged the sum's last write.
type SumRecord struct {
	GPU, Row    int
	Ready, Done sim.VTimeInSec
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

// ChunkSignal names the signal raised once K chunk chunk of a split band
// row has been written on participant gpu.
func ChunkSignal(gpu, row, chunk int) string {
	return fmt.Sprintf("gpu%d/a/row%d/chunk%d", gpu, row, chunk)
}

type pushState struct {
	data      map[int][]byte // bands read for sending
	next      int            // next entry of Sends
	sendsLeft int
	recvsLeft int
	sumsLeft  int
	rowTiles  map[int]int // completed tiles by row, for the sums
	sums      []SumRecord
}

func (e *Engine) armPush() {
	pc := e.cfg.Push
	if pc.BandRows == 0 || e.cfg.Rows%pc.BandRows != 0 {
		panic("commengine: push bands must divide the matrix")
	}
	e.push = &pushState{data: map[int][]byte{}, recvsLeft: pc.Receives,
		sumsLeft: len(pc.Sums), rowTiles: map[int]int{}}

	read := map[int]bool{}
	for _, s := range pc.Sends {
		if parts := max(1, s.Parts); e.cfg.Cols%uint32(parts) != 0 {
			panic("commengine: K chunks must divide the row")
		}
		e.push.sendsLeft += max(1, s.Parts)
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

// bandSegments lists a band's bytes in the matrix as line-sized physical
// segments.
func (e *Engine) bandSegments(band int) []segment {
	n := e.bandBytes()
	return e.spanSegments(e.cfg.VBase+uint64(band*n), n, 0)
}

// spanSegments lists n contiguous bytes at virtual address base as
// line-sized physical segments, at offset into the job's buffer.
func (e *Engine) spanSegments(base uint64, n, offset int) []segment {
	var segs []segment
	for off := 0; off < n; {
		addr := base + uint64(off)
		size := min(n-off, int(addr/64*64+64-addr))
		segs = append(segs, segment{pAddr: e.cfg.VToP(addr), offset: offset + off, size: size})
		off += size
	}
	return segs
}

// pushSends queues the sends whose bands have been read, in list order, a
// split band as its K chunks in order.
func (e *Engine) pushSends(now sim.VTimeInSec) {
	sends := e.cfg.Push.Sends
	for e.push.next < len(sends) {
		s := sends[e.push.next]
		data, ok := e.push.data[s.Band]
		if !ok {
			return
		}
		e.push.next++
		parts := max(1, s.Parts)
		for c := 0; c < parts; c++ {
			msg := e.enqueueSend(now, s.Band, false, s.To, c, e.chunk(data, c, parts),
				func(now sim.VTimeInSec) {
					e.push.sendsLeft--
					e.pushCheckDone(now)
				})
			msg.Parts, msg.record.Parts = parts, parts
		}
	}
}

// chunk cuts K chunk c of parts out of a band's rows.
func (e *Engine) chunk(band []byte, c, parts int) []byte {
	if parts == 1 {
		return band
	}
	row := int(e.cfg.Cols * 4)
	w := row / parts
	out := make([]byte, 0, len(band)/parts)
	for r := 0; r < len(band); r += row {
		out = append(out, band[r+c*w:r+(c+1)*w]...)
	}
	return out
}

// pushRecv writes a received band or chunk into this GPU's copy and raises
// its signal once the writes are acknowledged.
func (e *Engine) pushRecv(msg *TileMsg) {
	band := msg.Group
	segs := e.bandSegments(band)
	name := RowSignal(e.cfg.GPU, band)
	if msg.Parts > 1 {
		base := e.cfg.Push.Chunks[msg.Parts][msg.Part]
		segs = e.spanSegments(base+uint64(band*len(msg.Data)), len(msg.Data), 0)
		name = ChunkSignal(e.cfg.GPU, band, msg.Part)
	}
	e.memJobs = append(e.memJobs, &memJob{
		write: true, segs: segs, data: msg.Data,
		onDone: func(now sim.VTimeInSec, _ []byte) {
			if b := e.cfg.Push.Signals; b != nil {
				b.Raise(now, name, 0)
			}
			e.push.recvsLeft--
			e.pushCheckDone(now)
		},
	})
}

// OnRowTile is the tile tracker's listener for the summed matrix: once
// every tile of a row with a partial sum is complete, the engine reads the
// row and its partials, adds them and writes the sum back.
func (e *Engine) OnRowTile(ev tiletracker.TileEvent) {
	if e.push == nil {
		return
	}
	row := int(ev.Row)
	e.push.rowTiles[row]++
	if e.push.rowTiles[row] != e.cfg.Push.RowTiles {
		return
	}
	for _, sum := range e.cfg.Push.Sums {
		if sum.Row == row {
			e.enqueueSum(ev.Complete, sum)
		}
	}
	e.TickLater(ev.Complete)
}

func (e *Engine) enqueueSum(ready sim.VTimeInSec, sum PushSum) {
	segs := e.spanSegments(sum.Dst, sum.Bytes, 0)
	for i, src := range sum.Srcs {
		segs = append(segs, e.spanSegments(src, sum.Bytes, (i+1)*sum.Bytes)...)
	}
	e.memJobs = append(e.memJobs, &memJob{
		segs: segs, data: make([]byte, sum.Bytes*(1+len(sum.Srcs))),
		onDone: func(now sim.VTimeInSec, data []byte) {
			e.enqueueAdds(sum.Bytes, len(sum.Srcs), func(now sim.VTimeInSec) {
				out := data[:sum.Bytes]
				for i := range sum.Srcs {
					addFloat32s(out, data[(i+1)*sum.Bytes:(i+2)*sum.Bytes])
				}
				e.memJobs = append(e.memJobs, &memJob{
					write: true, segs: e.spanSegments(sum.Dst, sum.Bytes, 0), data: out,
					onDone: func(now sim.VTimeInSec, _ []byte) {
						e.push.sums = append(e.push.sums, SumRecord{
							GPU: e.cfg.GPU, Row: sum.Row, Ready: ready, Done: now})
						e.push.sumsLeft--
						e.pushCheckDone(now)
					},
				})
			})
		},
	})
}

// SumRecords returns the partial sums of the last push.
func (e *Engine) SumRecords() []SumRecord {
	if e.push == nil {
		return nil
	}
	return e.push.sums
}

func (e *Engine) pushCheckDone(now sim.VTimeInSec) {
	if e.finished || e.push.sendsLeft > 0 || e.push.recvsLeft > 0 ||
		e.push.sumsLeft > 0 {
		return
	}
	e.finished = true
	if e.cfg.OnDone != nil {
		e.cfg.OnDone(e.cfg.GPU, now)
	}
}
