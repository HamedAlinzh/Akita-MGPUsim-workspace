package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"sync"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/protocol"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

// wgTracer records when each work-group is dispatched to a CU and when the
// CP learns it completed, on every GPU, for -wg-trace.
type wgTracer struct {
	time sim.TimeTeller

	mu   sync.Mutex
	open map[string]*wgRecord
	all  []*wgRecord
}

type wgRecord struct {
	gpu        int
	kernel     string // the launch request's task ID at the CP
	gridY      uint32
	x, y       int
	cu         string
	start, end sim.VTimeInSec
}

type gpuTracer struct {
	t   *wgTracer
	gpu int
}

func (g gpuTracer) StartTask(task tracing.Task) {
	req, ok := task.Detail.(*protocol.MapWGReq)
	if !ok || task.Kind != "req_out" {
		return
	}
	g.t.mu.Lock()
	defer g.t.mu.Unlock()
	wg := req.WorkGroup
	r := &wgRecord{gpu: g.gpu, kernel: task.ParentID, gridY: wg.Packet.GridSizeY,
		x: wg.IDX, y: wg.IDY, cu: req.Dst.Name(), start: g.t.time.CurrentTime()}
	g.t.open[task.ID] = r
	g.t.all = append(g.t.all, r)
}

func (g gpuTracer) StepTask(tracing.Task) {}

func (g gpuTracer) EndTask(task tracing.Task) {
	g.t.mu.Lock()
	defer g.t.mu.Unlock()
	if r, ok := g.t.open[task.ID]; ok {
		r.end = g.t.time.CurrentTime()
		delete(g.t.open, task.ID)
	}
}

// traceWGs attaches a tracer to every dispatcher of every GPU.
func traceWGs(p *runner.Platform) *wgTracer {
	t := &wgTracer{time: p.Engine, open: map[string]*wgRecord{}}
	for i, gpu := range p.GPUs {
		for _, d := range gpu.CommandProcessor.Dispatchers {
			tracing.CollectTrace(d, gpuTracer{t: t, gpu: i + 1})
		}
	}
	return t
}

// write saves the records as CSV. Kernels are numbered per GPU in the order
// their first work-group was dispatched.
func (t *wgTracer) write(path string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	sort.SliceStable(t.all, func(i, j int) bool { return t.all[i].start < t.all[j].start })
	seq := map[string]int{}
	next := map[int]int{}
	f, err := os.Create(path)
	if err != nil {
		log.Panic(err)
	}
	defer f.Close()
	fmt.Fprintln(f, "gpu,kernel_seq,kernel_id,grid_y,wg_x,wg_y,cu,start_us,end_us")
	for _, r := range t.all {
		if _, ok := seq[r.kernel]; !ok {
			seq[r.kernel] = next[r.gpu]
			next[r.gpu]++
		}
		fmt.Fprintf(f, "%d,%d,%s,%d,%d,%d,%s,%.4f,%.4f\n", r.gpu, seq[r.kernel],
			r.kernel, r.gridY, r.x, r.y, r.cu, float64(r.start)*1e6, float64(r.end)*1e6)
	}
}
