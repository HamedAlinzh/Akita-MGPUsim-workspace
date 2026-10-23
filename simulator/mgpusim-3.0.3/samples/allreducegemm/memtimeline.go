package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

// memTimeline counts, per GPU and per time bucket, cache hits, misses and
// MSHR hits (L1V and L2), TLB misses (L1V TLBs and the L2 TLB), DRAM bytes,
// and DRAM reads with their total latency (from arrival at the controller's
// port to completion, in the bucket of the arrival), for -mem-timeline.
type memTimeline struct {
	time   sim.TimeTeller
	bucket float64 // seconds

	mu      sync.Mutex
	counts  map[[2]int]map[string]float64 // [gpu, bucket] -> column -> count
	pending map[string]sim.VTimeInSec     // DRAM read task ID -> arrival
}

var memColumns = []string{
	"l1v_read_hit", "l1v_read_miss", "l1v_read_mshr_hit", "l2_read_hit",
	"l2_read_miss", "l2_read_mshr_hit", "l2_write_hit", "l2_write_miss",
	"l1v_tlb_miss", "l2_tlb_miss", "dram_read_bytes", "dram_write_bytes",
	"dram_reads", "dram_read_ns",
}

type memTracer struct {
	t    *memTimeline
	gpu  int
	kind string // "l1v", "l2", "l1v_tlb", "l2_tlb" or "dram"
}

func (m memTracer) add(column string, v float64) {
	m.t.mu.Lock()
	defer m.t.mu.Unlock()
	m.addAt(m.t.time.CurrentTime(), column, v)
}

// addAt adds to the bucket holding time t; the caller holds the lock.
func (m memTracer) addAt(t sim.VTimeInSec, column string, v float64) {
	b := int(float64(t) / m.t.bucket)
	key := [2]int{m.gpu, b}
	if m.t.counts[key] == nil {
		m.t.counts[key] = map[string]float64{}
	}
	m.t.counts[key][column] += v
}

func (m memTracer) StartTask(task tracing.Task) {
	if m.kind != "dram" || task.Kind != "req_in" {
		return
	}
	switch req := task.Detail.(type) {
	case *mem.ReadReq:
		m.add("dram_read_bytes", float64(req.AccessByteSize))
		m.t.mu.Lock()
		m.t.pending[task.ID] = req.RecvTime
		m.t.mu.Unlock()
	case *mem.WriteReq:
		m.add("dram_write_bytes", float64(len(req.Data)))
	}
}

func (m memTracer) StepTask(task tracing.Task) {
	for _, s := range task.Steps {
		what := strings.ReplaceAll(s.What, "-", "_")
		switch m.kind {
		case "l1v", "l2":
			if what == "read_hit" || what == "read_miss" ||
				what == "read_mshr_hit" ||
				m.kind == "l2" && (what == "write_hit" || what == "write_miss") {
				m.add(m.kind+"_"+what, 1)
			}
		case "l1v_tlb", "l2_tlb":
			if what == "miss" {
				m.add(m.kind+"_miss", 1)
			}
		}
	}
}

func (m memTracer) EndTask(task tracing.Task) {
	if m.kind != "dram" {
		return
	}
	m.t.mu.Lock()
	defer m.t.mu.Unlock()
	arrival, ok := m.t.pending[task.ID]
	if !ok {
		return
	}
	delete(m.t.pending, task.ID)
	m.addAt(arrival, "dram_reads", 1)
	m.addAt(arrival, "dram_read_ns", float64(m.t.time.CurrentTime()-arrival)*1e9)
}

func traceMemory(p *runner.Platform, bucketUS float64) *memTimeline {
	t := &memTimeline{time: p.Engine, bucket: bucketUS * 1e-6,
		counts: map[[2]int]map[string]float64{}, pending: map[string]sim.VTimeInSec{}}
	for i, gpu := range p.GPUs {
		attach := func(kind string, comps []runner.TraceableComponent) {
			for _, c := range comps {
				tracing.CollectTrace(c, memTracer{t: t, gpu: i + 1, kind: kind})
			}
		}
		attach("l1v", gpu.L1VCaches)
		attach("l2", gpu.L2Caches)
		attach("l1v_tlb", gpu.L1VTLBs)
		attach("l2_tlb", gpu.L2TLBs)
		attach("dram", gpu.MemControllers)
	}
	return t
}

func (t *memTimeline) write(path string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	keys := make([][2]int, 0, len(t.counts))
	for k := range t.counts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i][0] < keys[j][0] || keys[i][0] == keys[j][0] && keys[i][1] < keys[j][1]
	})
	f, err := os.Create(path)
	if err != nil {
		log.Panic(err)
	}
	defer f.Close()
	fmt.Fprintf(f, "gpu,start_us,%s\n", strings.Join(memColumns, ","))
	for _, k := range keys {
		fmt.Fprintf(f, "%d,%.1f", k[0], float64(k[1])*t.bucket*1e6)
		for _, c := range memColumns {
			fmt.Fprintf(f, ",%g", t.counts[k][c])
		}
		fmt.Fprintln(f)
	}
}
