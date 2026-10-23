package main

import (
	"testing"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
)

type fakeTime struct{ now sim.VTimeInSec }

func (f *fakeTime) CurrentTime() sim.VTimeInSec { return f.now }

func newTestTimeline(now *fakeTime) *memTimeline {
	return &memTimeline{time: now, bucket: 10e-6,
		counts: map[[2]int]map[string]float64{}, pending: map[string]sim.VTimeInSec{}}
}

func step(what string) tracing.Task {
	return tracing.Task{ID: "t", Steps: []tracing.TaskStep{{What: what}}}
}

// The caches name an MSHR merge "read-mshr-hit"; the timeline once looked for
// "read_mshr_miss" and always wrote 0.
func TestMemTimelineCountsMSHRHits(t *testing.T) {
	now := &fakeTime{now: 1e-6}
	tl := newTestTimeline(now)
	l1 := memTracer{t: tl, gpu: 1, kind: "l1v"}
	l2 := memTracer{t: tl, gpu: 1, kind: "l2"}

	l1.StepTask(step("read-mshr-hit"))
	l2.StepTask(step("read-mshr-hit"))
	l2.StepTask(step("read-mshr-hit"))
	l2.StepTask(step("read-miss"))

	c := tl.counts[[2]int{1, 0}]
	if c["l1v_read_mshr_hit"] != 1 || c["l2_read_mshr_hit"] != 2 || c["l2_read_miss"] != 1 {
		t.Fatalf("counts %v", c)
	}
}

func TestMemTimelineDRAMReadLatency(t *testing.T) {
	now := &fakeTime{}
	tl := newTestTimeline(now)
	dram := memTracer{t: tl, gpu: 2, kind: "dram"}

	req := mem.ReadReqBuilder{}.WithAddress(0x40).WithByteSize(64).Build()
	req.RecvTime = 9.5e-6 // waited 0.5 us in the port before the controller took it
	now.now = 10e-6
	dram.StartTask(tracing.Task{ID: "r@DRAM", Kind: "req_in", Detail: req})
	now.now = 10.25e-6
	dram.EndTask(tracing.Task{ID: "r@DRAM"})
	dram.EndTask(tracing.Task{ID: "w@DRAM"}) // not a pending read

	arrival := tl.counts[[2]int{2, 0}]
	if arrival["dram_reads"] != 1 || arrival["dram_read_ns"] < 749.9 || arrival["dram_read_ns"] > 750.1 {
		t.Fatalf("arrival bucket %v", arrival)
	}
	if tl.counts[[2]int{2, 1}]["dram_read_bytes"] != 64 {
		t.Fatalf("issue bucket %v", tl.counts[[2]int{2, 1}])
	}
	if len(tl.pending) != 0 {
		t.Fatalf("pending %v", tl.pending)
	}
}
