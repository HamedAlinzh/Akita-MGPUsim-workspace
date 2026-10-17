// allreducebench runs mccl.AllReduceRing on its own, on the timing R9 Nano
// platform, and splits each All-Reduce's simulated time into where it goes:
// host-to-device copies, kernel-launch commands, GPU kernel execution, and
// remote (inter-GPU) traffic in flight, plus the bytes that crossed the link.
//
// Every (message size, chunk size) pair is one All-Reduce on fresh data,
// verified against the expected mean. Output is one CSV row per pair.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/benchmarks/mccl"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

var numGPUFlag = flag.Int("ring-gpus", 2, "Number of GPUs in the ring.")
var sizesFlag = flag.String("sizes", "1024,16384,65536,262144,1048576",
	"All-Reduce message sizes, in float32 elements.")
var chunksFlag = flag.String("chunks", "65536,262144,1048576",
	"All-Reduce chunk (scratch buffer) sizes, in float32 elements.")
var outFlag = flag.String("out", "allreducebench.csv", "Output CSV path.")

// category accumulates the union busy time of the tasks classified into it,
// plus how many tasks there were and how many payload bytes they carried.
type category struct {
	active int
	since  sim.VTimeInSec
	busy   sim.VTimeInSec
	count  uint64
	bytes  uint64
}

type phaseTracer struct {
	timeTeller sim.TimeTeller
	cats       map[string]*category
	inflight   map[string][]string
}

func newPhaseTracer(tt sim.TimeTeller) *phaseTracer {
	return &phaseTracer{
		timeTeller: tt,
		cats:       make(map[string]*category),
		inflight:   make(map[string][]string),
	}
}

func classify(task tracing.Task) []string {
	switch {
	case task.What == "*driver.MemCopyH2DCommand":
		return []string{"h2d", "driver"}
	case task.What == "*driver.LaunchKernelCommand":
		return []string{"launch", "driver"}
	case task.What == "*protocol.LaunchKernelReq" &&
		strings.HasSuffix(task.Where, "CommandProcessor"):
		return []string{"gpu_kernel", "kernels_on:" + task.Where}
	case task.Kind == "req_in" && strings.HasSuffix(task.Where, ".OutsideIn"):
		if _, ok := task.Detail.(*mem.WriteReq); ok {
			return []string{"remote", "remote_write"}
		}
		return []string{"remote", "remote_read"}
	}
	return nil
}

func payloadBytes(detail interface{}) uint64 {
	switch req := detail.(type) {
	case *mem.WriteReq:
		return uint64(len(req.Data))
	case *mem.ReadReq:
		return req.AccessByteSize
	}
	return 0
}

func (t *phaseTracer) cat(name string) *category {
	c, ok := t.cats[name]
	if !ok {
		c = &category{}
		t.cats[name] = c
	}
	return c
}

func (t *phaseTracer) StartTask(task tracing.Task) {
	names := classify(task)
	if names == nil {
		return
	}

	now := t.timeTeller.CurrentTime()
	for _, n := range names {
		c := t.cat(n)
		if c.active == 0 {
			c.since = now
		}
		c.active++
		c.count++
		c.bytes += payloadBytes(task.Detail)
	}
	t.inflight[task.ID] = names
}

func (t *phaseTracer) StepTask(_ tracing.Task) {}

func (t *phaseTracer) EndTask(task tracing.Task) {
	names, ok := t.inflight[task.ID]
	if !ok {
		return
	}
	delete(t.inflight, task.ID)

	now := t.timeTeller.CurrentTime()
	for _, n := range names {
		c := t.cat(n)
		c.active--
		if c.active == 0 {
			c.busy += now - c.since
		}
	}
}

type snapshot map[string]category

func (t *phaseTracer) snapshot() snapshot {
	s := make(snapshot)
	for n, c := range t.cats {
		s[n] = *c
	}
	return s
}

func (s snapshot) delta(before snapshot, name string) category {
	a, b := s[name], before[name]
	return category{
		busy:  a.busy - b.busy,
		count: a.count - b.count,
		bytes: a.bytes - b.bytes,
	}
}

// gpusRunningKernels counts the GPUs whose Command Processor executed at
// least one kernel between the two snapshots.
func (s snapshot) gpusRunningKernels(before snapshot) int {
	n := 0
	for name := range s {
		if strings.HasPrefix(name, "kernels_on:") &&
			s.delta(before, name).count > 0 {
			n++
		}
	}
	return n
}

func parseInts(s string) []int {
	var out []int
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		v, err := strconv.Atoi(f)
		if err != nil {
			log.Fatalf("bad integer %q", f)
		}
		out = append(out, v)
	}
	return out
}

func main() {
	flag.Parse()
	numGPU := *numGPUFlag
	sizes := parseInts(*sizesFlag)
	chunks := parseInts(*chunksFlag)

	// -h2d-cycles is the runner's flag; this sample builds its own platform,
	// so it reads the value directly.
	h2dCycles, err := strconv.Atoi(flag.Lookup("h2d-cycles").Value.String())
	if err != nil {
		log.Fatal(err)
	}
	platform := runner.MakeR9NanoBuilder().
		WithNumGPU(numGPU).
		WithH2DCycles(h2dCycles).
		Build()
	d := platform.Driver
	engine := platform.Engine

	tracer := newPhaseTracer(engine)
	tracing.CollectTrace(d, tracer)
	for _, gpu := range platform.GPUs {
		tracing.CollectTrace(gpu.CommandProcessor, tracer)
		tracing.CollectTrace(gpu.RDMAEngine, tracer)
	}

	d.Run()
	defer d.Terminate()

	ctx := d.Init()
	gpuIDs := make([]int, numGPU)
	for i := range gpuIDs {
		gpuIDs[i] = i + 1
	}

	maxSize, maxChunk := 0, 0
	for _, s := range sizes {
		if s > maxSize {
			maxSize = s
		}
	}
	for _, c := range chunks {
		if c > maxChunk {
			maxChunk = c
		}
	}

	data := make([]driver.Ptr, numGPU)
	bufs := make([]driver.Ptr, numGPU)
	for g, id := range gpuIDs {
		d.SelectGPU(ctx, id)
		data[g] = d.AllocateMemory(ctx, uint64(maxSize*4))
		bufs[g] = d.AllocateMemory(ctx, uint64(maxChunk*4))
	}
	comms := mccl.CommInitAll(numGPU, d, ctx, gpuIDs)

	// Rerunning with the same -out resumes: verified rows already in the
	// file are kept and their (size, chunk) pairs skipped.
	done := readDone(*outFlag)
	f, err := os.OpenFile(*outFlag, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	if done == nil {
		_ = w.Write([]string{
			"gpus", "size_elems", "chunk_elems", "num_chunks", "ring_steps",
			"kernel_launches", "total", "h2d_busy", "h2d_count", "launch_busy",
			"driver_busy", "gpu_kernel_busy", "remote_busy", "remote_bytes",
			"remote_reqs", "remote_write_bytes", "remote_read_bytes",
			"gpus_running_kernels", "verified",
		})
	}

	for _, size := range sizes {
		for _, chunk := range chunks {
			if prevChunk(chunks, chunk) >= size {
				// A smaller chunk already covers the whole message, and
				// every chunk >= size runs identically (one chunk).
				continue
			}
			if done[[2]int{size, chunk}] {
				log.Printf("gpus=%d size=%d chunk=%d already done", numGPU, size, chunk)
				continue
			}

			host := make([][]float32, numGPU)
			for g, id := range gpuIDs {
				host[g] = make([]float32, size)
				for i := range host[g] {
					host[g][i] = float32(g + 1)
				}
				d.SelectGPU(ctx, id)
				d.MemCopyH2D(ctx, data[g], host[g])
			}

			before := tracer.snapshot()
			start := engine.CurrentTime()
			mccl.AllReduceRing(d, comms, data, size, bufs, chunk)
			total := engine.CurrentTime() - start
			after := tracer.snapshot()

			want := float32(numGPU+1) / 2
			verified := true
			for g, id := range gpuIDs {
				d.SelectGPU(ctx, id)
				d.MemCopyD2H(ctx, host[g], data[g])
				for i, v := range host[g] {
					if v != want {
						log.Printf("gpu %d elem %d: got %v, want %v", id, i, v, want)
						verified = false
						break
					}
				}
			}

			numChunks := (size + chunk - 1) / chunk
			steps := 3 * (numGPU - 1) * numChunks
			h2d := after.delta(before, "h2d")
			launch := after.delta(before, "launch")
			drv := after.delta(before, "driver")
			kern := after.delta(before, "gpu_kernel")
			remote := after.delta(before, "remote")

			row := []string{
				strconv.Itoa(numGPU), strconv.Itoa(size), strconv.Itoa(chunk),
				strconv.Itoa(numChunks), strconv.Itoa(steps),
				strconv.FormatUint(launch.count, 10),
				fmt.Sprintf("%.9f", float64(total)),
				fmt.Sprintf("%.9f", float64(h2d.busy)),
				strconv.FormatUint(h2d.count, 10),
				fmt.Sprintf("%.9f", float64(launch.busy)),
				fmt.Sprintf("%.9f", float64(drv.busy)),
				fmt.Sprintf("%.9f", float64(kern.busy)),
				fmt.Sprintf("%.9f", float64(remote.busy)),
				strconv.FormatUint(remote.bytes, 10),
				strconv.FormatUint(remote.count, 10),
				strconv.FormatUint(after.delta(before, "remote_write").bytes, 10),
				strconv.FormatUint(after.delta(before, "remote_read").bytes, 10),
				strconv.Itoa(after.gpusRunningKernels(before)),
				strconv.FormatBool(verified),
			}
			_ = w.Write(row)
			w.Flush()
			log.Printf("gpus=%d size=%d chunk=%d total=%.1fus verified=%v",
				numGPU, size, chunk, float64(total)*1e6, verified)
		}
	}
}

// readDone returns the (size, chunk) pairs with a verified row in the CSV
// at path, or nil if the file does not exist yet.
func readDone(path string) map[[2]int]bool {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		log.Fatalf("%s: %v", path, err)
	}
	if len(rows) == 0 {
		return nil
	}

	done := make(map[[2]int]bool)
	for _, row := range rows[1:] {
		size, _ := strconv.Atoi(row[1])
		chunk, _ := strconv.Atoi(row[2])
		if row[len(row)-1] == "true" {
			done[[2]int{size, chunk}] = true
		}
	}
	return done
}

// prevChunk returns the largest chunk size in chunks smaller than c, or 0.
func prevChunk(chunks []int, c int) int {
	best := 0
	for _, x := range chunks {
		if x < c && x > best {
			best = x
		}
	}
	return best
}
