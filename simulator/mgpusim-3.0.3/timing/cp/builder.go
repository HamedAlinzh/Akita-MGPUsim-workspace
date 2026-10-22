package cp

import (
	"fmt"
	"math"

	"github.com/sarchlab/akita/v3/analysis"
	"github.com/sarchlab/akita/v3/monitoring"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/protocol"
	"github.com/sarchlab/mgpusim/v3/timing/cp/internal/dispatching"
	"github.com/sarchlab/mgpusim/v3/timing/cp/internal/resource"
)

// Builder can build Command Processors
type Builder struct {
	freq           sim.Freq
	engine         sim.Engine
	visTracer      tracing.Tracer
	monitor        *monitoring.Monitor
	perfAnalyzer   *analysis.PerfAnalyzer
	numDispatchers int
	ldsOccupancy   bool
	wgCap, wgRamp  int
}

// MakeBuilder creates a new builder with default configuration values.
func MakeBuilder() Builder {
	b := Builder{
		freq:           1 * sim.GHz,
		numDispatchers: 8,
	}
	return b
}

// WithVisTracer enables tracing for visualization on the command processor and
// the dispatchers.
func (b Builder) WithVisTracer(tracer tracing.Tracer) Builder {
	b.visTracer = tracer
	return b
}

// WithLDSOccupancy makes the dispatchers count each work-group's dynamic LDS
// (the kernel's local pointer arguments) when deciding how many work-groups a
// CU can hold. Off by default: then only the code object's static LDS counts.
func (b Builder) WithLDSOccupancy(on bool) Builder {
	b.ldsOccupancy = on
	return b
}

// WithWGShaping caps the work-groups in flight per CU (cap) and/or starts the
// cap at ramp and raises it by one per completed work-group. 0 turns each
// off.
func (b Builder) WithWGShaping(cap, ramp int) Builder {
	b.wgCap, b.wgRamp = cap, ramp
	return b
}

// WithNumDispatchers sets how many kernels the CP dispatches at once (0
// keeps the default, 8). Dispatchers serve CUs in index order, and a new
// kernel takes the lowest free one.
func (b Builder) WithNumDispatchers(n int) Builder {
	if n > 0 {
		b.numDispatchers = n
	}
	return b
}

// WithEngine sets the even-driven simulation engine to use.
func (b Builder) WithEngine(engine sim.Engine) Builder {
	b.engine = engine
	return b
}

// WithFreq sets the frequency that the Command Processor works at.
func (b Builder) WithFreq(freq sim.Freq) Builder {
	b.freq = freq
	return b
}

// WithMonitor sets the monitor used to show progress bars.
func (b Builder) WithMonitor(monitor *monitoring.Monitor) Builder {
	b.monitor = monitor
	return b
}

// WithPerfAnalyzer sets the buffer analyzer used to analyze the
// command processor's buffers.
func (b Builder) WithPerfAnalyzer(
	analyzer *analysis.PerfAnalyzer,
) Builder {
	b.perfAnalyzer = analyzer
	return b
}

// Build builds a new Command Processor
func (b Builder) Build(name string) *CommandProcessor {
	cp := new(CommandProcessor)
	cp.TickingComponent = sim.NewTickingComponent(name, b.engine, b.freq, cp)

	b.createPorts(cp, name)

	cp.bottomKernelLaunchReqIDToTopReqMap =
		make(map[string]*protocol.LaunchKernelReq)
	cp.bottomMemCopyH2DReqIDToTopReqMap =
		make(map[string]*protocol.MemCopyH2DReq)
	cp.bottomMemCopyD2HReqIDToTopReqMap =
		make(map[string]*protocol.MemCopyD2HReq)

	b.buildDispatchers(cp)

	if b.visTracer != nil {
		tracing.CollectTrace(cp, b.visTracer)
	}

	if b.perfAnalyzer != nil {
		b.perfAnalyzer.RegisterComponent(cp)
	}

	return cp
}

func (Builder) createPorts(cp *CommandProcessor, name string) {
	unlimited := math.MaxInt32
	cp.ToDriver = sim.NewLimitNumMsgPort(cp, 1, name+".ToDriver")
	cp.toDriverSender = sim.NewBufferedSender(
		cp.ToDriver,
		sim.NewBuffer(cp.Name()+".ToDriverSenderBuffer", unlimited),
	)
	cp.ToDMA = sim.NewLimitNumMsgPort(cp, 1, name+".ToDispatcher")
	cp.toDMASender = sim.NewBufferedSender(
		cp.ToDMA,
		sim.NewBuffer(cp.Name()+".ToDMASenderBuffer", unlimited),
	)
	cp.ToCUs = sim.NewLimitNumMsgPort(cp, 1, name+".ToCUs")
	cp.toCUsSender = sim.NewBufferedSender(
		cp.ToCUs,
		sim.NewBuffer(cp.Name()+".ToCUSenderBuffer", unlimited),
	)
	cp.ToTLBs = sim.NewLimitNumMsgPort(cp, 1, name+".ToTLBs")
	cp.toTLBsSender = sim.NewBufferedSender(
		cp.ToTLBs,
		sim.NewBuffer(cp.Name()+".ToTLBSenderBuffer", unlimited),
	)
	cp.ToRDMA = sim.NewLimitNumMsgPort(cp, 1, name+".ToRDMA")
	cp.toRDMASender = sim.NewBufferedSender(
		cp.ToRDMA,
		sim.NewBuffer(cp.Name()+".ToRDMASenderBuffer", unlimited),
	)
	cp.ToPMC = sim.NewLimitNumMsgPort(cp, 1, name+".ToPMC")
	cp.toPMCSender = sim.NewBufferedSender(
		cp.ToPMC,
		sim.NewBuffer(cp.Name()+".ToPMCSenderBuffer", unlimited),
	)
	cp.ToAddressTranslators = sim.NewLimitNumMsgPort(cp, 1,
		name+".ToAddressTranslators")
	cp.toAddressTranslatorsSender = sim.NewBufferedSender(
		cp.ToAddressTranslators,
		sim.NewBuffer(cp.Name()+".ToAddressTranslatorsBuffer", unlimited),
	)
	cp.ToCaches = sim.NewLimitNumMsgPort(cp, 1, name+".ToCaches")
	cp.toCachesSender = sim.NewBufferedSender(
		cp.ToCaches,
		sim.NewBuffer(cp.Name()+".ToCachesBuffer", unlimited),
	)
}

func (b *Builder) buildDispatchers(cp *CommandProcessor) {
	cuResourcePool := resource.NewCUResourcePool()
	cuResourcePool.CountDynamicLDS = b.ldsOccupancy
	cuResourcePool.WGCap = b.wgCap
	cuResourcePool.WGRampStart = b.wgRamp
	cp.cuPool = cuResourcePool
	builder := dispatching.MakeBuilder().
		WithCP(cp).
		WithAlg("round-robin").
		WithCUResourcePool(cuResourcePool).
		WithDispatchingPort(cp.ToCUs).
		WithRespondingPort(cp.ToDriver).
		WithMonitor(b.monitor)

	for i := 0; i < b.numDispatchers; i++ {
		disp := builder.Build(fmt.Sprintf("%s.Dispatcher%d", cp.Name(), i))
		if d, ok := disp.(interface {
			SetOnKernelDone(func(sim.VTimeInSec, *protocol.LaunchKernelReq))
		}); ok {
			d.SetOnKernelDone(cp.kernelDone)
		}

		if b.visTracer != nil {
			tracing.CollectTrace(disp, b.visTracer)
		}

		cp.Dispatchers = append(cp.Dispatchers, disp)
	}
}
