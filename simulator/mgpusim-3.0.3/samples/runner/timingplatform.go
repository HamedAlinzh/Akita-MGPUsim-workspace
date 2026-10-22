package runner

import (
	"fmt"
	"log"
	"math"
	"os"

	memtraces "github.com/sarchlab/akita/v3/mem/trace"

	"github.com/sarchlab/akita/v3/analysis"
	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/mem/vm"
	"github.com/sarchlab/akita/v3/mem/vm/mmu"
	"github.com/sarchlab/akita/v3/monitoring"
	"github.com/sarchlab/akita/v3/noc/networking/pcie"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/timing/commengine"
	"github.com/sarchlab/mgpusim/v3/timing/rdma"
)

// R9NanoPlatformBuilder can build a platform that equips R9Nano GPU.
type R9NanoPlatformBuilder struct {
	useParallelEngine                  bool
	debugISA                           bool
	traceVis                           bool
	traceVisStartTime, traceVisEndTime sim.VTimeInSec
	traceMem                           bool
	numGPU                             int
	numSAPerGPU                        int
	numCUPerSA                         int
	useMagicMemoryCopy                 bool
	log2PageSize                       uint64
	l1CacheSize                        uint64
	l1AcquireOnLaunch                  bool
	ldsOccupancy                       bool
	linkBW                             float64 // bytes/s; 0 = shared PCIe
	linkLatencyNS                      float64 // < 0 = shared PCIe
	peerMesh                           bool
	commEngine                         bool
	valuCyclesScale                    float64
	wgCap, wgRamp                      int
	numDispatchers                     int
	orderedDispatch                    bool
	msgOverheadNS                      float64
	h2dCycles                          int
	l2CacheSize                        uint64
	dramSize                           uint64

	engine               sim.Engine
	monitor              *monitoring.Monitor
	perfAnalysisFileName string
	perfAnalyzingPeriod  float64
	perfAnalyzer         *analysis.PerfAnalyzer
	visTracer            tracing.Tracer

	globalStorage *mem.Storage

	gpus []*GPU
}

// MakeR9NanoBuilder creates a EmuBuilder with default parameters.
func MakeR9NanoBuilder() R9NanoPlatformBuilder {
	b := R9NanoPlatformBuilder{
		numGPU:            4,
		numSAPerGPU:       16,
		numCUPerSA:        4,
		log2PageSize:      12,
		l1CacheSize:       16 * mem.KB,
		l1AcquireOnLaunch: true,
		valuCyclesScale:   1,
		h2dCycles:         14500,
		linkLatencyNS:     -1,
		l2CacheSize:       2 * mem.MB,
		dramSize:          4 * mem.GB,
		traceVisStartTime: -1,
		traceVisEndTime:   -1,
	}
	return b
}

// WithL1CacheSize sets the size of each CU's L1 vector (data) cache.
func (b R9NanoPlatformBuilder) WithL1CacheSize(
	size uint64,
) R9NanoPlatformBuilder {
	b.l1CacheSize = size
	return b
}

// WithL1AcquireOnLaunch makes every GPU invalidate its L1 vector caches
// before dispatching each kernel. See R9NanoGPUBuilder.WithL1AcquireOnLaunch.
func (b R9NanoPlatformBuilder) WithL1AcquireOnLaunch(
	on bool,
) R9NanoPlatformBuilder {
	b.l1AcquireOnLaunch = on
	return b
}

// WithH2DCycles sets the fixed latency, in driver cycles, that the driver
// charges for every host-to-device copy. Each kernel launch makes three such
// copies (code object, kernel arguments, dispatch packet).
// WithNumCUPerGPU sets the number of compute units per GPU. A multiple of 4
// is built as shader arrays of 4 CUs, as the default 64 CUs are; any other
// count up to 4 is one shader array.
func (b R9NanoPlatformBuilder) WithNumCUPerGPU(n int) R9NanoPlatformBuilder {
	switch {
	case n > 0 && n%4 == 0:
		b.numSAPerGPU, b.numCUPerSA = n/4, 4
	case n > 0 && n < 4:
		b.numSAPerGPU, b.numCUPerSA = 1, n
	default:
		log.Panicf("CUs per GPU must be 1-3 or a multiple of 4, not %d", n)
	}
	return b
}

// WithLinkBandwidth gives GPU-to-GPU (RDMA) traffic its own network with the
// given bandwidth per direction, in bytes per second, instead of sharing the
// PCIe network with the host. 0 keeps the shared PCIe network.
func (b R9NanoPlatformBuilder) WithLinkBandwidth(bytesPerSec float64) R9NanoPlatformBuilder {
	b.linkBW = bytesPerSec
	return b
}

// WithPeerMesh builds the peer fabric as a full mesh: every pair of GPUs gets
// a dedicated link with the -link-bw bandwidth and -link-latency latency, as
// MI300X's Infinity Fabric gives every GPU pair its own link. Without it, all
// GPUs share one switch through one link each.
func (b R9NanoPlatformBuilder) WithPeerMesh(on bool) R9NanoPlatformBuilder {
	b.peerMesh = on
	return b
}

// WithLinkLatency gives GPU-to-GPU (RDMA) traffic its own network with the
// given one-way latency in ns, the switch latency between two GPUs. A
// negative value keeps the shared PCIe network.
func (b R9NanoPlatformBuilder) WithLinkLatency(ns float64) R9NanoPlatformBuilder {
	b.linkLatencyNS = ns
	return b
}

// WithWGShaping caps work-groups in flight per CU and/or ramps the cap up
// from a start value; see cp.Builder.WithWGShaping.
func (b R9NanoPlatformBuilder) WithWGShaping(cap, ramp int) R9NanoPlatformBuilder {
	b.wgCap, b.wgRamp = cap, ramp
	return b
}

// WithNumDispatchers sets each CP's kernel dispatchers (0 = 8).
func (b R9NanoPlatformBuilder) WithNumDispatchers(n int) R9NanoPlatformBuilder {
	b.numDispatchers = n
	return b
}

// WithOrderedDispatch makes every CP dispatch work-groups in kernel launch
// order.
func (b R9NanoPlatformBuilder) WithOrderedDispatch(on bool) R9NanoPlatformBuilder {
	b.orderedDispatch = on
	return b
}

// WithVALUCyclesScale multiplies the cycles of every vector ALU instruction
// in every CU (see cu.SIMDUnit.CyclesScale).
func (b R9NanoPlatformBuilder) WithVALUCyclesScale(s float64) R9NanoPlatformBuilder {
	b.valuCyclesScale = s
	return b
}

// WithCommEngine gives every GPU a communication engine with the given
// per-message overhead in ns. The engines talk over the peer fabric, which is
// created even if no link knob is set.
func (b R9NanoPlatformBuilder) WithCommEngine(
	on bool, msgOverheadNS float64,
) R9NanoPlatformBuilder {
	b.commEngine = on
	b.msgOverheadNS = msgOverheadNS
	return b
}

// WithLDSOccupancy makes every GPU's dispatcher count dynamic LDS when
// deciding how many work-groups fit on a CU.
func (b R9NanoPlatformBuilder) WithLDSOccupancy(on bool) R9NanoPlatformBuilder {
	b.ldsOccupancy = on
	return b
}

func (b R9NanoPlatformBuilder) WithH2DCycles(cycles int) R9NanoPlatformBuilder {
	b.h2dCycles = cycles
	return b
}

// WithL2CacheSize sets the total L2 cache size per GPU. The size is split
// between memory banks.
func (b R9NanoPlatformBuilder) WithL2CacheSize(
	size uint64,
) R9NanoPlatformBuilder {
	b.l2CacheSize = size
	return b
}

// WithDRAMSize sets the size of DRAM per GPU. Must not exceed 4GB, the
// fixed per-GPU window in the flat global address space.
func (b R9NanoPlatformBuilder) WithDRAMSize(
	size uint64,
) R9NanoPlatformBuilder {
	b.dramSize = size
	return b
}

// WithParallelEngine lets the EmuBuilder to use parallel engine.
func (b R9NanoPlatformBuilder) WithParallelEngine() R9NanoPlatformBuilder {
	b.useParallelEngine = true
	return b
}

// WithISADebugging enables ISA debugging in the simulation.
func (b R9NanoPlatformBuilder) WithISADebugging() R9NanoPlatformBuilder {
	b.debugISA = true
	return b
}

// WithVisTracing lets the platform to record traces for visualization purposes.
func (b R9NanoPlatformBuilder) WithVisTracing() R9NanoPlatformBuilder {
	b.traceVis = true

	return b
}

// WithPartialVisTracing lets the platform to record traces for visualization
// purposes. The trace will only be collected from the start time to the end
// time.
func (b R9NanoPlatformBuilder) WithPartialVisTracing(
	start, end sim.VTimeInSec,
) R9NanoPlatformBuilder {
	b.traceVis = true
	b.traceVisStartTime = start
	b.traceVisEndTime = end

	return b
}

// WithMemTracing lets the platform to trace memory operations.
func (b R9NanoPlatformBuilder) WithMemTracing() R9NanoPlatformBuilder {
	b.traceMem = true
	return b
}

// WithNumGPU sets the number of GPUs to build.
func (b R9NanoPlatformBuilder) WithNumGPU(n int) R9NanoPlatformBuilder {
	b.numGPU = n
	return b
}

// WithLog2PageSize sets the page size as a power of 2.
func (b R9NanoPlatformBuilder) WithLog2PageSize(
	n uint64,
) R9NanoPlatformBuilder {
	b.log2PageSize = n
	return b
}

// WithMonitor sets the monitor that is used to monitor the simulation
func (b R9NanoPlatformBuilder) WithMonitor(
	m *monitoring.Monitor,
) R9NanoPlatformBuilder {
	b.monitor = m
	return b
}

// WithPerfAnalyzer sets the trace that dumps the WithPerfAnalyzer levers.
func (b R9NanoPlatformBuilder) WithPerfAnalyzer(
	traceDirName string,
	tracePeriod float64,
) R9NanoPlatformBuilder {
	b.perfAnalysisFileName = traceDirName
	b.perfAnalyzingPeriod = tracePeriod
	return b
}

// WithMagicMemoryCopy uses global storage as memory components
func (b R9NanoPlatformBuilder) WithMagicMemoryCopy() R9NanoPlatformBuilder {
	b.useMagicMemoryCopy = true
	return b
}

// Build builds a platform with R9Nano GPUs.
func (b R9NanoPlatformBuilder) Build() *Platform {
	b.engine = b.createEngine()
	if b.monitor != nil {
		b.monitor.RegisterEngine(b.engine)
	}

	b.setupPerformanceAnalyzer()
	b.setupVisTracing()

	b.globalStorage = mem.NewStorage(uint64(1+b.numGPU) * 4 * mem.GB)

	mmuComponent, pageTable := b.createMMU(b.engine)

	gpuDriver := b.buildGPUDriver(pageTable)

	gpuBuilder := b.createGPUBuilder(b.engine, gpuDriver, mmuComponent)
	pcieConnector, rootComplexID :=
		b.createConnection(b.engine, gpuDriver, mmuComponent)

	mmuComponent.MigrationServiceProvider = gpuDriver.GetPortByName("MMU")

	rdmaAddressTable := b.createRDMAAddrTable()
	pmcAddressTable := b.createPMCPageTable()

	b.createGPUs(
		rootComplexID, pcieConnector,
		gpuBuilder, gpuDriver,
		rdmaAddressTable, pmcAddressTable)

	pcieConnector.EstablishRoute()
	b.createPeerFabric()

	return &Platform{
		Engine: b.engine,
		Driver: gpuDriver,
		GPUs:   b.gpus,
	}
}

func (b R9NanoPlatformBuilder) buildGPUDriver(
	pageTable vm.PageTable,
) *driver.Driver {
	gpuDriverBuilder := driver.MakeBuilder()
	if b.useMagicMemoryCopy {
		gpuDriverBuilder = gpuDriverBuilder.WithMagicMemoryCopyMiddleware()
	}
	gpuDriver := gpuDriverBuilder.
		WithEngine(b.engine).
		WithPageTable(pageTable).
		WithLog2PageSize(b.log2PageSize).
		WithGlobalStorage(b.globalStorage).
		WithD2HCycles(8500).
		WithH2DCycles(b.h2dCycles).
		Build("Driver")
	if b.visTracer != nil {
		tracing.CollectTrace(gpuDriver, b.visTracer)
	}

	if b.monitor != nil {
		b.monitor.RegisterComponent(gpuDriver)
	}

	if b.perfAnalyzer != nil {
		b.perfAnalyzer.RegisterComponent(gpuDriver)
	}

	// file, err := os.Create("driver_comm.csv")
	// if err != nil {
	// 	panic(err)
	// }
	// gpuDriver.GetPortByName("GPU").AcceptHook(
	// 	sim.NewPortMsgLogger(log.New(file, "", 0)))

	return gpuDriver
}

func (b *R9NanoPlatformBuilder) setupVisTracing() {
	if !b.traceVis {
		return
	}

	var backend tracing.TracerBackend
	switch *visTracerDB {
	case "sqlite":
		be := tracing.NewSQLiteTraceWriter(*visTracerDBFileName)
		be.Init()
		backend = be
	case "csv":
		be := tracing.NewCSVTraceWriter(*visTracerDBFileName)
		be.Init()
		backend = be
	case "mysql":
		be := tracing.NewMySQLTraceWriter()
		be.Init()
		backend = be
	default:
		panic(fmt.Sprintf(
			"Tracer database type must be [sqlite|csv|mysql]. "+
				"Provided value %s is not supported.",
			*visTracerDB))
	}

	visTracer := tracing.NewDBTracer(b.engine, backend)
	visTracer.SetTimeRange(b.traceVisStartTime, b.traceVisEndTime)

	b.visTracer = visTracer
}

func (b *R9NanoPlatformBuilder) setupPerformanceAnalyzer() {
	if b.perfAnalysisFileName != "" {
		b.perfAnalyzer = analysis.MakePerfAnalyzerBuilder().
			WithPeriod(sim.VTimeInSec(b.perfAnalyzingPeriod)).
			WithDBFilename(b.perfAnalysisFileName).
			WithEngine(b.engine).
			Build()
	}
}

func (b *R9NanoPlatformBuilder) createGPUs(
	rootComplexID int,
	pcieConnector *pcie.Connector,
	gpuBuilder R9NanoGPUBuilder,
	gpuDriver *driver.Driver,
	rdmaAddressTable *mem.BankedLowModuleFinder,
	pmcAddressTable *mem.BankedLowModuleFinder,
) {
	lastSwitchID := rootComplexID
	for i := 1; i < b.numGPU+1; i++ {
		if i%2 == 1 {
			lastSwitchID = pcieConnector.AddSwitch(rootComplexID)
		}

		b.createGPU(i, gpuBuilder, gpuDriver,
			rdmaAddressTable, pmcAddressTable,
			pcieConnector, lastSwitchID)
	}
}

func (b R9NanoPlatformBuilder) createPMCPageTable() *mem.BankedLowModuleFinder {
	pmcAddressTable := new(mem.BankedLowModuleFinder)
	pmcAddressTable.BankSize = 4 * mem.GB
	pmcAddressTable.LowModules = append(pmcAddressTable.LowModules, nil)
	return pmcAddressTable
}

func (b R9NanoPlatformBuilder) createRDMAAddrTable() *mem.BankedLowModuleFinder {
	rdmaAddressTable := new(mem.BankedLowModuleFinder)
	rdmaAddressTable.BankSize = 4 * mem.GB
	rdmaAddressTable.LowModules = append(rdmaAddressTable.LowModules, nil)
	return rdmaAddressTable
}

func (b R9NanoPlatformBuilder) createConnection(
	engine sim.Engine,
	gpuDriver *driver.Driver,
	mmuComponent *mmu.MMU,
) (*pcie.Connector, int) {
	//connection := sim.NewDirectConnection(engine)
	// connection := noc.NewFixedBandwidthConnection(32, engine, 1*sim.GHz)
	// connection.SrcBufferCapacity = 40960000
	pcieConnector := pcie.NewConnector().
		WithEngine(engine).
		WithVersion(4, 16).
		WithSwitchLatency(140)

	if b.visTracer != nil {
		pcieConnector = pcieConnector.WithVisTracer(b.visTracer)
	}

	pcieConnector.CreateNetwork("PCIe")
	rootComplexID := pcieConnector.AddRootComplex(
		[]sim.Port{
			gpuDriver.GetPortByName("GPU"),
			gpuDriver.GetPortByName("MMU"),
			mmuComponent.GetPortByName("Migration"),
			mmuComponent.GetPortByName("Top"),
		})
	return pcieConnector, rootComplexID
}

func (b R9NanoPlatformBuilder) createEngine() sim.Engine {
	var engine sim.Engine

	if b.useParallelEngine {
		engine = sim.NewParallelEngine()
	} else {
		engine = sim.NewSerialEngine()
	}
	// engine.AcceptHook(sim.NewEventLogger(log.New(os.Stdout, "", 0)))

	return engine
}

func (b R9NanoPlatformBuilder) createMMU(
	engine sim.Engine,
) (*mmu.MMU, vm.PageTable) {
	pageTable := vm.NewPageTable(b.log2PageSize)
	mmuBuilder := mmu.MakeBuilder().
		WithEngine(engine).
		WithFreq(1 * sim.GHz).
		WithPageWalkingLatency(100).
		WithLog2PageSize(b.log2PageSize).
		WithPageTable(pageTable)

	mmuComponent := mmuBuilder.Build("MMU")

	if b.monitor != nil {
		b.monitor.RegisterComponent(mmuComponent)
	}

	return mmuComponent, pageTable
}

func (b *R9NanoPlatformBuilder) createGPUBuilder(
	engine sim.Engine,
	gpuDriver *driver.Driver,
	mmuComponent *mmu.MMU,
) R9NanoGPUBuilder {
	gpuBuilder := MakeR9NanoGPUBuilder().
		WithEngine(engine).
		WithMMU(mmuComponent).
		WithNumCUPerShaderArray(b.numCUPerSA).
		WithNumShaderArray(b.numSAPerGPU).
		WithNumMemoryBank(16).
		WithLog2MemoryBankInterleavingSize(7).
		WithLog2PageSize(b.log2PageSize).
		WithGlobalStorage(b.globalStorage).
		WithL1CacheSize(b.l1CacheSize).
		WithL1AcquireOnLaunch(b.l1AcquireOnLaunch).
		WithLDSOccupancy(b.ldsOccupancy).
		WithCommEngine(b.commEngine, sim.VTimeInSec(b.msgOverheadNS*1e-9)).
		WithVALUCyclesScale(b.valuCyclesScale).
		WithWGShaping(b.wgCap, b.wgRamp).
		WithNumDispatchers(b.numDispatchers).
		WithOrderedDispatch(b.orderedDispatch).
		WithL2CacheSize(b.l2CacheSize).
		WithDRAMSize(b.dramSize)

	if b.monitor != nil {
		gpuBuilder = gpuBuilder.WithMonitor(b.monitor)
	}

	if b.perfAnalyzer != nil {
		gpuBuilder = gpuBuilder.WithPerfAnalyzer(b.perfAnalyzer)
	}

	if b.visTracer != nil {
		gpuBuilder = gpuBuilder.WithVisTracer(b.visTracer)
	}

	gpuBuilder = b.setMemTracer(gpuBuilder)
	gpuBuilder = b.setISADebugger(gpuBuilder)

	return gpuBuilder
}

func (b *R9NanoPlatformBuilder) setISADebugger(
	gpuBuilder R9NanoGPUBuilder,
) R9NanoGPUBuilder {
	if !b.debugISA {
		return gpuBuilder
	}

	gpuBuilder = gpuBuilder.WithISADebugging()
	return gpuBuilder
}

func (b *R9NanoPlatformBuilder) setMemTracer(
	gpuBuilder R9NanoGPUBuilder,
) R9NanoGPUBuilder {
	if !b.traceMem {
		return gpuBuilder
	}

	file, err := os.Create("mem.trace")
	if err != nil {
		panic(err)
	}
	logger := log.New(file, "", 0)
	memTracer := memtraces.NewTracer(logger, b.engine)
	gpuBuilder = gpuBuilder.WithMemTracer(memTracer)
	return gpuBuilder
}

func (b *R9NanoPlatformBuilder) createGPU(
	index int,
	gpuBuilder R9NanoGPUBuilder,
	gpuDriver *driver.Driver,
	rdmaAddressTable *mem.BankedLowModuleFinder,
	pmcAddressTable *mem.BankedLowModuleFinder,
	pcieConnector *pcie.Connector,
	pcieSwitchID int,
) *GPU {
	name := fmt.Sprintf("GPU[%d]", index)
	memAddrOffset := uint64(index) * 4 * mem.GB
	gpu := gpuBuilder.
		WithMemAddrOffset(memAddrOffset).
		Build(name, uint64(index))
	gpuDriver.RegisterGPU(
		gpu.Domain.GetPortByName("CommandProcessor"),
		driver.DeviceProperties{
			CUCount:  b.numCUPerSA * b.numSAPerGPU,
			DRAMSize: b.dramSize,
		},
	)
	gpu.CommandProcessor.Driver = gpuDriver.GetPortByName("GPU")

	b.configRDMAEngine(gpu, rdmaAddressTable)
	b.configPMC(gpu, gpuDriver, pmcAddressTable)

	ports := gpu.Domain.Ports()
	if b.usePeerFabric() {
		ports = withoutPort(ports, gpu.RDMAEngine.ToOutside)
	}
	pcieConnector.PlugInDevice(pcieSwitchID, ports)

	b.gpus = append(b.gpus, gpu)

	return gpu
}

func (b *R9NanoPlatformBuilder) configRDMAEngine(
	gpu *GPU,
	addrTable *mem.BankedLowModuleFinder,
) {
	gpu.RDMAEngine.RemoteRDMAAddressTable = addrTable
	addrTable.LowModules = append(
		addrTable.LowModules,
		gpu.RDMAEngine.ToOutside)
}

func (b *R9NanoPlatformBuilder) configPMC(
	gpu *GPU,
	gpuDriver *driver.Driver,
	addrTable *mem.BankedLowModuleFinder,
) {
	gpu.PMC.RemotePMCAddressTable = addrTable
	addrTable.LowModules = append(
		addrTable.LowModules,
		gpu.PMC.GetPortByName("Remote"))
	gpuDriver.RemotePMCPorts = append(
		gpuDriver.RemotePMCPorts, gpu.PMC.GetPortByName("Remote"))
}

func (b *R9NanoPlatformBuilder) usePeerFabric() bool {
	return b.linkBW > 0 || b.linkLatencyNS >= 0 || b.commEngine
}

func withoutPort(ports []sim.Port, drop sim.Port) []sim.Port {
	var out []sim.Port
	for _, p := range ports {
		if p != drop {
			out = append(out, p)
		}
	}
	return out
}

// peerFabricFlitBytes is the flit size of the GPU-to-GPU fabric. Akita moves
// one flit per network cycle, so the fabric's clock is the bandwidth divided
// by this, which realizes any bandwidth exactly.
const peerFabricFlitBytes = 16

// createPeerFabric connects every GPU's RDMA engine to one switch of a
// separate network, so peer traffic has its own bandwidth and latency and
// no longer shares links with host traffic. Two GPUs are then one switch
// apart, and a flit crosses one switch input pipeline of the latency.
func (b *R9NanoPlatformBuilder) createPeerFabric() {
	if !b.usePeerFabric() {
		return
	}

	bw := b.linkBW
	freq := 1 * sim.GHz
	if bw > 0 {
		freq = sim.Freq(bw / peerFabricFlitBytes)
	}
	latencyNS := b.linkLatencyNS
	if latencyNS < 0 {
		latencyNS = 140 // the PCIe switch latency, 140 cycles at 1 GHz
	}
	cycles := max(1, int(math.Round(latencyNS*float64(freq)/1e9)))

	// Without a bandwidth the fabric keeps PCIe 4.0 x16's: 34 B flits at
	// 1 GHz (32 GiB/s rounded to whole bytes per cycle).
	flit, realized := 34, 34e9
	if bw > 0 {
		flit, realized = peerFabricFlitBytes, float64(freq)*peerFabricFlitBytes
	}
	topology := "one shared switch"
	if b.peerMesh {
		b.createPeerMesh(bw, freq, cycles)
		topology = "full mesh, one link per GPU pair"
	} else {
		b.createPeerSwitch(bw, freq, cycles)
	}

	log.Printf("peer fabric (%s): clock %.3f MHz, flit %d B, %.3f GB/s per "+
		"direction per link, latency %d cycles = %.1f ns", topology,
		float64(freq)/1e6, flit, realized/1e9,
		cycles, float64(cycles)/float64(freq)*1e9)
}

func (b *R9NanoPlatformBuilder) peerPorts(gpu *GPU) []sim.Port {
	ports := []sim.Port{gpu.RDMAEngine.ToOutside}
	if gpu.CommEngine != nil {
		ports = append(ports, gpu.CommEngine.ToFabric)
	}
	return ports
}

// createPeerMesh gives every pair of GPUs a dedicated link: a two-GPU
// network of the same kind as the shared switch (one switch, 16 B flits,
// the -link-bw bandwidth, the -link-latency switch latency), carrying that
// pair's RDMA and engine traffic. Each GPU gets one RDMA port and one engine
// port per peer, so its links run in parallel, as MI300X's Infinity Fabric
// links do.
func (b *R9NanoPlatformBuilder) createPeerMesh(bw float64, freq sim.Freq, cycles int) {
	for i, gi := range b.gpus {
		for j := i + 1; j < len(b.gpus); j++ {
			gj := b.gpus[j]
			ri, rj := b.rdmaLink(gi, gj, j+1), b.rdmaLink(gj, gi, i+1)
			ei, ej := []sim.Port{ri}, []sim.Port{rj}
			if gi.CommEngine != nil && gj.CommEngine != nil {
				ei = append(ei, engineLink(gi, gj, j+1))
				ej = append(ej, engineLink(gj, gi, i+1))
			}

			link := pcie.NewConnector().
				WithEngine(b.engine).
				WithFrequency(freq).
				WithSwitchLatency(cycles)
			if bw > 0 {
				link = link.WithBandwidth(uint64(math.Round(bw)))
			}
			link.CreateNetwork(fmt.Sprintf("PeerLink%dto%d", i+1, j+1))
			sw := link.AddRootComplex(ei)
			link.PlugInDevice(sw, ej)
			link.EstablishRoute()
		}
	}
}

// rdmaLink creates from's RDMA port toward to, and routes from's requests for
// to through it.
func (b *R9NanoPlatformBuilder) rdmaLink(from, to *GPU, toID int) sim.Port {
	r := from.RDMAEngine
	port := sim.NewLimitNumMsgPort(r, 64,
		fmt.Sprintf("%s.ToGPU[%d]", r.Name(), toID))
	if r.PeerRoutes == nil {
		r.PeerRoutes = map[sim.Port]rdma.PeerRoute{}
	}
	route := r.PeerRoutes[to.RDMAEngine.ToOutside]
	route.Local = port
	r.PeerRoutes[to.RDMAEngine.ToOutside] = route
	// The remote end is filled in when to's port toward from is created.
	if back, ok := to.RDMAEngine.PeerRoutes[r.ToOutside]; ok {
		back.Remote = port
		to.RDMAEngine.PeerRoutes[r.ToOutside] = back
		route.Remote = back.Local
		r.PeerRoutes[to.RDMAEngine.ToOutside] = route
	}
	return port
}

func engineLink(from, to *GPU, toID int) sim.Port {
	e := from.CommEngine
	port := sim.NewLimitNumMsgPort(e, 16,
		fmt.Sprintf("%s.ToGPU[%d]", e.Name(), toID))
	if e.Links == nil {
		e.Links = map[*commengine.Engine]sim.Port{}
	}
	e.Links[to.CommEngine] = port
	return port
}

// createPeerSwitch puts every GPU on one switch, one link each.
func (b *R9NanoPlatformBuilder) createPeerSwitch(bw float64, freq sim.Freq, cycles int) {
	fabric := pcie.NewConnector().
		WithEngine(b.engine).
		WithFrequency(freq).
		WithSwitchLatency(cycles)
	if bw > 0 {
		fabric = fabric.WithBandwidth(uint64(math.Round(bw)))
	}
	fabric.CreateNetwork("PeerFabric")

	switchID := -1
	for _, gpu := range b.gpus {
		port := b.peerPorts(gpu)
		if switchID < 0 {
			switchID = fabric.AddRootComplex(port)
			continue
		}
		fabric.PlugInDevice(switchID, port)
	}
	fabric.EstablishRoute()
}
