package commengine

import (
	"github.com/sarchlab/akita/v3/sim"
)

// Builder builds communication engines.
type Builder struct {
	engine              sim.Engine
	freq                sim.Freq
	msgOverhead         sim.VTimeInSec
	reduceCyclesPerLine int
	maxInflight         int
}

// MakeBuilder returns a builder with default parameters: 1 GHz, no
// per-message overhead, one cycle to add each 64-byte line, and up to 64
// memory requests in flight.
func MakeBuilder() Builder {
	return Builder{
		freq:                1 * sim.GHz,
		reduceCyclesPerLine: 1,
		maxInflight:         64,
	}
}

// WithEngine sets the simulation engine.
func (b Builder) WithEngine(e sim.Engine) Builder {
	b.engine = e
	return b
}

// WithFreq sets the engine's clock.
func (b Builder) WithFreq(f sim.Freq) Builder {
	b.freq = f
	return b
}

// WithMsgOverhead sets the fixed cost of every message, spent before the
// message is handed to the network.
func (b Builder) WithMsgOverhead(t sim.VTimeInSec) Builder {
	b.msgOverhead = t
	return b
}

// WithReduceCyclesPerLine sets the cycles to add one 64-byte line of one
// partial into the owner's copy.
func (b Builder) WithReduceCyclesPerLine(n int) Builder {
	b.reduceCyclesPerLine = n
	return b
}

// WithMaxInflight sets how many memory requests may be outstanding.
func (b Builder) WithMaxInflight(n int) Builder {
	b.maxInflight = n
	return b
}

// Build creates an engine. Its ToMem port must be connected to the L2 caches
// and SetMemFinder called; its ToFabric port must be on the GPU-to-GPU
// network.
func (b Builder) Build(name string) *Engine {
	e := &Engine{
		msgOverhead:         b.msgOverhead,
		reduceCyclesPerLine: b.reduceCyclesPerLine,
		maxInflight:         b.maxInflight,
	}
	e.TickingComponent = sim.NewTickingComponent(name, b.engine, b.freq, e)
	e.ToMem = sim.NewLimitNumMsgPort(e, b.maxInflight, name+".ToMem")
	e.AddPort("ToMem", e.ToMem)
	e.ToFabric = sim.NewLimitNumMsgPort(e, 16, name+".ToFabric")
	e.AddPort("ToFabric", e.ToFabric)
	return e
}
