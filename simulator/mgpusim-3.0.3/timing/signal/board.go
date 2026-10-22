// Package signal provides named device-side signals shared by every GPU's
// command processor, for kernels that wait on other kernels or on a tile
// tracker (insts.HsaCo.WaitSignals and Signal) instead of on the host.
package signal

import (
	"sort"
	"sync"

	"github.com/sarchlab/akita/v3/sim"
)

// Waker is woken when a signal is raised, such as a command processor
// holding kernels that wait for it.
type Waker interface {
	TickLater(now sim.VTimeInSec)
}

// Record is one held kernel's timeline, by the signal it raises.
type Record struct {
	Name                         string
	Held, Released, Done, Raised sim.VTimeInSec
}

// Board holds the raised signals.
type Board struct {
	mu      sync.Mutex
	engine  sim.Engine
	raised  map[string]sim.VTimeInSec
	wakers  []Waker
	records map[string]*Record

	numHeld   int
	heldWant  int
	heldReady chan struct{}
}

// NewBoard creates an empty board that schedules on the engine.
func NewBoard(engine sim.Engine) *Board {
	return &Board{engine: engine, raised: map[string]sim.VTimeInSec{},
		records: map[string]*Record{}}
}

// AddWaker registers a component woken whenever a signal is raised.
func (b *Board) AddWaker(w Waker) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.wakers = append(b.wakers, w)
}

type raiseEvent struct {
	*sim.EventBase
	name string
}

// Raise raises the signal delay seconds after now.
func (b *Board) Raise(now sim.VTimeInSec, name string, delay float64) {
	e := &raiseEvent{sim.NewEventBase(now+sim.VTimeInSec(delay), b), name}
	b.engine.Schedule(e)
}

// Handle raises a signal and wakes the wakers.
func (b *Board) Handle(e sim.Event) error {
	ev := e.(*raiseEvent)
	now := ev.Time()
	b.mu.Lock()
	if _, ok := b.raised[ev.name]; !ok {
		b.raised[ev.name] = now
	}
	if r, ok := b.records[ev.name]; ok {
		r.Raised = now
	}
	wakers := append([]Waker(nil), b.wakers...)
	b.mu.Unlock()
	for _, w := range wakers {
		w.TickLater(now)
	}
	return nil
}

// AllRaised tells whether every named signal is raised.
func (b *Board) AllRaised(names []string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, n := range names {
		if _, ok := b.raised[n]; !ok {
			return false
		}
	}
	return true
}

// RaisedAt returns when a signal was raised.
func (b *Board) RaisedAt(name string) (sim.VTimeInSec, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	t, ok := b.raised[name]
	return t, ok
}

// ExpectHeld returns a channel closed once n kernels (counted from now) are
// held by the command processors, so the host can start timing after
// every pre-enqueued kernel has reached its GPU.
func (b *Board) ExpectHeld(n int) <-chan struct{} {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.numHeld, b.heldWant = 0, n
	b.heldReady = make(chan struct{})
	if n == 0 {
		close(b.heldReady)
	}
	return b.heldReady
}

// Held records that a CP holds the kernel that raises name.
func (b *Board) Held(now sim.VTimeInSec, name string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.records[name] = &Record{Name: name, Held: now}
	b.numHeld++
	if b.heldReady != nil && b.numHeld == b.heldWant {
		close(b.heldReady)
	}
}

// Released records that the kernel that raises name started dispatching.
func (b *Board) Released(now sim.VTimeInSec, name string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if r, ok := b.records[name]; ok {
		r.Released = now
	}
}

// Done records that the kernel that raises name completed.
func (b *Board) Done(now sim.VTimeInSec, name string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if r, ok := b.records[name]; ok {
		r.Done = now
	}
}

// Records returns every held kernel's timeline, by name.
func (b *Board) Records() []Record {
	b.mu.Lock()
	defer b.mu.Unlock()
	var out []Record
	for _, r := range b.records {
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}
