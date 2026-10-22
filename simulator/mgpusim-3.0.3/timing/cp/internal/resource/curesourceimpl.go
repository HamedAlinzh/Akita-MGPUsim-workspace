package resource

import (
	"sync"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/kernels"
)

// CUResourceImpl implements CUResource
type CUResourceImpl struct {
	sync.Mutex

	port sim.Port

	wfPoolFreeCount []int

	sregCount       int
	sregGranularity int
	sregMask        resourceMask

	vregCounts      []int
	vregGranularity int
	vregMasks       []resourceMask

	ldsByteSize    int
	ldsGranularity int
	ldsMask        resourceMask

	// countDynamicLDS makes a work-group occupy the LDS in its dispatch
	// packet, which includes the dynamic local arguments (driver.LocalPtr),
	// rather than only the code object's static group segment.
	countDynamicLDS bool

	// Dispatch shaping (0 = off): wgCap caps the work-groups in flight on
	// the CU; with rampStart the cap starts there and grows by one for every
	// work-group the CU completes in the current kernel.
	wgCap     int
	rampStart int
	completed int

	pool  *CUResourcePoolImpl
	index int

	nextSIMD    int
	reservedWGs map[*kernels.WorkGroup][]WfLocation
}

// DispatchingPort returns the port that the dispatcher send message to.
func (r *CUResourceImpl) DispatchingPort() sim.Port {
	return r.port
}

// ReserveResourceForWG checks if there is space to hold the work-group. If so,
// this function reserves the resouces for the work-group and returns how the
// resources are allocated.
func (r *CUResourceImpl) ReserveResourceForWG(wg *kernels.WorkGroup) (
	locations []WfLocation,
	ok bool,
) {
	if limit := r.wgLimit(wg); limit > 0 && len(r.reservedWGs) >= limit {
		return nil, false
	}
	if r.pool != nil && !r.pool.allows(r.index, wg) {
		return nil, false
	}

	ok = true
	locations = make([]WfLocation, len(wg.Wavefronts))

	for i, wf := range wg.Wavefronts {
		locations[i].Wavefront = wf
	}

	if !r.withinSGPRLimitation(wg, locations) {
		ok = false
	}

	if ok && !r.withinLDSLimitation(wg, locations) {
		ok = false
	}

	if ok && !r.matchWfWithSIMDs(wg, locations) {
		ok = false
	}

	if ok {
		r.reserveResources(wg, locations)
		return locations, true
	}

	r.clearTempReservation(wg)
	return nil, ok
}

func (r *CUResourceImpl) withinSGPRLimitation(
	wg *kernels.WorkGroup,
	locations []WfLocation,
) bool {
	co := wg.CodeObject
	required := r.unitsOccupy(int(co.WFSgprCount), r.sregGranularity)

	for i := range wg.Wavefronts {
		location := &locations[i]
		offset, ok := r.sregMask.nextRegion(required, allocStatusFree)
		if !ok {
			return false
		}
		location.SGPROffset = offset * 16 * 4 // 16 reg, 4 byte each
		r.sregMask.setStatus(offset, required, allocStatusToReserve)
	}

	return true
}

func (r *CUResourceImpl) withinLDSLimitation(
	wg *kernels.WorkGroup,
	locations []WfLocation,
) bool {
	required := r.unitsOccupy(r.ldsBytesOf(wg), r.ldsGranularity)
	offset, ok := r.ldsMask.nextRegion(required, allocStatusFree)
	if !ok {
		return false
	}

	for i := range wg.Wavefronts {
		location := &locations[i]
		location.LDSOffset = offset * r.ldsGranularity
	}

	r.ldsMask.setStatus(offset, required, allocStatusToReserve)
	return true
}

// Maps the wfs of a work-group to the SIMDs in the compute unit
// This function sets the value of req.WfDispatchMap, to keep the information
// about which SIMD should a wf dispatch to. This function also returns
// a boolean value for if the matching is successful.
func (r *CUResourceImpl) matchWfWithSIMDs(
	wg *kernels.WorkGroup,
	locations []WfLocation,
) bool {
	vgprToUse := make([]int, len(r.wfPoolFreeCount))
	wfPoolEntryUsed := make([]int, len(r.wfPoolFreeCount))
	co := wg.CodeObject

	for i := range wg.Wavefronts {
		location := &locations[i]
		firstSIMDTested := r.nextSIMD
		firstTry := true
		found := false
		required := r.unitsOccupy(int(co.WIVgprCount), r.vregGranularity)

		for firstTry || r.nextSIMD != firstSIMDTested {
			firstTry = false
			offset, ok := r.vregMasks[r.nextSIMD].
				nextRegion(required, allocStatusFree)

			if ok && r.wfPoolFreeCount[r.nextSIMD]-wfPoolEntryUsed[r.nextSIMD] > 0 {
				found = true
				vgprToUse[r.nextSIMD] += required
				wfPoolEntryUsed[r.nextSIMD]++
				location.SIMDID = r.nextSIMD
				location.VGPROffset = offset * r.vregGranularity * 4 //  4 bytes per register
				r.vregMasks[r.nextSIMD].setStatus(offset, required,
					allocStatusToReserve)
			}

			r.nextSIMD++
			if r.nextSIMD >= len(r.wfPoolFreeCount) {
				r.nextSIMD = 0
			}

			if found {
				break
			}
		}

		if !found {
			return false
		}
	}

	return true
}

// wgLimit returns the current cap on work-groups in flight, 0 for none. A
// kernel's own WGCap replaces the platform-wide cap.
func (r *CUResourceImpl) wgLimit(wg *kernels.WorkGroup) int {
	limit := r.wgCap
	if wg.CodeObject != nil && wg.CodeObject.WGCap > 0 {
		limit = wg.CodeObject.WGCap
	}
	if r.rampStart > 0 {
		ramp := r.rampStart + r.completed
		if limit == 0 || ramp < limit {
			limit = ramp
		}
	}
	return limit
}

// ldsBytesOf returns the LDS bytes a work-group occupies.
func (r *CUResourceImpl) ldsBytesOf(wg *kernels.WorkGroup) int {
	if r.countDynamicLDS && wg.Packet != nil {
		return int(wg.Packet.GroupSegmentSize)
	}
	return int(wg.CodeObject.WGGroupSegmentByteSize)
}

func (r *CUResourceImpl) unitsOccupy(amount, granularity int) int {
	if amount%granularity == 0 {
		return amount / granularity
	}
	return amount/granularity + 1
}

func (r *CUResourceImpl) reserveResources(
	wg *kernels.WorkGroup,
	locations []WfLocation,
) {
	for _, location := range locations {
		r.wfPoolFreeCount[location.SIMDID]--
	}

	r.sregMask.convertStatus(allocStatusToReserve, allocStatusReserved)
	r.ldsMask.convertStatus(allocStatusToReserve, allocStatusReserved)
	for i := 0; i < len(r.wfPoolFreeCount); i++ {
		r.vregMasks[i].convertStatus(allocStatusToReserve, allocStatusReserved)
	}

	r.neverReserveTwice(wg)
	r.reservedWGs[wg] = locations
}

func (r *CUResourceImpl) neverReserveTwice(wg *kernels.WorkGroup) {
	if _, found := r.reservedWGs[wg]; found {
		panic("reserving a work-group twice")
	}
}

func (r *CUResourceImpl) clearTempReservation(wg *kernels.WorkGroup) {
	r.sregMask.convertStatus(allocStatusToReserve, allocStatusFree)
	r.ldsMask.convertStatus(allocStatusToReserve, allocStatusFree)
	for i := 0; i < len(r.wfPoolFreeCount); i++ {
		r.vregMasks[i].convertStatus(allocStatusToReserve, allocStatusFree)
	}
}

// FreeResourcesForWG marks all the resources used by a work-group available.
func (r *CUResourceImpl) FreeResourcesForWG(wg *kernels.WorkGroup) {
	locations, found := r.reservedWGs[wg]
	if !found {
		panic("work-group not found")
	}

	co := wg.CodeObject
	for _, location := range locations {
		r.wfPoolFreeCount[location.SIMDID]++

		ldsUnits := r.unitsOccupy(r.ldsBytesOf(wg), r.ldsGranularity)
		r.ldsMask.setStatus(location.LDSOffset/r.ldsGranularity, ldsUnits,
			allocStatusFree)

		sgprUnits := r.unitsOccupy(int(co.WFSgprCount), r.sregGranularity)
		r.sregMask.setStatus(location.SGPROffset/4/r.sregGranularity,
			sgprUnits, allocStatusFree)

		vgprUnits := r.unitsOccupy(int(co.WIVgprCount), r.vregGranularity)
		r.vregMasks[location.SIMDID].setStatus(
			location.VGPROffset/4/r.vregGranularity, vgprUnits,
			allocStatusFree)
	}

	delete(r.reservedWGs, wg)
	r.completed++
}
