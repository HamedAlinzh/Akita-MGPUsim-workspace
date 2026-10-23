#!/usr/bin/env python3
"""Step 8 model: the Step 6 work-group schedule with an event-driven message
model in place of 'link queue + round-trip tail'.

Tile t is owned by GPU t mod p. When a tile is ready on GPU a, a non-owner
queues its partial for the owner. Once all p-1 partials of a tile have
arrived, the owner queues the sum for every peer (all-reduce); a
reduce-scatter stops there. Each engine lane (one per engine, or one per peer
with -engine-peer-queues) takes its messages in ready order, spends the
per-message overhead on each after handing over the previous one, then the
link sends one message at a time; a message arrives one wire latency after
its transfer ends. The window ends at the last arrival.

Calibration (Step 2, TestCalibration): one message's latency beyond its
transfer is 1.07 x (overhead + latency), so the wire latency used here is
1.07 x (overhead + latency) - overhead.
"""
import heapq
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402

ROOT = m.ROOT


class Comm:
    def __init__(self, p, collective, bw, lat_ns, ovh_ns, tile_bytes, header=16,
                 eff=0.8, calib=1.07, per_peer=False):
        self.p, self.collective, self.per_peer = p, collective, per_peer
        self.x = (tile_bytes + header) / (bw * eff) * 1e6
        self.ovh = ovh_ns / 1e3
        self.wire = calib * (ovh_ns + lat_ns) / 1e3 - self.ovh

    def window(self, ready, bulk_at=None):
        """ready: per-tile ready time on every GPU (us). bulk_at: release
        every tile at this time instead (bulk mode)."""
        p = self.p
        heap = []  # (ready, seq, from, to, tile, reduced)
        seq = itertools.count()
        for t, r in enumerate(ready):
            r = bulk_at if bulk_at is not None else r
            o = t % p
            for a in range(p):
                if a != o:
                    heapq.heappush(heap, (r, next(seq), a, o, t, False))
        need = {t: p - 1 for t in range(len(ready))}
        got = {t: 0.0 for t in range(len(ready))}
        lane = {}
        link = {}
        last = 0.0
        while heap:
            r, _, a, b, t, reduced = heapq.heappop(heap)
            key = (a, b) if self.per_peer else a
            handed = max(lane.get(key, 0.0), r) + self.ovh
            lane[key] = handed
            done = max(link.get((a, b), 0.0), handed) + self.x
            link[(a, b)] = done
            arrive = done + self.wire
            last = max(last, arrive)
            if reduced:
                continue
            need[t] -= 1
            got[t] = max(got[t], arrive)
            if need[t] == 0 and self.collective == "all-reduce":
                for c in range(p):
                    if c != b:
                        heapq.heappush(heap, (got[t], next(seq), b, c, t, True))
        return last


def knob_comm(key, p=2, coll="all-reduce", per_peer=False):
    bw, lat, ovh = m.knobs()[key]
    return Comm(p, coll, bw, lat, ovh, 4096, per_peer=per_peer)


def search(gemm, comm, plans, top=3, bulk=False):
    res = []
    for plan in plans:
        ready, end = gemm.run(plan)
        w = comm.window(ready, bulk_at=end if bulk else None)
        res.append((w, end, tuple(plan)))
    res.sort()
    return res[:top]


class PSGemm:
    """Work-group schedule with processor sharing: CUS CUs, at most CAP
    resident work-groups each; a CU's throughput is split equally among its
    resident work-groups, so a work-group that takes w us with CAP resident
    takes w x n / CAP with n. Work-groups dispatch in order to the CU with
    the fewest resident (ordered dispatch). Times per work-group as in
    step6_model.Gemm: steady[s], or first[s] if dispatched at time 0."""

    def __init__(self, rows, cols, steady, first=None, launch=0.0, cus=4, cap=8):
        self.rows, self.cols = rows, cols
        self.steady, self.first, self.launch = steady, first, launch
        self.cus, self.cap = cus, cap

    def run(self, plan):
        jobs = [(row * self.cols + col, s) for row, s in enumerate(plan)
                for _ in range(s) for col in range(self.cols)]
        active = [[] for _ in range(self.cus)]  # [remaining work, tile]
        ready = [0.0] * (self.rows * self.cols)
        t, j = 0.0, 0
        while j < len(jobs) or any(active):
            while j < len(jobs):
                cu = min(range(self.cus), key=lambda c: len(active[c]))
                if len(active[cu]) >= self.cap:
                    break
                tile, s = jobs[j]
                w = self.first[s] if self.first and t == 0 else self.steady[s]
                active[cu].append([w, tile])
                j += 1
            # Next completion: work w at rate cap/n per resident work-group.
            dt = min(a[0] * len(act) / self.cap for act in active for a in act)
            for act in active:
                rate = self.cap / len(act) if act else 0
                for a in act:
                    a[0] -= dt * rate
            t += dt
            for act in active:
                for a in [a for a in act if a[0] <= 1e-9]:
                    act.remove(a)
                    ready[a[1]] = max(ready[a[1]], t)
        return [self.launch + r for r in ready], self.launch + t
