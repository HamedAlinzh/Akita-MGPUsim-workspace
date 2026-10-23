#!/usr/bin/env python3
"""Step 8 item 1c: when does the split plan help software overlap?

Model (step7_model's software model, refined): the host all-reduces bands of
C one after another once each band's GEMM is done; each call costs F plus its
bytes at eta of the link (eta = 0.502, fitted to the capped mccl runs), split
bands first pay (s-1) pre-sum calls of PRESUM each (scaled with F, since it is
one more launched kernel: 49.4 us at F = 277). While communication runs, the
GEMM loses a share phi of its CU throughput: work-groups that start after the
first call begins run 1/(1-phi) slower.

Sweep: F in {277 (this simulator), 20 (published: RCCL point-to-point minimum on
MI300A, arXiv 2508.11298 Sec. 6.1.1), 5}; phi in {0.5 (measured: overlapped GEMM
work-groups ran 137 -> 270 us), 1/4, 1/8, 0}. For each, the best band layout
for s = 1 (one band = no overlap) and the best layout with any split factor
per band, against the hardware engine (s = 1 632.4 us, plan 524.9 us).
"""
import heapq
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step7_model as m7  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step8_edges")
ROWS, COLS, SLOTS = 8, 8, 32
F_FIT, ETA = None, None
HW = {"s1": 632.4, "plan": 524.9}


def gemm(bands, phi, comm_start):
    heap = [0.0] * SLOTS
    ready = [0.0] * ROWS
    for r0, n, s in bands:
        for _ in range(s):
            for r in range(r0, r0 + n):
                for _ in range(COLS):
                    start = heapq.heappop(heap)
                    dur = m.SIM_FIRST[s] if start == 0 else m.SIM_STEADY[s]
                    if m.SIM_LAUNCH + start >= comm_start:
                        dur /= 1 - phi
                    heapq.heappush(heap, start + dur)
                    ready[r] = max(ready[r], start + dur)
    return [m.SIM_LAUNCH + x for x in ready]


def window(bands, F, phi, presum, bw):
    # The first call starts when the first band is done; that band's own
    # ready time does not depend on phi.
    first = max(gemm(bands, 0.0, float("inf"))[bands[0][0]:bands[0][0] + bands[0][1]])
    ready = gemm(bands, phi, first)
    t = 0.0
    for r0, n, s in bands:
        t = max(t, max(ready[r0:r0 + n])) + (s - 1) * presum + F + \
            n * COLS * 4096 / (ETA * bw) * 1e6
    return t


def layouts():
    for cuts in itertools.product((0, 1), repeat=ROWS - 1):
        bands, start = [], 0
        for i, c in enumerate(cuts + (1,)):
            if c:
                bands.append((start, i + 1 - start))
                start = i + 1
        yield bands


def main():
    global F_FIT, ETA
    F_FIT, ETA, _, _ = m7.fit_sw()
    bw = m.knobs()["f05"][0]
    all_layouts = list(layouts())
    lines = [f"Software overlap at f~0.5, 2 GPUs (eta {ETA:.3f}); engine: s=1 {HW['s1']} us, "
             f"plan {HW['plan']} us (-17.0%)",
             "F (us) | phi | best s=1 (layout) | best with splits (layout: rows x s) | "
             "split vs s=1 | split vs engine plan"]
    for F in (F_FIT, 20.0, 5.0):
        presum = m7.PRESUM_US * F / F_FIT
        for phi in (0.5, 0.25, 0.125, 0.0):
            b1 = min((window([(a, n, 1) for a, n in L], F, phi, presum, bw), L)
                     for L in all_layouts)
            best = None
            for L in all_layouts:
                for ss in itertools.product((1, 2, 4), repeat=len(L)):
                    bands = [(a, n, s) for (a, n), s in zip(L, ss)]
                    w = window(bands, F, phi, presum, bw)
                    if best is None or w < best[0]:
                        best = (w, bands)
            fmt1 = ",".join(str(n * 32) for _, n in b1[1])
            fmt2 = ",".join(f"{n * 32}x{s}" for _, n, s in best[1])
            lines.append(f"{F:.0f} | {phi:g} | {b1[0]:.1f} ({fmt1}) | {best[0]:.1f} ({fmt2}) | "
                         f"{best[0] / b1[0] - 1:+.1%} | {best[0] / HW['plan'] - 1:+.1%}")
            print(lines[-1], flush=True)
    with open(os.path.join(OUT, "sw_sweep.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:2]))


if __name__ == "__main__":
    main()
