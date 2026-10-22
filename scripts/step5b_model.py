#!/usr/bin/env python3
"""Step 5b A: queue-model bounds and host-side split-K schedules.

GEMM model: 2 GPUs run the same schedule; 4 CUs x cap 8 = 32 slots; work-groups
start in dispatch order whenever a slot frees (ordered dispatch). A tile is
ready when its last work-group ends; the GEMM starts LAUNCH us after the
launch. Work-group times (us) from the Step 5 traces:
  steady: plain 137; split s=2 1.04 x 137/2; s=4 1.15 x 137/4 (last-band
          work-groups; f05_tile_s4 and gemm_s4)
  started at time 0 (first round), "slowdown kept": plain 172, s=2 72.5,
          s=4 48.8 (1.43 x 137/4); "removed": the steady times.
Comm model (step5_summary.py's queue model): tiles are served in ready order,
one tile's link time per direction each, + one tile round trip.

A schedule is a split factor per tile row (8 rows of 8 tiles), dispatched top
to bottom; a split row runs its K chunks one after another (chunk 0 of every
tile, then chunk 1, ...), as -splitk-ways launches do.
"""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_summary as s5  # noqa: E402

SLOTS, ROWS, COLS = 32, 8, 8
TILES = ROWS * COLS
T = 137.0
STEADY = {1: T, 2: 1.04 * T / 2, 4: 1.15 * T / 4}
FIRST_KEPT = {1: 172.0, 2: 72.5, 4: 48.8}
LAUNCH = 46.0  # launch to first dispatch, us (f05_tile_s1: 205.6 - 159.7)
OUT = os.path.join(s5.ROOT, "results", "step5b_bounds")


def gemm(plan, kept):
    """Tile ready times (us after launch) and GEMM end for a split plan."""
    jobs = []  # (tile, duration kind)
    for row, s in enumerate(plan):
        for _ in range(s):
            for col in range(COLS):
                jobs.append((row * COLS + col, s))
    free = [0.0] * SLOTS
    ready = [0.0] * TILES
    for tile, s in jobs:
        i = min(range(SLOTS), key=lambda k: free[k])
        start = free[i]
        dur = FIRST_KEPT[s] if kept and start == 0 else STEADY[s]
        free[i] = start + dur
        ready[tile] = max(ready[tile], free[i])
    return [LAUNCH + r for r in ready], LAUNCH + max(free)


def window(p, ready, presum=True, splits=None):
    c, rt = s5.per_msg(p)
    costs = [c] * TILES
    if not presum:
        costs = [c * ((splits[t // COLS] + 1) / 2) for t in range(TILES)]
    t = 0.0
    for when, cost in sorted(zip(ready, costs)):
        t = max(t, when) + cost
    return t + rt


def stream(span, first=None):
    """Tiles ready at an even rate over span us after the launch delay."""
    return [LAUNCH + span * (i + 1) / TILES for i in range(TILES)]


def plan_name(plan):
    return "".join(str(s) for s in plan)


def measured(name):
    r = s5.parse(os.path.join(s5.RUNS, name))
    return r["window"] if r else None


def main():
    lines = []
    out = lines.append
    plans = {"plain": (1,) * 8, "step5 s=2": (2, 2, 1, 1, 1, 1, 2, 2),
             "step5 s=4": (4, 1, 1, 1, 1, 1, 1, 4)}

    out("Calibration: model (slowdown kept) vs Step 5 measurements")
    out("plan | point | gemm_model | gemm_measured | window_model | window_measured | error")
    runs = {("plain", "f05"): "f05_tile_s1", ("step5 s=2", "f05"): "f05_tile_s2",
            ("step5 s=4", "f05"): "f05_tile_s4", ("step5 s=2", "f08"): "f08_tile_s2",
            ("step5 s=4", "f08"): "f08_tile_s4"}
    base = {"f05": 632.4, "f08": 1695.4}
    for (pname, p), run in runs.items():
        ready, end = gemm(plans[pname], True)
        w = window(p, ready)
        r = s5.parse(os.path.join(s5.RUNS, run))
        out(f"{pname} | {p} | {end:.1f} | {r['gemm']:.1f} | {w:.1f} | {r['window']:.1f} | "
            f"{w / r['window'] - 1:+.1%}")
    ready, _ = gemm(plans["plain"], True)
    out(f"plain | f08 | - | - | {window('f08', ready):.1f} | {base['f08']} | "
        f"{window('f08', ready) / base['f08'] - 1:+.1%}")

    ref = {"f05": 578.0, "f08": 1585.7}
    out("\nA.1/A.2: even streams (tiles ready at an even rate from span/64 to span)")
    out("stream | slowdown | span_us | f05_window | vs 578.0 | f08_window | vs 1585.7")
    for kept in (True, False):
        plain_span = (FIRST_KEPT[1] if kept else T) + T  # two rounds of the plain GEMM
        for label, span in (("ideal, no split cost", plain_span),
                            ("s=2 split cost 1.04", plain_span * 1.04),
                            ("s=4 split cost 1.43", plain_span * 1.43),
                            ("s=4 split cost 1.15", plain_span * 1.15)):
            if not kept and label.startswith("s=4 split cost 1.43"):
                continue
            if kept and label.startswith("s=4 split cost 1.15"):
                continue
            w5, w8 = window("f05", stream(span)), window("f08", stream(span))
            out(f"{label} | {'kept' if kept else 'removed'} | {span:.0f} | {w5:.1f} | "
                f"{w5 / ref['f05'] - 1:+.1%} | {w8:.1f} | {w8 / ref['f08'] - 1:+.1%}")

    out("\nA.3: every per-row split plan (s in {1,2,4} for each of 8 tile rows, "
        "dispatched top to bottom); best 5 per point and slowdown case, plus named plans")
    named = dict(plans)
    named["two s=2 bands first"] = (2, 2, 2, 2, 1, 1, 2, 2)
    named["graded 4,2 first"] = (4, 2, 1, 1, 1, 1, 2, 4)
    named["graded 4,2,2 first"] = (4, 2, 2, 1, 1, 1, 2, 2)
    for p in ("f05", "f08"):
        for kept in (True, False):
            res = []
            for plan in itertools.product((1, 2, 4), repeat=ROWS):
                ready, end = gemm(plan, kept)
                res.append((window(p, ready), end, plan))
            res.sort()
            tag = "kept" if kept else "removed"
            out(f"{p}, slowdown {tag}: plan | gemm_us | window_us | vs {ref[p]}")
            for w, end, plan in res[:5]:
                out(f"  {plan_name(plan)} | {end:.0f} | {w:.1f} | {w / ref[p] - 1:+.1%}")
            for name, plan in named.items():
                ready, end = gemm(plan, kept)
                w = window(p, ready)
                out(f"  {name} {plan_name(plan)} | {end:.0f} | {w:.1f} | {w / ref[p] - 1:+.1%}")

    text = "\n".join(lines)
    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "model.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
