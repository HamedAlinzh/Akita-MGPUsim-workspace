#!/usr/bin/env python3
"""Step 11 item 1: the first round of work-groups against the later rounds,
from existing -wg-trace (and, where a run has one, -mem-timeline) files.

Per s = 1 GEMM kernel (GPU 1): round 1 = the first `cap` work-groups by start
on each CU (they start together at the launch), later = the rest. Prints the
mean work-group time per round, its spread over CUs, tile rows and tile
columns, and the start/end spreads. For a run with mem.csv, the 10 us buckets
are split at the round-1 boundary (mean round-1 end) and the memory counters
(DRAM bytes, L2 read misses and MSHR merges, TLB misses) are given per
work-group-microsecond of each phase. Warm runs' traces also hold the cold
warm-up kernel (kernel_seq 0). Prints a summary; writes
results/step11_firstround/trace.txt.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_wgs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "step11_firstround")
CAP = 8

# label, run directory, kernel_seq (0 = the only or the warm-up kernel)
KERNELS = [
    ("4 CUs cold", "step9_baselines/runs/eng_f05_s1", 0),
    ("4 CUs A+C warm (warm-up: cold, dummy B)", "step9_baselines/runs/wab_s1", 0),
    ("4 CUs A+C warm", "step9_baselines/runs/wab_s1", 1),
    ("4 CUs all warm", "step9_baselines/runs/warm_s1", 1),
    ("4 CUs GEMM alone, warm-up (cold) [mem]", "step5b_bounds/runs/warm_plain", 0),
    ("4 CUs GEMM alone, all warm [mem]", "step5b_bounds/runs/warm_plain", 1),
    ("8 CUs cold [mem]", "step11_firstround/runs/c8_s1", 0),
    ("16 CUs cold", "step10_cushare/runs/eng_s1", 0),
    ("16 CUs A+C warm (warm-up: cold, dummy B)", "step10_cushare/runs/wab_s1", 0),
    ("16 CUs A+C warm", "step10_cushare/runs/wab_s1", 1),
    ("16 CUs all warm (warm-up: cold) [mem]", "step11_firstround/runs/w16_s1", 0),
    ("16 CUs all warm [mem]", "step11_firstround/runs/w16_s1", 1),
]

MEM_COLS = ["dram_read_bytes", "dram_write_bytes", "l2_read_miss", "l2_read_mshr_hit",
            "l2_read_hit", "l1v_read_miss", "l1v_read_mshr_hit", "l1v_tlb_miss", "l2_tlb_miss",
            "dram_reads"]


def kernel_rows(run, seq):
    path = os.path.join(R, run, "wgs.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in step5_wgs.load(path) if r["gpu"] == "1" and int(r["kernel_seq"]) == seq]
    if not rows or len({r["kernel_id"] for r in rows}) != 1:
        return None
    by_cu = collections.defaultdict(list)
    for r in rows:
        by_cu[r["cu"]].append(r)
    for ws in by_cu.values():
        ws.sort(key=lambda r: r["start"])
        for i, w in enumerate(ws):
            w["round1"] = i < CAP
            w["dur"] = w["end"] - w["start"]
    return rows, by_cu


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def spread(groups):
    """min-max of the group means."""
    ms = [mean(w["dur"] for w in ws) for ws in groups.values() if ws]
    return f"{min(ms):.1f}-{max(ms):.1f}"


def mem_phases(run, t0, t1, t2):
    """Counter sums over buckets in [t0, t1) and [t1, t2) (GPU 1), with a
    bucket split pro rata at the boundaries."""
    path = os.path.join(R, run, "mem.csv")
    if not os.path.exists(path):
        return None
    out = [collections.Counter(), collections.Counter()]
    rows = csv.DictReader(open(path))
    present = set(rows.fieldnames)
    for r in rows:
        if r["gpu"] != "1":
            continue
        a = float(r["start_us"])
        b = a + 10.0
        for i, (lo, hi) in enumerate(((t0, t1), (t1, t2))):
            f = max(0.0, min(b, hi) - max(a, lo)) / 10.0
            if f > 0:
                for c in MEM_COLS + ["dram_read_ns"]:
                    if c in present:
                        out[i][c] += f * float(r[c])
    return out, present


def main():
    L = [f"Round 1 = the first {CAP} work-groups per CU (they start at the launch); GPU 1; "
         "times in us"]
    L.append("kernel | WGs | round 1 mean | later mean | r1/later | round-1 start spread | "
             "round-1 end range | round-1 by CU | by tile row | by tile col | later by CU")
    mems = []
    for label, run, seq in KERNELS:
        k = kernel_rows(run, seq)
        if not k:
            L.append(f"{label} | (no trace)")
            continue
        rows, by_cu = k
        r1 = [w for w in rows if w["round1"]]
        rest = [w for w in rows if not w["round1"]]
        m1, m2 = mean(w["dur"] for w in r1), mean(w["dur"] for w in rest)
        g = lambda key, ws: {v: [w for w in ws if w[key] == v] for v in {w[key] for w in ws}}
        L.append(f"{label} | {len(rows)} | {m1:.1f} | {m2:.1f} | {m1 / m2:.2f} | "
                 f"{max(w['start'] for w in r1) - min(w['start'] for w in r1):.2f} | "
                 f"{min(w['end'] for w in r1) - min(w['start'] for w in r1):.1f}-"
                 f"{max(w['end'] for w in r1) - min(w['start'] for w in r1):.1f} | "
                 f"{spread(g('cu', r1))} | {spread(g('wg_y', r1))} | {spread(g('wg_x', r1))} | "
                 f"{spread(g('cu', rest))}")
        t0 = min(w["start"] for w in rows)
        t1 = mean(w["end"] for w in r1)
        t2 = max(w["end"] for w in rows)
        ph = mem_phases(run, t0, t1, t2)
        if ph:
            mems.append((label, ph, (t0, t1, t2), (len(r1), len(rest)), (m1, m2)))
    if mems:
        L.append("\nMemory counters per phase (GPU 1): phase 1 = launch to mean round-1 end, "
                 "phase 2 = to the kernel end; totals, and per work-group in that phase")
        L.append("(n/a: column not in that run's mem.csv, written before Step 11; DRAM read "
                 "latency = port arrival to completion)")
        L.append("kernel | phase | us | " + " | ".join(MEM_COLS) + " | mean DRAM read latency ns")
        for label, (ph, present), (t0, t1, t2), (n1, n2), _ in mems:
            for i, (lo, hi, n) in enumerate(((t0, t1, n1), (t1, t2, n2))):
                cells = [f"{ph[i][c]:.0f} ({ph[i][c] / n:.1f}/WG)" if c in present else "n/a"
                         for c in MEM_COLS]
                lat = (f"{ph[i]['dram_read_ns'] / ph[i]['dram_reads']:.0f}"
                       if ph[i]["dram_reads"] > 0 else "-")
                L.append(f"{label} | {i + 1} ({n} WGs) | {hi - lo:.1f} | " + " | ".join(cells) +
                         f" | {lat}")
    text = "\n".join(L)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "trace.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
