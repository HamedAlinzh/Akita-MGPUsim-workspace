#!/usr/bin/env python3
"""Step 5b B: round-1 vs round-2 work-group times and memory counters, per
launch (cold = the -warmup launch, warm = the measured one), from a run's
wgs.csv and mem.csv. Plain: round 1 = the first 32 work-groups dispatched.
Split: the first band, middle, last band. Memory counters are summed over
the 10 us buckets inside each phase's window (first dispatch to the next
phase's first dispatch)."""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_wgs  # noqa: E402

SLOTS = 32


def phases(rows, split):
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["kernel_id"].split("@")[0])].append(r)
    ids = sorted(by)
    launches = []
    if not split:
        for k in ids:
            ws = sorted(by[k], key=lambda w: w["start"])
            launches.append([("round 1", ws[:SLOTS]), ("round 2", ws[SLOTS:])])
    else:
        per = 5  # s = 2: 2 first-band chunks, the middle, 2 last-band chunks
        for i in range(0, len(ids), per):
            ks = ids[i:i + per]
            launches.append([("first band", by[ks[0]] + by[ks[1]]), ("middle", by[ks[2]]),
                             ("last band", by[ks[3]] + by[ks[4]])])
    return launches


def main(run_dir, split):
    rows = step5_wgs.load(os.path.join(run_dir, "wgs.csv"))
    step5_wgs.residency(rows)
    mem = list(csv.DictReader(open(os.path.join(run_dir, "mem.csv"))))
    cols = [c for c in mem[0] if c not in ("gpu", "start_us")]
    print(f"== {os.path.basename(run_dir)}")
    print("launch | phase | wgs | mean_wg_us | resident | window_us | " + " | ".join(cols))
    for li, launch in enumerate(phases(rows, split)):
        starts = [min(w["start"] for w in ws) for _, ws in launch]
        end = max(w["end"] for _, ws in launch for w in ws)
        for pi, (name, ws) in enumerate(launch):
            t0 = starts[pi]
            t1 = starts[pi + 1] if pi + 1 < len(starts) else end
            tot = collections.Counter()
            for m in mem:
                b = float(m["start_us"])
                if t0 <= b + 5 < t1:  # bucket centre inside the window
                    for c in cols:
                        tot[c] += float(m[c])
            dur = sum(w["end"] - w["start"] for w in ws) / len(ws)
            res = sum(w["resident"] for w in ws) / len(ws)
            label = "cold" if li == 0 else "warm"
            print(f"{label} | {name} | {len(ws)} | {dur:.1f} | {res:.2f} | {t1 - t0:.1f} | " +
                  " | ".join(f"{tot[c]:.0f}" for c in cols))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d, "s2" in os.path.basename(d))
