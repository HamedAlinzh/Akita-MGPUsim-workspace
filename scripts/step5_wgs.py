#!/usr/bin/env python3
"""Per-kernel work-group timing from a -wg-trace CSV: dispatch window, end
window, mean duration, and mean work-groups resident on the same CU over each
work-group's life (time-weighted)."""
import csv
import collections
import sys


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["start"], r["end"] = float(r["start_us"]), float(r["end_us"])
    return rows


def residency(rows):
    """Mean number of work-groups on the CU while each one ran."""
    by_cu = collections.defaultdict(list)
    for r in rows:
        by_cu[(r["gpu"], r["cu"])].append(r)
    for wgs in by_cu.values():
        events = sorted([(w["start"], 1) for w in wgs] + [(w["end"], -1) for w in wgs])
        times, counts, n = [], [], 0
        for t, d in events:
            n += d
            times.append(t)
            counts.append(n)
        for w in wgs:
            area, prev, cur = 0.0, w["start"], None
            for t, c in zip(times, counts):
                if t <= w["start"]:
                    cur = c
                    continue
                if t > w["end"]:
                    break
                area += cur * (t - prev)
                prev, cur = t, c
            area += cur * (w["end"] - prev)
            w["resident"] = area / max(w["end"] - w["start"], 1e-12)


def kernels(rows):
    residency(rows)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["gpu"], int(r["kernel_id"].split("@")[0]))].append(r)
    out = []
    for (gpu, kid), ws in sorted(by.items()):
        d = [w["end"] - w["start"] for w in ws]
        out.append(dict(gpu=gpu, kid=kid, grid_y=int(ws[0]["grid_y"]), n=len(ws),
                        start0=min(w["start"] for w in ws), start1=max(w["start"] for w in ws),
                        end0=min(w["end"] for w in ws), end1=max(w["end"] for w in ws),
                        dur=sum(d) / len(d),
                        resident=sum(w["resident"] for w in ws) / len(ws)))
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print("==", path)
        for k in kernels(load(path)):
            print("gpu{gpu} launch {kid} grid_y={grid_y} wgs={n} dispatched {start0:.1f}-{start1:.1f} "
                  "ended {end0:.1f}-{end1:.1f} mean_dur {dur:.1f} resident {resident:.2f}".format(**k))
