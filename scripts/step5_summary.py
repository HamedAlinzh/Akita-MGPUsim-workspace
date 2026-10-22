#!/usr/bin/env python3
"""Step 5 summary: split-K first and last bands against the capped baseline
(results/step4b_capped/runs, s = 1).

Two window models, both + one tile round trip:
  simple: max(GEMM, first ready + comm)
  queue:  tiles become ready when their last work-group ends (from the -wg-trace
          CSV, the later GPU of the two) and the link serves them in ready order,
          one tile's messages per direction at a time.
comm, transfer and round trip are as in step4b_summary.py: messages per direction
x max(transfer, overhead), transfer = traffic / (0.8 x link bandwidth), round trip
= 2 x 1.07 x (overhead + latency) + 2 x transfer. Without pre-sum a split tile
the peer owns costs s partial messages instead of one.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_wgs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "step5_splitk", "runs")
BASE = os.path.join(ROOT, "results", "step4b_capped", "runs")
KNOBS = {"f05": (0.7807e9, 441e-9, 2121e-9), "f08": (0.223e9, 1542e-9, 7424e-9)}
M, N, EDGE = 256, 256, 32
TILES_X = N // EDGE
TILES = (M // EDGE) * TILES_X
SPLIT = {"s2": (2, 64), "s4": (4, 32)}


def parse(path):
    try:
        text = open(os.path.join(path, "stdout.txt")).read()
    except FileNotFoundError:
        return None
    m = re.search(r"window=([\d.]+) gemm_wait=([\d.]+) \w+=([\d.]+)(?: first_ready=([\d.]+))?", text)
    if not m:
        return dict(ok=False)
    msgs = sorted(set(int(c) for c in re.findall(r"count=(\d+)", text)))
    wall = re.search(r"wall_s=([\d.]+)", text)
    return dict(ok="Passed!" in text, window=float(m.group(1)) * 1e6,
                gemm=float(m.group(2)) * 1e6,
                first=float(m.group(4)) * 1e6 if m.group(4) else None, msgs=msgs,
                wall=float(wall.group(1)) if wall else None)


def per_msg(p):
    bw, lat, ovh = KNOBS[p]
    x = (4096 + 16) / (0.8 * bw)
    return max(x, ovh) * 1e6, (2 * 1.07 * (ovh + lat) + 2 * x) * 1e6


def tile_costs(p, s, rows, presum):
    """Link time per direction for each tile's messages (an average of the two
    directions: one carries the partials, the other the sum)."""
    c, _ = per_msg(p)
    costs = []
    for t in range(TILES):
        row = t // TILES_X * EDGE
        split = s > 1 and (row < rows or row >= M - rows)
        costs.append(c * ((s + 1) / 2 if split and not presum else 1))
    return costs


def tile_ready(wgs_csv, s, rows):
    """Absolute time each tile is ready on both GPUs: the latest end of the
    work-groups that write it. Launches are numbered per GPU in launch order:
    s first-band chunks, the middle, s last-band chunks (or one launch)."""
    recs = step5_wgs.load(wgs_csv)
    launches = {}
    for r in recs:
        launches.setdefault(r["gpu"], set()).add(int(r["kernel_id"].split("@")[0]))
    ready = [0.0] * TILES
    for r in recs:
        order = sorted(launches[r["gpu"]]).index(int(r["kernel_id"].split("@")[0]))
        if s == 1:
            row0 = 0
        elif order < s:  # first band
            row0 = 0
        elif order == s:
            row0 = rows
        else:
            row0 = M - rows
        tile = (row0 // EDGE + int(r["wg_y"])) * TILES_X + int(r["wg_x"])
        ready[tile] = max(ready[tile], r["end"])
    return ready


def queue_model(p, ready, first_rel, costs):
    _, rt = per_msg(p)
    offset = min(ready) - first_rel
    t = 0.0
    for when, c in sorted(zip(ready, costs)):
        t = max(t, when - offset) + c
    return t + rt


def simple_model(p, gemm, first, costs):
    _, rt = per_msg(p)
    return max(gemm, first + sum(costs)) + rt


def main():
    base = {k: parse(os.path.join(BASE, k)) for k in ("f05_tile", "f08_tile", "f05_bulk")}
    rows = []
    for name, p, mode, s_key, presum in [
        ("f05_tile", "f05", "tile", None, True),
        ("f05_tile_s2", "f05", "tile", "s2", True),
        ("f05_tile_s4", "f05", "tile", "s4", True),
        ("f05_tile_s2_nopresum", "f05", "tile", "s2", False),
        ("f08_tile", "f08", "tile", None, True),
        ("f08_tile_s2", "f08", "tile", "s2", True),
        ("f08_tile_s4", "f08", "tile", "s4", True),
        ("f05_bulk", "f05", "bulk", None, True),
        ("f05_bulk_s2", "f05", "bulk", "s2", True),
    ]:
        s, r_rows = SPLIT[s_key] if s_key else (1, 0)
        r = base.get(name) if s_key is None else parse(os.path.join(RUNS, name))
        if name == "f05_tile":  # the s = 1 rerun with a trace, if it matches
            r = parse(os.path.join(RUNS, "f05_tile_s1")) or r
        if not r or "window" not in r:
            print(f"{name}: missing")
            continue
        ref = base[f"{p}_{mode}"]
        costs = tile_costs(p, s, r_rows, presum)
        simple = simple_model(p, r["gemm"], r["first"], costs)
        if mode == "bulk":
            simple = r["gemm"] + sum(costs) + per_msg(p)[1]
        q = "-"
        csv = os.path.join(RUNS, name, "wgs.csv")
        if mode == "tile" and os.path.exists(csv):
            q = f"{queue_model(p, tile_ready(csv, s, r_rows), r['first'], costs):.0f}"
        rows.append((name, s, presum, r, ref, simple, q))

    print("run | s | presum | verified | first_ready_us | gemm_us | gemm_vs_s1 | window_us | "
          "window_vs_s1 | simple_model | queue_model | msgs/dir")
    for name, s, presum, r, ref, simple, q in rows:
        print(f"{name} | {s} | {'on' if presum else 'off'} | {r['ok']} | {r['first']:.1f} | "
              f"{r['gemm']:.1f} | {r['gemm'] / ref['gemm'] - 1:+.1%} | {r['window']:.1f} | "
              f"{r['window'] / ref['window'] - 1:+.1%} | {simple:.0f} | {q} | {r['msgs']}")

    print("\nGEMM alone (1 GPU, K 256) | s | verified | first_ready_us | gemm_us")
    for s_key in ("s2", "s4"):
        g = parse(os.path.join(RUNS, f"gemm_{s_key}"))
        if g and "window" in g:
            print(f"gemm_{s_key} | {SPLIT[s_key][0]} | {g['ok']} | {g['first']:.1f} | {g['gemm']:.1f}")

    print("\nwork-groups (mean duration us, mean resident on the CU) by band, GPU 1")
    s1 = os.path.join(RUNS, "f05_tile_s1", "wgs.csv")
    if os.path.exists(s1):
        ks = [k for k in step5_wgs.kernels(step5_wgs.load(s1)) if k["gpu"] == "1"]
        recs = [r for r in step5_wgs.load(s1) if r["gpu"] == "1"]
        step5_wgs.residency(recs)
        recs.sort(key=lambda r: r["start"])
        for label, rr in (("round 1", recs[:32]), ("round 2", recs[32:])):
            print(f"f05_tile_s1 {label}: {sum(r['end'] - r['start'] for r in rr) / len(rr):.1f} us "
                  f"x{sum(r['resident'] for r in rr) / len(rr):.2f}, dispatched "
                  f"{rr[0]['start']:.1f}-{rr[-1]['start']:.1f}, ended "
                  f"{min(r['end'] for r in rr):.1f}-{max(r['end'] for r in rr):.1f}")
    for name in ("gemm_s2", "gemm_s4", "f05_tile_s2", "f05_tile_s4"):
        csv = os.path.join(RUNS, name, "wgs.csv")
        if not os.path.exists(csv):
            continue
        s = SPLIT[name.split("_")[-1]][0]
        ks = [k for k in step5_wgs.kernels(step5_wgs.load(csv)) if k["gpu"] == "1"]
        ks.sort(key=lambda k: k["kid"])
        groups = {"first": ks[:s], "middle": ks[s:s + 1], "last": ks[s + 1:]}
        parts = []
        for band, kk in groups.items():
            n = sum(k["n"] for k in kk)
            dur = sum(k["dur"] * k["n"] for k in kk) / n
            res = sum(k["resident"] * k["n"] for k in kk) / n
            parts.append(f"{band} {dur:.1f} us x{res:.2f}")
        mid = groups["middle"][0]["dur"]
        split = sum(k["dur"] * k["n"] for k in groups["first"]) / sum(k["n"] for k in groups["first"])
        print(f"{name}: " + ", ".join(parts) + f"; first-band WG x s / middle WG = {split * s / mid:.2f}")

    d = parse(os.path.join(RUNS, "defaults_433"))
    if d:
        m = re.search(r"window=([\d.]+)", open(os.path.join(RUNS, "defaults_433", "stdout.txt")).read())
        print(f"\ndefaults: window {float(m.group(1)) * 1e6:.3f} us, verified {d['ok']}")


if __name__ == "__main__":
    main()
