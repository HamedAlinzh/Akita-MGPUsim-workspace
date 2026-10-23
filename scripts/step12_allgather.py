#!/usr/bin/env python3
"""Step 12: all-gather of A before the GEMM on the engines, s = 1.

Every GPU holds M/p rows of A (full K 256) and its own 256 x 256 block of B,
and computes C = A x B_local (256 x 256, 4 CUs, tileSize 8, cap 8). The
engines push each GPU's rows to every peer, one row of tiles (32 rows x K)
per message, direct order on per-peer queues, and raise gpuN/a/rowR once a
row is written in L2. Runs (results/step12_allgather/runs/):
  g{2,8}_bulk   gather, then one GEMM launch held until every row is in
  g{2,8}_rows   one launch per row of tiles, local rows first, remote rows
                held until their row arrives
  g{2,8}_local  A already on every GPU, one launch (the floor)
  g2_mccl       mccl direct all-gather (one push kernel per direction,
                -collective-threads 16384), then the GEMM
  defaults_433  default flags, expect 432.742 us

Model: scripts/step9_allgather.py (queue model of the design note), s = 1,
direct order. The mccl run has no fitted model; its estimate is one launch
(46 us) + the shard at 0.50 of the link (the Step 7 fit's eta) + the GEMM.

Writes runs.txt (before running) and, from the runs' stdout, summary.txt.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step9_allgather as ag  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step12_allgather")
KNOBS = {2: (0.487, 707, 3403), 8: (0.122, 2827, 13611)}  # design note, f 0.5
COMMON = ("-timing -verify -m=256 -n=256 -k=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 "
          "-gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv -allgather-a")
SHARD_BYTES = 128 * 256 * 4  # one GPU's rows of A at 2 GPUs


def flags(p, mode):
    bw, lat, ovh = KNOBS[p]
    f = (f"-gpus={','.join(str(i + 1) for i in range(p))} {COMMON} "
         f"-link-bw={bw} -link-latency={lat} -msg-overhead={ovh}")
    if p > 2:
        f += " -peer-mesh"
    if mode in ("bulk", "rows"):
        f += " -engine-mode=tile" + (" -engine-peer-queues" if p > 2 else "")
    if mode != "rows":
        f += f" -allgather-{mode}"
    if mode == "mccl":
        f += " -collective-threads=16384"
    return f


RUNS = [("g2_bulk", 2, "bulk"), ("g2_rows", 2, "rows"), ("g8_bulk", 8, "bulk"),
        ("g8_rows", 8, "rows"), ("g2_local", 2, "local"), ("g8_local", 8, "local"),
        ("g2_mccl", 2, "mccl")]


def model():
    ones = (1,) * ag.ROWS
    alone = ag.gemm(ones, {(r, 0): 0.0 for r in range(ag.ROWS)})
    out = {}
    for p in (2, 8):
        arr = ag.arrivals(p, ones)
        out[(p, "bulk")] = max(arr.values()) + alone
        out[(p, "rows")] = ag.gemm(ones, arr)
        out[(p, "local")] = alone
    out[(2, "mccl")] = m.SIM_LAUNCH + SHARD_BYTES / (0.5 * KNOBS[2][0] * 1e9) * 1e6 + alone
    return out


def parse(name):
    path = os.path.join(OUT, "runs", name, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    r = {"ok": "Passed!" in text}
    mm = re.search(r"allgather phases: mode=\w+ window=([\d.]+) gather=([\d.]+) "
                   r"gemm_wait=([\d.]+)", text) or \
        re.search(r"allreducegemm phases: window=([\d.]+)", text)
    if mm:
        r["window"] = float(mm.group(1)) * 1e6
    if mm and mm.lastindex == 3:
        r["gather"], r["gemm"] = float(mm.group(2)) * 1e6, float(mm.group(3)) * 1e6
    w = re.search(r"wall_s=([\d.]+)", text)
    r["wall"] = float(w.group(1)) if w else None
    r["links"] = [tuple(float(x) for x in g) for g in re.findall(
        r"allgather link: from=\d+ to=\d+ msgs=(\d+) traffic_bytes=(\d+) "
        r"first_sent=([\d.]+) last_received=([\d.]+)", text)]
    r["launches"] = [tuple(float(x) for x in g) for g in re.findall(
        r"allgather launch: name=\S+ held=([\d.]+) signal=([\d.]+) released=([\d.]+) "
        r"done=([\d.]+)", text)]
    p = re.search(r"allgather push: done=([\d.]+) last_arrival=([\d.]+)", text)
    if p:
        r["push_done"], r["last_arrival"] = float(p.group(1)) * 1e6, float(p.group(2)) * 1e6
    r["dispatchers"] = [int(x) for x in re.findall(r"allgather dispatchers: gpu=\d+ "
                                                   r"max_busy=(\d+)", text)]
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step12_allgather.py\n")
        for name, p, mode in RUNS:
            fh.write(f"{name} {flags(p, mode)}\n")
        fh.write("defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 "
                 "-algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 "
                 "-collective-threads=16384\n")

    pred = model()
    L = ["run | GPUs | mode | measured window us | model us | error | gather / GEMM wait us | "
         "last row in L2 us | push link efficiency (from start; from first send) | "
         "held launches | max dispatchers busy | releases late (>0.1 us after max(held, "
         "signal)) | launch reached CP after its row | verified | wall s"]
    for name, p, mode in RUNS:
        r = parse(name)
        mp = pred[(p, mode)]
        if not r or "window" not in r:
            L.append(f"{name} | {p} | {mode} | (not run) | {mp:.1f}")
            continue
        err = f"{r['window'] / mp - 1:+.1%}" if r["ok"] else "-"
        bw = KNOBS[p][0] * 1e9
        effs = [(b / (lr * bw), b / ((lr - fs) * bw)) for _, b, fs, lr in r["links"]]
        eff = (f"{min(e[0] for e in effs):.3f}-{max(e[0] for e in effs):.3f}; "
               f"{min(e[1] for e in effs):.3f}-{max(e[1] for e in effs):.3f}") if effs else "-"
        late = [rel - max(h, s) for h, s, rel, _ in r["launches"]]
        n_late = sum(1 for d in late if d > 1e-7)
        after = sum(1 for h, s, _, _ in r["launches"] if h > s)
        phases = (f"{r['gather']:.1f} / {r['gemm']:.1f}" if "gather" in r else "-")
        L.append(f"{name} | {p} | {mode} | {r['window']:.1f} | {mp:.1f} | {err} | {phases} | "
                 f"{r.get('last_arrival', float('nan')):.1f} | {eff} | {len(r['launches'])} | "
                 f"{max(r['dispatchers'], default=0)} | {n_late} (max "
                 f"{max(late, default=0) * 1e9:.0f} ns) | {after} | {r['ok']} | {r['wall']}")
    d = parse("defaults_433")
    if d and "window" in d:
        L.append(f"defaults: window {d['window']:.3f} us, verified {d['ok']}")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
