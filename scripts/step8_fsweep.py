#!/usr/bin/env python3
"""Step 8 item 2: s = 1 and the best per-row plan over f on 2 GPUs (f~0.5
shape: 256x256, K/GPU 256, 4 CUs, cap 8). For each f the link knobs come from
ratio_model.derive_knobs with the capped GEMM's measured time (as for the f~0.5
and f~0.8 points): -link-bw sets f, latency and overhead keep latency/transfer
at 0.39. Model: Step 6 work-group schedule + the Step 8 event model; full
search over 3^8 plans. With --runs, also prints each run point's knobs and
plan, and the measured windows if the runs exist."""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step5_summary as s5  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step8_edges")
RUNS = os.path.join(OUT, "runs")
SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RUN_F = [0.35, 0.65, 0.9]


def knobs(f):
    pt = rm.capped_points(4096)[1]  # the f~0.5 capped point: shape and GEMM
    k = rm.derive_knobs(pt, f, measured=rm.capped_measured())
    return k["link_bw_gbps"], k["link_latency_ns"], k["msg_overhead_ns"]


def best(f):
    bw, lat, ovh = knobs(f)
    g = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    cm = e.Comm(2, "all-reduce", bw * 1e9, lat, ovh, 4096)
    w1 = cm.window(g.run((1,) * 8)[0])
    top = e.search(g, cm, itertools.product((1, 2, 4), repeat=8), top=1)[0]
    return (bw, lat, ovh), w1, top


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["f | -link-bw GB/s | -link-latency ns | -msg-overhead ns | model s=1 | "
             "best plan | model plan | model change"]
    rows = {}
    for f in sorted(set(SWEEP + RUN_F)):
        (bw, lat, ovh), w1, (w, _, plan) = best(f)
        rows[f] = (bw, lat, ovh, w1, w, plan)
        lines.append(f"{f:.2f} | {bw:.4f} | {lat:.0f} | {ovh:.0f} | {w1:.1f} | {m.name(plan)} | "
                     f"{w:.1f} | {w / w1 - 1:+.1%}")
    lines.append("\nRun points: f | s=1 measured / model / error | plan measured / model / "
                 "error | measured change")
    for f in RUN_F:
        bw, lat, ovh, w1, w, plan = rows[f]
        tag = f"f{int(round(f * 100)):03d}"
        a = s5.parse(os.path.join(RUNS, f"{tag}_s1"))
        b = s5.parse(os.path.join(RUNS, f"{tag}_plan"))
        print(f"RUN {tag} -link-bw={bw:.4f} -link-latency={lat:.0f} -msg-overhead={ovh:.0f} "
              f"-splitk-plan={m.name(plan)}")
        if a and b and "window" in a and "window" in b:
            lines.append(f"{f:.2f} | {a['window']:.1f} / {w1:.1f} / {w1 / a['window'] - 1:+.1%} | "
                         f"{b['window']:.1f} / {w:.1f} / {w / b['window'] - 1:+.1%} | "
                         f"{b['window'] / a['window'] - 1:+.1%} (verified {a['ok'] and b['ok']})")
    with open(os.path.join(OUT, "fsweep.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
