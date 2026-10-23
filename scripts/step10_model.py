#!/usr/bin/env python3
"""Step 10: the f~0.5 point at 16 CUs per GPU, so one comm CU is about 6% of
the GPU. Shape 512x512, K/GPU 256, cap 8: 256 tiles / (16 CUs x 8) = 2.0
waves, as 256x256 on 4 CUs. Link knobs from ratio_model.derive_knobs at the
f~0.5 point's f (0.555), with the 4-CU capped GEMM time as the estimate (each
CU does the same 16 tiles); achieved f is reported from the s = 1 run.

Engine model: the Step 6 list schedule (16 x 8 = 128 slots, the 4-CU
work-group times) and the Step 8 event messages. Plans: the structured
16-row family (4s, 2s, 1s, 2s, 4s), at most 36 launches so that 40 CP
dispatchers suffice. Writes results/step10_cushare/model.txt and runs.txt.

Step 10b: the FlashOverlap-style runs (1 comm CU of 16) use the Step 9
software-path model with its saved constants (results/step9_baselines/
params.json, not refitted) on 16 x 16 tiles and 15 GEMM CUs; the partition
is the model's best for s = 1 and for the engine's plan. The A-and-C-warm
engine runs use half of the cold first round's extra time, as in Step 9.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402
import step9_model as s9  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step10_cushare")
RUNS = os.path.join(OUT, "runs")
M = N = 512
ROWS = COLS = 16
CUS = 16
BASE = (f"-timing -verify -m={M} -n={N} -k=512 -gpus=1,2 -cus-per-gpu={CUS} -lds-occupancy "
        "-tile-size=8 -gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv")


def knobs():
    p4 = rm.capped_points(4096)[1]
    pt = rm.Point(p4.label, p4.layer, M, N, p4.kg, CUS)
    k = rm.derive_knobs(pt, 0.555, measured=rm.capped_measured())
    return k["link_bw_gbps"], k["link_latency_ns"], k["msg_overhead_ns"]


def launches(plan):
    n, prev = 0, None
    for s in plan:
        if s > 1:
            n += s
        elif prev != 1:
            n += 1
        prev = s
    return n


def link_flags():
    bw, lat, ovh = knobs()
    return f"-link-bw={bw:.4f} -link-latency={lat:.0f} -msg-overhead={ovh:.0f}"


def engine():
    bw, lat, ovh = knobs()
    g = m.Gemm(ROWS, COLS, CUS * 8, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    cm = e.Comm(2, "all-reduce", bw * 1e9, lat, ovh, 4096)
    ones = (1,) * ROWS
    w1 = cm.window(g.run(ones)[0])
    plans = [p for p in m.rows_plans_structured(ROWS) if launches(p) <= 36]
    wp, _, plan = e.search(g, cm, plans, top=1)[0]
    return (w1, ones), (wp, plan), g.run(ones)[1]


FLASH = "-sw-comm-cus=1 -sw-signal-cost=0 -collective-threads=1024"


def flash_best(plan):
    """Best (window, groups) for the plan, Step 9 model on this shape."""
    bw, lat, _ = knobs()
    prm = s9.params()
    s9.ROWS, s9.COLS, s9.ROW_BYTES = ROWS, COLS, 32 * N * 4
    sc = prm.get("wg_scale", 1.0)
    g = s9.FlashGemm(ROWS, COLS, {k: v * sc for k, v in m.SIM_STEADY.items()},
                     {k: v * sc for k, v in m.SIM_FIRST.items()}, m.SIM_LAUNCH, cus=CUS - 1)
    ready, _ = g.run(plan)
    return min((s9.comm(ready, gr, plan, 2, bw * 1e9, lat, prm)[0], gr)
               for gr in s9.partitions(ROWS, first_max=8, last_max=16))


def link_efficiency(run):
    """Per directed link: push and gather bytes over the union of their
    [release, done] intervals x the link bandwidth; mean over links."""
    path = os.path.join(RUNS, run, "stdout.txt")
    text = open(path).read()
    bw = knobs()[0] * 1e9
    rows = {int(g): int(n) for g, n in re.findall(r"group=(\d+) gpu=0 rows=(\d+)", text)}
    links = {}
    for g, kind, i, j, a, b in re.findall(
            r"name=group(\d+)/(push|gather)/(\d+)-(\d+) released=([\d.]+) done=([\d.]+)", text):
        links.setdefault((i, j), []).append((float(a), float(b), rows[int(g)] * N * 4 / 2))
    effs = []
    for ivs in links.values():
        ivs.sort()
        busy, end = 0.0, -1.0
        for a, b, _ in ivs:
            busy += max(0.0, b - max(a, end))
            end = max(end, b)
        effs.append(sum(x for _, _, x in ivs) / (busy * bw))
    return sum(effs) / len(effs)


def parse(run):
    path = os.path.join(RUNS, run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    mm = re.search(r" window=([\d.]+) gemm_wait=([\d.]+)", text)
    wall = re.search(r"wall_s=([\d.]+)", text)
    if not mm:
        return None
    return dict(window=float(mm.group(1)) * 1e6, gemm=float(mm.group(2)) * 1e6,
                ok="Passed!" in text, wall=float(wall.group(1)) if wall else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    bw, lat, ovh = knobs()
    (w1, ones), (wp, plan), gend = engine()
    lines = [f"16 CUs, {M}x{N}, K/GPU 256, 2.0 waves; knobs {link_flags()} "
             f"(4-CU f~0.5: 0.7807 / 441 / 2121); model GEMM end {gend:.1f} us",
             "run | model (us) | what | measured | error | wall s"]
    pred = [("eng_s1", w1, "engine window, s=1"),
            ("eng_plan", wp, f"engine window, plan {m.name(plan)} ({launches(plan)} launches)")]
    runs = [f"eng_s1 {BASE} {link_flags()} -engine-mode=tile",
            f"eng_plan {BASE} {link_flags()} -engine-mode=tile -splitk-plan={m.name(plan)}"]
    lf = link_flags()
    for name, pl in (("fo_s1", ones), ("fo_plan", plan)):
        w, gr = flash_best(pl)
        grs = ",".join(str(32 * n) for n in gr)
        extra = "" if pl == ones else f" -splitk-plan={m.name(pl)}"
        runs.append(f"{name} {BASE} {lf} {FLASH} -sw-flash={grs}{extra}")
        pred.append((name, w, f"flash window, groups {grs}"))
    half = {k: (m.SIM_FIRST[k] + m.SIM_STEADY[k]) / 2 for k in m.SIM_STEADY}
    gh = m.Gemm(ROWS, COLS, CUS * 8, m.SIM_STEADY, half, m.SIM_LAUNCH)
    cm = e.Comm(2, "all-reduce", bw * 1e9, lat, ovh, 4096)
    for name, pl in (("wab_s1", ones), ("wab_plan", plan)):
        extra = "" if pl == ones else f" -splitk-plan={m.name(pl)}"
        runs.append(f"{name} {BASE} {lf} -engine-mode=tile -warmup -warmup-skip-b{extra}")
        pred.append((name, cm.window(gh.run(pl)[0]), "engine window, A and C warm"))
    for name, v, what in pred:
        r = parse(name)
        meas = f"{r['window']:.1f} | {v / r['window'] - 1:+.1%} | {r['wall']}" if r and r["ok"] else "- | - | -"
        lines.append(f"{name} | {v:.1f} | {what} | {meas}")
    r = parse("eng_s1")
    if r and r["ok"]:
        # Achieved f: comm time without overlap (bytes at 0.8 of the link) over
        # comm + this run's GEMM time minus the launch.
        tc = 2 * M * N * 4 / 2 / (0.8 * bw * 1e9) * 1e6
        tg = r["gemm"] - rm.SIM_LAUNCH_US
        lines.append(f"achieved f (s=1 GEMM {r['gemm']:.1f} us): {tc / (tc + tg):.3f}")
    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step10_model.py\n" + "\n".join(runs) + "\n")
    with open(os.path.join(OUT, "model.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
