#!/usr/bin/env python3
"""Step 11: the plan's gain split (GEMM part, overlap part) against CU count,
and the engine model with measured work-group times.

Points (f~0.5, 2 GPUs, cap 8, K/GPU 256, 2.0 waves; each CU does 16 tiles,
so ratio_model.derive_knobs keeps the 4-CU GEMM time and scales the link
with the bytes, the Step 10 rule):
  4 CUs  256x256,  8 x 8 tiles, Step 9 runs (cold, A+C warm, all warm)
  8 CUs  512x256, 16 x 8 tiles (M doubled; N and K as at 4 CUs), new: c8_*
  16 CUs 512x512, 16 x 16 tiles, Step 10 runs (cold, A+C warm), new: w16_*
           (A, B and C warm: -warmup without -warmup-skip-b)

Model "before": the Step 10 engine model unchanged (Step 6 list schedule with
the fixed 4-CU work-group times 172 / 137 us, first round dropped when all
warm and halved when A and C are warm; Step 8 event messages). The 8-CU plan
is its best over the structured 16-row family with at most 36 launches.
Model "after": the same, but each run's first-round and later-round
work-group times are the ones measured in the s = 1 run of the same point and
cache state (wgs.csv, GPU 1, round 1 = first 8 work-groups per CU): split
work-group times scale with them, steady[s] x r2 / 137 and first[s] x r1 / 172.
No other change. s = 1 runs are then in-sample; the plans are the test.

Writes results/step11_firstround/runs.txt (before running) and model.txt.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratio_model as rm  # noqa: E402
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402
import step11_trace as tr  # noqa: E402
import step10_model as s10  # noqa: E402

R = os.path.join(m.ROOT, "results")
OUT = os.path.join(R, "step11_firstround")
COMMON = ("-timing -verify -k=512 -gpus=1,2 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 "
          "-ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv -mem-timeline=mem.csv "
          "-engine-mode=tile")
PLAN4 = (4, 4, 4, 4, 1, 1, 2, 2)
PLAN16 = (4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 4)  # Step 10's model choice


def knobs(M, N, cus):
    if cus == 4:  # the capped f~0.5 point as run since Step 4
        return 0.7807, 441.0, 2121.0
    p4 = rm.capped_points(4096)[1]
    k = rm.derive_knobs(rm.Point(p4.label, p4.layer, M, N, p4.kg, cus), 0.555,
                        measured=rm.capped_measured())
    return round(k["link_bw_gbps"], 4), round(k["link_latency_ns"]), round(k["msg_overhead_ns"])


POINTS = {  # cus: (M, N, rows, cols)
    4: (256, 256, 8, 8), 8: (512, 256, 16, 8), 16: (512, 512, 16, 16)}

# name, cus, cache state, plan, run dir, timed kernel_seq
RUNS = [
    ("4 cold s=1", 4, "cold", None, "step9_baselines/runs/eng_f05_s1", 0),
    ("4 cold plan", 4, "cold", PLAN4, "step9_baselines/runs/eng_f05_plan", 0),
    ("4 A+C warm s=1", 4, "ac", None, "step9_baselines/runs/wab_s1", 1),
    ("4 A+C warm plan", 4, "ac", PLAN4, "step9_baselines/runs/wab_plan", 1),
    ("4 warm s=1", 4, "warm", None, "step9_baselines/runs/warm_s1", 1),
    ("4 warm plan", 4, "warm", PLAN4, "step9_baselines/runs/warm_plan", 1),
    ("8 cold s=1", 8, "cold", None, "step11_firstround/runs/c8_s1", 0),
    ("8 cold plan", 8, "cold", "search", "step11_firstround/runs/c8_plan", 0),
    ("16 cold s=1", 16, "cold", None, "step10_cushare/runs/eng_s1", 0),
    ("16 cold plan", 16, "cold", PLAN16, "step10_cushare/runs/eng_plan", 0),
    ("16 A+C warm s=1", 16, "ac", None, "step10_cushare/runs/wab_s1", 1),
    ("16 A+C warm plan", 16, "ac", PLAN16, "step10_cushare/runs/wab_plan", 1),
    ("16 warm s=1", 16, "warm", None, "step11_firstround/runs/w16_s1", 1),
    ("16 warm plan", 16, "warm", PLAN16, "step11_firstround/runs/w16_plan", 1),
]


def comm(cus):
    bw, lat, ovh = knobs(*POINTS[cus][:2], cus)
    return e.Comm(2, "all-reduce", bw * 1e9, lat, ovh, 4096)


def gemm_before(cus, state):
    _, _, rows, cols = POINTS[cus]
    first = {"cold": m.SIM_FIRST, "warm": None,
             "ac": {k: (m.SIM_FIRST[k] + m.SIM_STEADY[k]) / 2 for k in m.SIM_STEADY}}[state]
    return m.Gemm(rows, cols, cus * 8, m.SIM_STEADY, first, m.SIM_LAUNCH)


def measured_rounds(run, seq):
    k = tr.kernel_rows(run, seq)
    if not k:
        return None
    rows, _ = k
    return (tr.mean(w["dur"] for w in rows if w["round1"]),
            tr.mean(w["dur"] for w in rows if not w["round1"]))


def gemm_after(cus, r1, r2):
    _, _, rows, cols = POINTS[cus]
    steady = {s: v * r2 / m.T for s, v in m.SIM_STEADY.items()}
    first = {s: v * r1 / m.SIM_FIRST[1] for s, v in m.SIM_FIRST.items()}
    return m.Gemm(rows, cols, cus * 8, steady, first, m.SIM_LAUNCH)


def best_plan(g, cus):
    rows = POINTS[cus][2]
    plans = [p for p in m.rows_plans_structured(rows) if s10.launches(p) <= 36]
    return e.search(g, comm(cus), plans, top=1)[0]


def flags(cus, plan, state):
    M, N, _, _ = POINTS[cus]
    bw, lat, ovh = knobs(M, N, cus)
    f = (f"-m={M} -n={N} -cus-per-gpu={cus} {COMMON} -link-bw={bw:.4f} "
         f"-link-latency={lat:.0f} -msg-overhead={ovh:.0f}")
    if state == "warm":
        f += " -warmup"
    if plan:
        f += f" -splitk-plan={m.name(plan)}"
    return f


def parse(run):
    path = os.path.join(R, run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    mm = re.search(r" window=([\d.]+) gemm_wait=([\d.]+)", text)
    if not mm:
        return None
    wall = re.search(r"wall_s=([\d.]+)", text)
    return dict(window=float(mm.group(1)) * 1e6, gemm=float(mm.group(2)) * 1e6,
                ok="Passed!" in text, wall=float(wall.group(1)) if wall else None)


def main():
    os.makedirs(OUT, exist_ok=True)
    L = []
    for cus, (M, N, rows, cols) in POINTS.items():
        bw, lat, ovh = knobs(M, N, cus)
        L.append(f"{cus} CUs: {M}x{N}, {rows}x{cols} tiles, {cus * 8} slots, "
                 f"{rows * cols / (cus * 8):.1f} waves; -link-bw={bw:.4f} -link-latency={lat:.0f} "
                 f"-msg-overhead={ovh:.0f}")
    wp8, _, plan8 = best_plan(gemm_before(8, "cold"), 8)
    L.append(f"8-CU plan (model before, structured 16-row family, <= 36 launches): "
             f"{m.name(plan8)} ({s10.launches(plan8)} launches), window {wp8:.1f} us")

    new = []
    for name, cus, state, plan, run, _ in RUNS:
        if run.startswith("step11_firstround/"):
            pl = plan8 if plan == "search" else plan
            new.append(f"{os.path.basename(run)} {flags(cus, pl, state)}")
    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step11_model.py\n" + "\n".join(new) + "\n"
                 "defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 "
                 "-algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 "
                 "-collective-threads=16384\n")

    L.append("\nrun | plan | measured window | model before | error | s=1 rounds r1 / later (us) | "
             "model after | error | wall s")
    errs = {"before": {}, "after": {}}
    rounds = {}
    for name, cus, state, plan, run, seq in RUNS:
        pl = plan8 if plan == "search" else plan
        ones = (1,) * POINTS[cus][2]
        if pl is None:
            rounds[(cus, state)] = measured_rounds(run, seq)
        cm = comm(cus)
        wb = cm.window(gemm_before(cus, state).run(pl or ones)[0])
        rr = rounds.get((cus, state))
        wa = cm.window(gemm_after(cus, *rr).run(pl or ones)[0]) if rr else None
        r = parse(run)
        meas = r["window"] if r and r["ok"] else None
        cell = lambda v: f"{v:.1f}" if v is not None else "-"
        err = lambda v: f"{v / meas - 1:+.1%}" if v is not None and meas else "-"
        if meas:
            errs["before"][name] = wb / meas - 1
            if wa:
                errs["after"][name] = wa / meas - 1
        L.append(f"{name} | {m.name(pl) if pl else 's=1'} | {cell(meas)}"
                 f"{'' if not r or r['ok'] else ' (NOT VERIFIED)'} | {wb:.1f} | {err(wb)} | "
                 f"{'%.1f / %.1f' % rr if rr else '-'} | {cell(wa)} | {err(wa)} | "
                 f"{r['wall'] if r else '-'}")
    for k, d in errs.items():
        if d:
            for grp, sel in (("s=1", "s=1"), ("plans", "plan")):
                v = [abs(x) for n, x in d.items() if n.endswith(sel)]
                if v:
                    L.append(f"model {k}, {grp}: mean |error| {sum(v) / len(v):.1%}, "
                             f"max {max(v):.1%} ({len(v)} runs)")
    rr = rounds.get((8, "cold"))
    if rr:
        wa8, _, pa8 = best_plan(gemm_after(8, *rr), 8)
        L.append(f"8-CU plan the model after would pick: {m.name(pa8)} (window {wa8:.1f} us)")
    r = parse("step11_firstround/runs/c8_s1")
    if r and r["ok"]:
        bw = knobs(512, 256, 8)[0]
        tc = 2 * 512 * 256 * 4 / 2 / (0.8 * bw * 1e9) * 1e6
        tg = r["gemm"] - rm.SIM_LAUNCH_US
        L.append(f"8-CU achieved f (s=1 GEMM {r['gemm']:.1f} us): {tc / (tc + tg):.3f}")
    d = parse("step11_firstround/runs/defaults_433")
    if d:
        L.append(f"defaults: window {d['window']:.3f} us, verified {d['ok']}")
    with open(os.path.join(OUT, "model.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
