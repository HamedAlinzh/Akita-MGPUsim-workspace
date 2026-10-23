#!/usr/bin/env python3
"""Step 14: do the all-gather results (Steps 12-13) hold at other link speeds
and at 4 GPUs? Same GEMM (256 x 256 x 256 per GPU, 4 CUs, tileSize 8, cap 8).

Configurations: 8 GPUs at f 0.35 and f 0.8, 4 GPUs at f 0.5. Knobs from
ratio_model.derive_knobs as in the design note (step9_allgather.knobs: bytes
per link = one GPU's A shard, reduce-scatter rule, capped GEMM time), so f =
t_comm / (t_comm + t_gemm). Per configuration: push then GEMM (bulk), per
row s = 1, per row with the plan the queue model picks (Step 13 model with
the partial-sum pass, every remote row s in {1, 2, 4}, local rows 1), chosen
before running. Floor: Step 12's g8_local; a new g4_local for 4 GPUs.

The model indexes the plan by row of GPU 0 (peer q's rows over link q); the
benchmark's -allgather-plan is by position in each GPU's arrival order
(local rows, then the k-th row of every peer in peer order), converted here.

Writes results/step14_allgather_sweep/runs.txt and plans.txt (before
running) and summary.txt.
"""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step9_allgather as ag  # noqa: E402
import step12_allgather as s12  # noqa: E402
import step13_allgather as s13  # noqa: E402

OUT = os.path.join(m.ROOT, "results", "step14_allgather_sweep")
CONFIGS = [("g8_f035", 8, 0.35), ("g8_f08", 8, 0.8), ("g4_f05", 4, 0.5)]
_knobs = ag.knobs


def use_f(f):
    ag.knobs = lambda p, _f=0.5: _knobs(p, f)


def knobs(p, f):
    bw, lat, ovh = _knobs(p, f)
    return round(bw / 1e9, 4), round(lat), round(ovh)


def achieved_f(p, f):
    import ratio_model as rm
    pt = rm.capped_points(4096)[1]
    pt = rm.Point(pt.label, rm.RealLayer(pt.layer.model, pt.layer.gemm, "reduce-scatter",
                                         pt.layer.p, pt.layer.tokens), pt.m, pt.n, pt.kg, pt.cus)
    k = rm.derive_knobs(pt, f, measured=rm.capped_measured(), num_gpu=p)
    return k["t_comm_us"] / (k["t_comm_us"] + k["t_gemm_us"]), k


def arrival_order(p):
    per = ag.ROWS // p
    return list(range(per)) + [q * per + k for k in range(per) for q in range(1, p)]


def best_plan(p, f):
    use_f(f)
    per = ag.ROWS // p
    best = None
    for ss in itertools.product((1, 2, 4), repeat=ag.ROWS - per):
        plan = (1,) * per + ss
        w = s13.model(p, plan)
        if best is None or w < best[0]:
            best = (w, plan)
    return best


def flags(p, f, mode, plan=None):
    bw, lat, ovh = knobs(p, f)
    fl = (f"-gpus={','.join(str(i + 1) for i in range(p))} {s12.COMMON} "
          f"-link-bw={bw} -link-latency={lat} -msg-overhead={ovh}")
    if p > 2:
        fl += " -peer-mesh"
    if mode in ("bulk", "rows"):
        fl += " -engine-mode=tile" + (" -engine-peer-queues" if p > 2 else "")
    if mode in ("bulk", "local"):
        fl += f" -allgather-{mode}"
    if plan:
        fl += " -allgather-plan=" + ",".join(str(plan[r]) for r in arrival_order(p))
    return fl


def parse(run):
    path = os.path.join(OUT, "runs", run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    mm = re.search(r"allgather phases: mode=\w+ window=([\d.]+)", text) or \
        re.search(r"allreducegemm phases: window=([\d.]+)", text)
    w = re.search(r"wall_s=([\d.]+)", text)
    return {"ok": "Passed!" in text, "window": float(mm.group(1)) * 1e6 if mm else None,
            "wall": float(w.group(1)) if w else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    L, runs, plans = [], [], {}
    for name, p, f in CONFIGS:
        fa, k = achieved_f(p, f)
        bw, lat, ovh = knobs(p, f)
        w, plan = best_plan(p, f)
        plans[name] = plan
        use_f(f)
        ones = (1,) * ag.ROWS
        alone = ag.gemm(ones, {(r, 0): 0.0 for r in range(ag.ROWS)})
        pred = {"bulk": max(ag.arrivals(p, ones).values()) + alone,
                "s1": s13.model(p, ones), "plan": w}
        L.append(f"{name}: {p} GPUs, target f {f}: -link-bw {bw} -link-latency {lat} "
                 f"-msg-overhead {ovh} (t_comm {k['t_comm_us']:.1f} us, t_gemm {k['t_gemm_us']:.1f} "
                 f"us, f {fa:.3f}); model plan (by row) {s13.name(plan)}, as -allgather-plan "
                 f"{','.join(str(plan[r]) for r in arrival_order(p))}; model: push-then-GEMM "
                 f"{pred['bulk']:.1f}, s = 1 {pred['s1']:.1f}, plan {pred['plan']:.1f} us")
        plans[name] = (plan, pred)
        runs += [(f"{name}_bulk", flags(p, f, "bulk")), (f"{name}_s1", flags(p, f, "rows")),
                 (f"{name}_plan", flags(p, f, "rows", plan))]
    runs.append(("g4_local", flags(4, 0.5, "local")))
    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step14_allgather_sweep.py\n")
        for n, fl in runs:
            fh.write(f"{n} {fl}\n")
        fh.write("defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 "
                 "-algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 "
                 "-collective-threads=16384\n")
    with open(os.path.join(OUT, "plans.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    floor8 = s12.parse("g8_local")
    floors = {8: floor8["window"] if floor8 else None,
              4: (parse("g4_local") or {}).get("window")}
    L.append("\nconfig | push-then-GEMM | s = 1 | plan | floor | plan vs s = 1 | plan vs push-then-GEMM | "
             "plan above floor | model plan / error | model s = 1 / error | model push-then-GEMM / "
             "error | plan (by row) | verified | wall s")
    for name, p, f in CONFIGS:
        plan, pred = plans[name]
        r = {mode: parse(f"{name}_{mode}") for mode in ("bulk", "s1", "plan")}
        fl = floors[p]
        if not all(x and x["window"] for x in r.values()):
            L.append(f"{name} | (not all run)")
            continue
        b, s1, pl = (r[k]["window"] for k in ("bulk", "s1", "plan"))
        e = lambda k: f"{pred[k]:.1f} / {r[k]['window'] / pred[k] - 1:+.1%}"
        L.append(f"{name} | {b:.1f} | {s1:.1f} | {pl:.1f} | {fl:.1f} | {pl / s1 - 1:+.1%} | "
                 f"{pl / b - 1:+.1%} | {pl / fl - 1:+.1%} | {e('plan')} | {e('s1')} | {e('bulk')} | "
                 f"{s13.name(plan)} | {all(x['ok'] for x in r.values())} | "
                 f"{', '.join(str(x['wall']) for x in r.values())}")
    g4 = parse("g4_local")
    if g4:
        L.append(f"g4_local (floor, 4 GPUs): {g4['window']:.1f} us, verified {g4['ok']}, model "
                 f"{ag.gemm((1,) * ag.ROWS, {(r, 0): 0.0 for r in range(ag.ROWS)}):.1f}")
    d = parse("defaults_433")
    if d and d["window"]:
        L.append(f"defaults: window {d['window']:.3f} us, verified {d['ok']}")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
