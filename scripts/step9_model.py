#!/usr/bin/env python3
"""Step 9 model: the FlashOverlap-style software baseline (-sw-flash).

GEMM: the Step 8 processor-sharing schedule (up to 8 resident work-groups
per CU, ordered dispatch, cold first round unless warm) on the 4 - comm_cus
CUs left: the comm CUs are taken from the comm kernels' launch, before the
GEMM's, until the last finishes (FlashOverlap gives communication its SMs
first; its waiting kernel occupies them). FlashGemm can also close them only
during given intervals, the rule of the first two calibration runs
(runs_dynamic/), where a GEMM work-group left on the comm CU straggled.

Communication, per group of tile rows (every GPU alike): the group signal
comes when its last tile is ready (+ the signal cost); the comm CU adds the
group's split-K partials (one accumulate kernel per extra K chunk of each
split row, one after another), then pushes each peer its share; the owner
adds the p-1 shares one after another (each reduce waits for its push and the
peer signal: signal cost + link latency) and pushes the reduced share to every
peer. Each push takes a fixed cost and then sends its bytes at eta of the
link; pushes on the same link (one direction, any group) go one at a time in
request order. The window ends at the last gather.

Constants (push fixed cost, eta, reduce and accumulate costs) are fitted to
the calibration runs cal_one and cal_rows (`--fit`); until those exist, a
priori values are used (eta 0.50 from the mccl fit, 5 us per kernel).

Usage: step9_model.py            predictions, run list, measured vs model
       step9_model.py --fit      also refit the constants from the cal runs
"""
import heapq
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step8_model as e  # noqa: E402

ROOT = m.ROOT
OUT = os.path.join(ROOT, "results", "step9_baselines")
RUNS = os.path.join(OUT, "runs")
PARAMS = os.path.join(OUT, "params.json")
ROWS, COLS = 8, 8
ROW_BYTES = 32 * 256 * 4  # one row of tiles of C
PLAN = (4, 4, 4, 4, 1, 1, 2, 2)
PRIOR = dict(push_k=5.0, eta=0.50, red_k=5.0, red_gbps=20.0, acc_k=5.0, acc_gbps=20.0,
             wg_scale=1.0)


BASE = ("-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 "
        "-gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv")
FLASH = "-sw-comm-cus=1 -sw-signal-cost=0 -collective-threads=1024"
X2 = "-gpus=1,2 -k=512"
X8 = "-gpus=1,2,3,4,5,6,7,8 -k=2048 -peer-mesh"


def link(key):
    bw, lat, ovh = m.knobs()[key]
    return bw, lat, ovh, f"-link-bw={bw / 1e9:g} -link-latency={lat:g} -msg-overhead={ovh:g}"


def params():
    if os.path.exists(PARAMS):
        return json.load(open(PARAMS))
    return dict(PRIOR)


class FlashGemm(e.PSGemm):
    """PSGemm whose last comm_cus CUs take no new work-group during busy
    intervals (absolute us, launch included)."""

    def run(self, plan, busy=(), comm_cus=1):
        jobs = [(row * self.cols + col, s) for row, s in enumerate(plan)
                for _ in range(s) for col in range(self.cols)]
        active = [[] for _ in range(self.cus)]
        ready = [0.0] * (self.rows * self.cols)
        bounds = sorted({x - self.launch for iv in busy for x in iv})
        t, j = 0.0, 0

        def closed(c):
            return c >= self.cus - comm_cus and any(
                a <= t + self.launch + 1e-9 < b for a, b in busy)

        while j < len(jobs) or any(active):
            while j < len(jobs):
                open_cus = [c for c in range(self.cus)
                            if not closed(c) and len(active[c]) < self.cap]
                if not open_cus:
                    break
                cu = min(open_cus, key=lambda c: len(active[c]))
                tile, s = jobs[j]
                w = self.first[s] if self.first and t == 0 else self.steady[s]
                active[cu].append([w, tile])
                j += 1
            steps = [a[0] * len(act) / self.cap for act in active for a in act]
            steps += [b - t for b in bounds if b > t + 1e-9]
            dt = min(steps)
            for act in active:
                rate = self.cap / len(act) if act else 0
                for a in act:
                    a[0] -= dt * rate
            t += dt
            for act in active:
                for a in [a for a in act if a[0] <= 1e-9]:
                    act.remove(a)
                    ready[a[1]] = max(ready[a[1]], t)
        return [self.launch + r for r in ready], self.launch + t


def gemm(warm=False, comm_cus=0, scale=1.0):
    steady = {s: v * scale for s, v in m.SIM_STEADY.items()}
    first = None if warm else {s: v * scale for s, v in m.SIM_FIRST.items()}
    return FlashGemm(ROWS, COLS, steady, first, m.SIM_LAUNCH, cus=4 - comm_cus)


def comm(ready, groups, plan, p, bw, lat_ns, prm, sig_ns=0.0, detail=False):
    """Window (us) and comm-busy intervals. ready: per tile; groups: tile
    rows per group."""
    sig, lat = sig_ns / 1e3, lat_ns / 1e3
    # Several groups in flight share the comm CU: the link runs at eta_conc
    # and reduces move their bytes red_slow times slower (fitted to cal_rows).
    multi = len(groups) > 1
    eta = prm.get("eta_conc", prm["eta"]) if multi else prm["eta"]
    red_rate = prm["red_gbps"] / (prm.get("red_slow", 1.0) if multi else 1.0)
    row_ready = [max(ready[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    free = 0.0  # one directed link; every link carries the same traffic
    heap, out, r0 = [], [], 0
    seq = itertools.count()
    for g, n in enumerate(groups):
        rows = range(r0, r0 + n)
        start = max(row_ready[r] for r in rows) + sig
        acc = sum((plan[r] - 1) * (prm["acc_k"] + ROW_BYTES / (prm["acc_gbps"] * 1e3))
                  for r in rows)
        share = n * ROW_BYTES / p
        heapq.heappush(heap, (start + acc + prm["push_k"], next(seq), g, "push", share, start))
        r0 += n
    end = {}
    while heap:
        t, _, g, stage, share, start = heapq.heappop(heap)
        x = share / (eta * bw) * 1e6
        done = max(t, free) + x
        free = done
        if stage == "push":
            red = (p - 1) * (prm["red_k"] + share / (red_rate * 1e3))
            gather = done + sig + lat + red + sig + prm["push_k"]
            heapq.heappush(heap, (gather, next(seq), g, "gather", share, start))
        else:
            end[g] = (start, done)
            out.append((start, done))
    window = max(d for _, d in out)
    if detail:
        return window, out, end
    return window, out


def flash(groups, plan, p, key, warm=False, prm=None, sig_ns=0.0, comm_cus=1):
    """Window and GEMM end, the GEMM on the CUs comm does not hold."""
    prm = prm or params()
    bw, lat, _, _ = link(key)
    ready, end = gemm(warm, comm_cus, prm.get("wg_scale", 1.0)).run(plan)
    return comm(ready, groups, plan, p, bw, lat, prm, sig_ns)[0], end


def partitions(rows=ROWS, first_max=8, last_max=16):
    """Every split of the rows into consecutive groups, with FlashOverlap's
    rule (first group at most 2 waves, last at most 4; a wave is 4 rows here,
    32 resident work-groups of 8 tiles a row), which 8 rows never break."""
    for cuts in itertools.product((0, 1), repeat=rows - 1):
        groups, n = [], 1
        for c in cuts:
            if c:
                groups.append(n)
                n = 0
            n += 1
        groups.append(n)
        if groups[0] <= first_max and groups[-1] <= last_max:
            yield tuple(groups)


def plan_candidates(top=40):
    """Plans worth trying: the engine model's best 40 at f~0.5 and all-ones."""
    g = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    cm = e.knob_comm("f05")
    best = e.search(g, cm, itertools.product((1, 2, 4), repeat=8), top=top)
    return [pl for _, _, pl in best]


def best_flash(p, key, plans, prm, warm=False):
    """Best (window, partition, plan) with one comm CU."""
    bw, lat, _, _ = link(key)
    g = gemm(warm, 1, prm.get("wg_scale", 1.0))
    coarse = []
    for plan in plans:
        ready, _ = g.run(plan)
        for groups in partitions():
            w, _ = comm(ready, groups, plan, p, bw, lat, prm)
            coarse.append((w, groups, plan))
    coarse.sort()
    return coarse[0]


def gname(groups):
    return ",".join(str(32 * n) for n in groups)


def parse_flash(run):
    path = os.path.join(RUNS, run, "stdout.txt")
    if not os.path.exists(path):
        return None
    text = open(path).read()
    mm = re.search(r"flash phases: window=([\d.]+) gemm_wait=([\d.]+)", text)
    if not mm:
        return dict(ok=False)
    kern = [(k, float(a) * 1e6, float(b) * 1e6) for k, a, b in re.findall(
        r"flash kernel: name=(\S+) released=([\d.]+) done=([\d.]+)", text)]
    return dict(ok="Passed!" in text, window=float(mm.group(1)) * 1e6,
                gemm=float(mm.group(2)) * 1e6, kernels=kern)


def fit():
    """From the two calibration runs. eta: the isolated 128 KB pushes of
    cal_one (one group, after the GEMM); reduce rate: its reduces; GEMM
    work-group scale on 3 CUs: cal_one's GEMM end. Then the push and reduce
    fixed costs are the grid point that best fits both windows."""
    one, rows = parse_flash("cal_one"), parse_flash("cal_rows")
    bw = link("f05")[0]
    # On 3 CUs the simulator's dispatch left one CU a full last round (24,
    # 20, 20 work-groups), which the balanced model misses: scale every
    # work-group so the model's GEMM end matches cal_one's.
    end1 = gemm(False, 1).run((1,) * ROWS)[1]
    wg_scale = (one["gemm"] - m.SIM_LAUNCH) / (end1 - m.SIM_LAUNCH)

    def mean(r, kind):
        v = [b - a for k, a, b in r["kernels"] if f"/{kind}/" in k]
        return sum(v) / len(v)
    p_big, r_big = mean(one, "push"), mean(one, "red")
    big = 4 * ROW_BYTES
    best = None
    for push_k in [x * 0.5 for x in range(0, 61)]:
        for red_k in [x * 0.5 for x in range(0, 61)]:
            if red_k >= r_big - 1:
                continue
            prm = dict(PRIOR, push_k=push_k, red_k=red_k, acc_k=red_k,
                       eta=big / (bw * (p_big - push_k) * 1e-6),
                       red_gbps=big / ((r_big - red_k) * 1e3), wg_scale=wg_scale)
            prm["acc_gbps"] = prm["red_gbps"]
            err = sum((flash(g, (1,) * ROWS, 2, "f05", prm=prm)[0] / r["window"] - 1) ** 2
                      for g, r in (((8,), one), ((1,) * 8, rows)))
            if best is None or err < best[0]:
                best = (err, prm)
    prm = best[1]
    # Contention, from cal_rows: the link's efficiency while a push or gather
    # is in flight, then the extra cost per reduce that fits its window.
    import step9_summary
    prm["eta_conc"] = step9_summary.link_efficiency("cal_rows", "f05", 2)
    prm["red_slow"] = min((abs(flash((1,) * 8, (1,) * ROWS, 2, "f05",
                                     prm=dict(prm, red_slow=x))[0] - rows["window"]), x)
                          for x in [1 + i * 0.5 for i in range(0, 200)])[1]
    json.dump(prm, open(PARAMS, "w"), indent=1)
    return prm, dict(push_big=p_big, red_big=r_big, gather_big=mean(one, "gather"))


def main():
    os.makedirs(OUT, exist_ok=True)
    refit = "--fit" in sys.argv
    lines = []
    if refit:
        prm, raw = fit()
        lines.append("Calibration (cal_one: one 256-row group; cal_rows: eight 32-row groups; "
                     "kernels released after the GEMM):")
        lines.append("  cal_one: push of a 128 KB share %.1f us, gather %.1f us, reduce %.1f us"
                     % (raw["push_big"], raw["gather_big"], raw["red_big"]))
        lines.append("  fit: push fixed %.1f us, eta %.3f of the link; reduce fixed %.1f us + "
                     "%.2f GB/s; GEMM work-groups x%.3f on 3 CUs" % (
                         prm["push_k"], prm["eta"], prm["red_k"], prm["red_gbps"],
                         prm["wg_scale"]))
        lines.append("  several groups in flight (cal_rows): eta %.3f, reduces %.1fx slower" % (
            prm["eta_conc"], prm["red_slow"]))
    prm = params()
    fitted = os.path.exists(PARAMS)
    lines.append("constants: " + ", ".join(f"{k} {v:.3g}" for k, v in prm.items()) +
                 ("" if fitted else " (a priori)"))

    f05, f08, f8 = link("f05")[3], link("f08")[3], link("f05_8gpu")[3]
    ones = (1,) * ROWS
    pred, runs = [], []

    def add(name, flags, value, what):
        runs.append(f"{name} {flags}")
        pred.append((name, value, what))

    # Stage 1: calibration, GEMM alone with the plan, A+C warm, defaults.
    add("cal_one", f"{BASE} {X2} {f05} {FLASH} -sw-flash=256",
        flash((8,), ones, 2, "f05", prm=prm)[0], "flash window")
    add("cal_rows", f"{BASE} {X2} {f05} {FLASH} -sw-flash=" + gname((1,) * 8),
        flash((1,) * 8, ones, 2, "f05", prm=prm)[0], "flash window")
    add("cal_one_cus2", f"{BASE} {X2} {f05} {FLASH} -sw-flash=256".replace(
        "-sw-comm-cus=1", "-sw-comm-cus=2"),
        flash((8,), ones, 2, "f05", prm=prm, comm_cus=2)[0], "flash window")
    plan_flag = "-splitk-plan=" + m.name(PLAN)
    add("gemm_plan", f"{BASE} {X2} {f05} -engine-mode=bulk {plan_flag}",
        gemm().run(PLAN)[1], "GEMM end")
    add("gemm_plan_warm", f"{BASE} {X2} {f05} -engine-mode=bulk -warmup {plan_flag}",
        gemm(warm=True).run(PLAN)[1], "GEMM end")
    # A+C warm, B cold: half of the cold first round's extra time (A and B
    # are read in equal bytes per work-group).
    half = {s: (m.SIM_FIRST[s] + m.SIM_STEADY[s]) / 2 for s in m.SIM_STEADY}
    gh = m.Gemm(8, 8, 32, m.SIM_STEADY, half, m.SIM_LAUNCH)
    cm = e.knob_comm("f05")
    for name, plan in (("wab_s1", ones), ("wab_plan", PLAN)):
        flags = f"{BASE} {X2} {f05} -engine-mode=tile -warmup -warmup-skip-b"
        if plan != ones:
            flags += " " + plan_flag
        add(name, flags, cm.window(gh.run(plan)[0]), "engine window")
    warm = m.Gemm(8, 8, 32, m.SIM_STEADY, None, m.SIM_LAUNCH)
    for name, plan in (("warm_s1", ones), ("warm_plan", PLAN)):
        flags = f"{BASE} {X2} {f05} -engine-mode=tile -warmup"
        if plan != ones:
            flags += " " + plan_flag
        add(name, flags, cm.window(warm.run(plan)[0]), "engine window")
    # Engine reruns (same flags as Steps 4b/5b/7) with the Step 8 event model
    # and the Step 6 list schedule, as in step8_validate.
    cold = m.Gemm(8, 8, 32, m.SIM_STEADY, m.SIM_FIRST, m.SIM_LAUNCH)
    for tag, xs, fl, key, p, pq in (("f05", X2, f05, "f05", 2, False),
                                    ("f08", X2, f08, "f08", 2, False),
                                    ("x8", X8, f8, "f05_8gpu", 8, True)):
        cm = e.knob_comm(key, p, per_peer=pq)
        extra = " -engine-peer-queues" if pq else ""
        for name, plan in ((f"eng_{tag}_s1", ones), (f"eng_{tag}_plan", PLAN)):
            flags = f"{BASE} {xs} {fl} -engine-mode=tile{extra}"
            if plan != ones:
                flags += " " + plan_flag
            add(name, flags, cm.window(cold.run(plan)[0]), "engine window")
    runs.append("defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 "
                "-algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 "
                "-collective-threads=16384")
    pred.append(("defaults_433", 432.742, "window"))

    # Stage 2 (once fitted): the FlashOverlap-style runs, partition and plan
    # picked by the model.
    choice = {}
    if fitted:
        cands = plan_candidates()
        for tag, key, p, xs, fl in (("f05", "f05", 2, X2, f05), ("f08", "f08", 2, X2, f08),
                                    ("x8", "f05_8gpu", 8, X8, f8)):
            w1, g1, _ = best_flash(p, key, [ones], prm)
            wp, gp, pp = best_flash(p, key, cands, prm)
            choice[tag] = ((w1, g1), (wp, gp, pp))
            add(f"fo_{tag}_s1", f"{BASE} {xs} {fl} {FLASH} -sw-flash={gname(g1)}", w1,
                f"flash window, groups {gname(g1)}")
            add(f"fo_{tag}_plan", f"{BASE} {xs} {fl} {FLASH} -sw-flash={gname(gp)} "
                f"-splitk-plan={m.name(pp)}", wp,
                f"flash window, groups {gname(gp)}, plan {m.name(pp)}")

    if fitted:
        g8 = choice["x8"][0][1]
        add("fo_x8_s1_cus2", f"{BASE} {X8} {f8} {FLASH} -sw-flash={gname(g8)}".replace(
            "-sw-comm-cus=1", "-sw-comm-cus=2"),
            flash(g8, ones, 8, "f05_8gpu", prm=prm, comm_cus=2)[0], "flash window, 2 comm CUs")

    with open(os.path.join(OUT, "runs.txt"), "w") as fh:
        fh.write("# written by scripts/step9_model.py\n" + "\n".join(runs) + "\n")

    lines.append("\nrun | model (us) | what | measured | error")
    for name, value, what in pred:
        r = parse_flash(name) if name.startswith(("cal_", "fo_")) else None
        meas = None
        if r and r.get("ok"):
            meas = r["window"]
        elif not name.startswith(("cal_", "fo_")):
            path = os.path.join(RUNS, name, "stdout.txt")
            if os.path.exists(path):
                text = open(path).read()
                mm = re.search(r" window=([\d.]+) gemm_wait=([\d.]+)", text)
                if mm and "Passed!" in text:
                    meas = float(mm.group(2 if what == "GEMM end" else 1)) * 1e6
        err = f"{value / meas - 1:+.1%}" if meas else "-"
        lines.append(f"{name} | {value:.1f} | {what} | {meas and round(meas, 1) or '-'} | {err}")

    if fitted:
        lines.append("\nOther settings (model): window at f~0.5 on 2 GPUs, best partition and plan")
        for cus in (1, 2):
            for sig in (0.0, 5000.0, 20000.0):
                w1 = min(flash(gr, (1,) * 8, 2, "f05", prm=prm, sig_ns=sig, comm_cus=cus)[0]
                         for gr in partitions())
                wp = min(flash(gr, pl, 2, "f05", prm=prm, sig_ns=sig, comm_cus=cus)[0]
                         for gr in partitions() for pl in cands[:10])
                lines.append(f"  -sw-comm-cus {cus}, -sw-signal-cost {sig / 1e3:g} us: "
                             f"s=1 {w1:.1f}, plan {wp:.1f} ({wp / w1 - 1:+.1%})")
    with open(os.path.join(OUT, "model.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
