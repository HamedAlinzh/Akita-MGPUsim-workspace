#!/usr/bin/env python3
"""Step 9 summary (results/step9_baselines/summary.txt): the four-way
comparison, the link efficiency of the FlashOverlap-style path, the gain split
(step9_split.py), the cache states, the L1 fix's effect, the model's other
settings and the all-gather model. Reads only the runs' log lines."""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step6_model as m  # noqa: E402
import step9_model as s9  # noqa: E402

R = os.path.join(m.ROOT, "results")
OUT = os.path.join(R, "step9_baselines")
RUNS = os.path.join(OUT, "runs")
OLD_ENGINE = {  # Steps 4b/5b/7, before the L1 fix
    "eng_f05_s1": "step4b_capped/runs/f05_tile", "eng_f05_plan": "step5b_bounds/runs/c_f05_plan",
    "eng_f08_s1": "step4b_capped/runs/f08_tile", "eng_f08_plan": "step5b_bounds/runs/c_f08_plan",
    "eng_x8_s1": "step7_baselines/runs/x8_tile_s1", "eng_x8_plan": "step7_baselines/runs/x8_tile_plan",
    "warm_s1": "step6_paper/runs/warm_f05_s1", "warm_plan": "step6_paper/runs/warm_f05_plan",
}


def window(path):
    p = os.path.join(R, path, "stdout.txt")
    if not os.path.exists(p):
        return None
    text = open(p).read()
    mm = re.search(r" window=([\d.]+) gemm_wait=([\d.]+)", text)
    if not mm:
        return None
    return float(mm.group(1)) * 1e6, float(mm.group(2)) * 1e6, "Passed!" in text


def predictions():
    pred = {}
    for line in open(os.path.join(OUT, "predictions_extra.txt")):
        f = [x.strip() for x in line.split("|")]
        if len(f) == 5 and f[0] and f[0] != "run":
            pred[f[0]] = float(f[1])
    return pred


def link_efficiency(run, key, p):
    """Per directed link: bytes of its push and gather kernels over the union
    of their [release, done] intervals x the link bandwidth; mean over
    links. Also the isolated pushes' own efficiency."""
    r = s9.parse_flash(run)
    if not r or not r.get("ok"):
        return None
    bw = s9.link(key)[0]
    text = open(os.path.join(RUNS, run, "stdout.txt")).read()
    rows = {int(g): int(n) for g, n in re.findall(r"group=(\d+) gpu=0 rows=(\d+)", text)}
    links = {}
    for k, a, b in r["kernels"]:
        mm = re.match(r"group(\d+)/(push|gather)/(\d+)-(\d+)", k)
        if not mm:
            continue
        g, i, j = int(mm.group(1)), int(mm.group(3)), int(mm.group(4))
        share = rows[g] * 256 * 4 / p
        links.setdefault((i, j), []).append((a, b, share))
    effs = []
    for ivs in links.values():
        ivs.sort()
        busy, end = 0.0, -1.0
        for a, b, _ in ivs:
            if a > end:
                busy += b - a
            elif b > end:
                busy += b - end
            end = max(end, b)
        effs.append(sum(x for _, _, x in ivs) / (busy * 1e-6 * bw))
    return sum(effs) / len(effs)


def isolated_push(run, key):
    r = s9.parse_flash(run)
    bw = s9.link(key)[0]
    out = []
    for k, a, b in r["kernels"]:
        if "/push/" in k or "/gather/" in k:
            out.append(4 * s9.ROW_BYTES / ((b - a) * 1e-6 * bw))
    return sum(out) / len(out)


def main():
    pred = predictions()
    L = []
    L.append("1. Four-way comparison (window, us; model error in brackets)")
    L.append("point | FlashOverlap-style s=1 | FlashOverlap-style plan | engine s=1 | engine plan | "
             "plan gain: software / engine | engine vs software (plan)")
    for tag, label in (("f05", "f0.5, 2 GPUs"), ("f08", "f0.8, 2 GPUs"), ("x8", "f0.5, 8 GPUs")):
        cells, vals = [], {}
        for name in (f"fo_{tag}_s1", f"fo_{tag}_plan", f"eng_{tag}_s1", f"eng_{tag}_plan"):
            w = window(f"step9_baselines/runs/{name}")
            if not w:
                cells.append("-")
                continue
            vals[name] = w[0]
            e = f" [{pred[name] / w[0] - 1:+.1%}]" if name in pred else ""
            cells.append(f"{w[0]:.1f}{e}{'' if w[2] else ' (NOT VERIFIED)'}")
        g = lambda a, b: f"{vals[b] / vals[a] - 1:+.1%}" if a in vals and b in vals else "-"
        L.append(f"{label} | " + " | ".join(cells) + f" | {g(f'fo_{tag}_s1', f'fo_{tag}_plan')} / "
                 f"{g(f'eng_{tag}_s1', f'eng_{tag}_plan')} | {g(f'fo_{tag}_plan', f'eng_{tag}_plan')}")
    groups = {}
    for line in open(os.path.join(OUT, "runs.txt")):
        mm = re.match(r"(fo_\S+) .*-sw-flash=(\S+)(?: -splitk-plan=(\S+))?", line)
        if mm:
            groups[mm.group(1)] = (mm.group(2), mm.group(3) or "s=1")
    w = window("step9_baselines/runs/fo_x8_s1_cus2")
    if w:
        L.append(f"  8 GPUs, FlashOverlap-style s=1 with 2 comm CUs: {w[0]:.1f} "
                 f"[model {pred.get('fo_x8_s1_cus2', 0) / w[0] - 1:+.1%}]")
    L.append("  groups (rows) and plan: " + "; ".join(f"{k} {v[0]} / {v[1]}" for k, v in groups.items()))

    L.append("\n2. Link efficiency of the FlashOverlap-style path (bytes / busy time x link "
             "bandwidth; FlashOverlap Sec. 3.2.2: one 192 KB tile message gets 13% on 4 RTX 4090s)")
    L.append(f"  isolated 128 KB shares (cal_one, 1 comm CU): {isolated_push('cal_one', 'f05'):.3f}; "
             f"with 2 comm CUs (cal_one_cus2): {isolated_push('cal_one_cus2', 'f05'):.3f}")
    for run, key, p in (("cal_rows", "f05", 2), ("fo_f05_s1", "f05", 2), ("fo_f05_plan", "f05", 2),
                        ("fo_f08_s1", "f08", 2), ("fo_f08_plan", "f08", 2),
                        ("fo_x8_s1", "f05_8gpu", 8), ("fo_x8_plan", "f05_8gpu", 8),
                        ("fo_x8_s1_cus2", "f05_8gpu", 8)):
        e = link_efficiency(run, key, p)
        if e:
            L.append(f"  {run}: {e:.3f} of the link while a push or gather is in flight")

    L.append("\n3. Gain split (scripts/step9_split.py, split.txt)")
    sys.stdout.flush()
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "step9_split.py")],
                         capture_output=True, text=True, check=True).stdout
    L.extend("  " + x for x in out.strip().splitlines())

    L.append("\n4. Cache states, f0.5, 2 GPUs, engine (us)")
    L.append("state | s=1 | plan | change")
    for label, a, b in (("cold", "eng_f05_s1", "eng_f05_plan"), ("A and C warm, B cold", "wab_s1", "wab_plan"),
                        ("all warm", "warm_s1", "warm_plan")):
        wa, wb = window(f"step9_baselines/runs/{a}"), window(f"step9_baselines/runs/{b}")
        if wa and wb:
            L.append(f"{label} | {wa[0]:.1f} | {wb[0]:.1f} | {wb[0] / wa[0] - 1:+.1%}")

    L.append("\n5. Effect of the L1 invalidate fix (same flags, before -> after)")
    for name in sorted(os.listdir(os.path.join(OUT, "runs_prefix"))):
        a, b = window(f"step9_baselines/runs_prefix/{name}"), window(f"step9_baselines/runs/{name}")
        if a and b:
            L.append(f"  {name}: {a[0]:.1f} -> {b[0]:.1f} ({b[0] / a[0] - 1:+.2%}), verified "
                     f"{a[2]} -> {b[2]}")
    for name, old in OLD_ENGINE.items():
        a, b = window(old), window(f"step9_baselines/runs/{name}")
        if a and b:
            L.append(f"  {old} -> {name}: {a[0]:.1f} -> {b[0]:.1f} ({b[0] / a[0] - 1:+.2%})")

    L.append("\n6. Model: run predictions, calibration, other settings (model.txt)")
    L.extend("  " + x for x in open(os.path.join(OUT, "model.txt")).read().strip().splitlines())

    L.append("\n7. All-gather before the GEMM (model only, allgather.txt)")
    L.extend("  " + x for x in open(os.path.join(OUT, "allgather.txt")).read().strip().splitlines())

    d = window("step9_baselines/runs/defaults_433")
    if d:
        L.append(f"\ndefaults: window {d[0]:.3f} us, verified {d[2]}")
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
