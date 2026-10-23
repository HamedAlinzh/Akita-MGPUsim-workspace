#!/usr/bin/env python3
"""Step 8 item 1a: where the time of one mccl ring all-reduce call goes, from
the work-group trace of step7_baselines/runs/sw_s1 (2 GPUs, two 128-row
bands, -collective-threads 16384). A 2-GPU ring call is three kernels per
GPU, each launched after the host drained the previous one on both GPUs:
push (reduce-scatter), reduce, push (all-gather). Per call:
  gap before each kernel = from the previous step's end on the slower GPU
      (for the first kernel, the later of the band's GEMM end and the last
      call's end) to this kernel's first
      dispatch: the launch's three host copies (3 x 7.58 us at the default
      -h2d-cycles) plus the rest (driver, CP acquire, dispatch);
  wait for the slower GPU = each step's end spread across the two GPUs;
  kernel time = first dispatch to last work-group end, on the slower GPU;
  transfer = the push's bytes at the link's measured 0.794 x -link-bw."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step5_wgs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "results", "step7_baselines", "runs", "sw_s1")
LAUNCH_COPIES = 3 * 7.583
LINK = 0.794 * 0.7807e9
BAND_BYTES = 128 * 256 * 4


def main():
    rows = step5_wgs.load(os.path.join(RUN, "wgs.csv"))
    k = collections.defaultdict(list)
    for r in rows:
        k[(r["gpu"], int(r["kernel_id"].split("@")[0]))].append(r)
    per_gpu = collections.defaultdict(list)
    for (gpu, kid), ws in k.items():
        per_gpu[gpu].append(dict(kid=kid, mccl=ws[0]["grid_y"] == "1",
                                 start=min(w["start"] for w in ws), end=max(w["end"] for w in ws)))
    for gpu in per_gpu:
        per_gpu[gpu].sort(key=lambda x: x["start"])
    gemm = {g: [x for x in v if not x["mccl"]] for g, v in per_gpu.items()}
    coll = {g: [x for x in v if x["mccl"]] for g, v in per_gpu.items()}
    names = ["push (reduce-scatter)", "reduce", "push (all-gather)"]
    print("call | step | gap before (us) | of which launch copies | rest | kernel (us) | "
          "slower-GPU wait (us)")
    totals = collections.Counter()
    prev = 0.0
    for call in range(len(gemm["1"])):
        # The host starts a call once its band is done and the last call returned.
        prev = max(prev, max(gemm[g][call]["end"] for g in gemm))
        start = prev
        for step in range(3):
            ks = [coll[g][call * 3 + step] for g in coll]
            first = min(x["start"] for x in ks)
            end = max(x["end"] for x in ks)
            spread = end - min(x["end"] for x in ks)
            gap = first - prev
            kern = end - first
            print(f"{call + 1} | {names[step]} | {gap:.1f} | {LAUNCH_COPIES:.1f} | "
                  f"{gap - LAUNCH_COPIES:.1f} | {kern:.1f} | {spread:.1f}")
            totals["launch copies"] += LAUNCH_COPIES
            totals["other launch / sync"] += gap - LAUNCH_COPIES
            totals["push kernels" if step != 1 else "reduce kernel"] += kern
            prev = end
        totals["calls"] += 1
        totals["total"] += prev - start
    n = totals["calls"]
    transfer = (BAND_BYTES / 2) / LINK * 1e6
    print(f"\nper call (mean of {n}): total {totals['total'] / n:.1f} us = launch copies "
          f"{totals['launch copies'] / n:.1f} + other launch/sync {totals['other launch / sync'] / n:.1f} "
          f"+ push kernels {totals['push kernels'] / n:.1f} + reduce kernel "
          f"{totals['reduce kernel'] / n:.1f}")
    print(f"pure transfer in the two pushes: 2 x {transfer:.1f} = {2 * transfer:.1f} us "
          f"({BAND_BYTES // 2} B each at {LINK / 1e9:.3f} GB/s); push kernels beyond it: "
          f"{totals['push kernels'] / n - 2 * transfer:.1f} us")


if __name__ == "__main__":
    main()
