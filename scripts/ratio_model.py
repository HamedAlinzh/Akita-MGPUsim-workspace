#!/usr/bin/env python3
"""
Ratio-matched workload sizing for the fused GEMM + collective study.

We can only simulate small shapes, and at small sizes the communication share
can look very different from a real system. So we match ratios, not sizes:

  comm fraction  f = t_comm / (t_gemm + t_comm)

computed for the GEMM that produces a collective's input (the part a
hardware-triggered design overlaps), with both phases serialized. The script

  1. estimates f for real tensor-parallel and MoE layers on an MI300X-class
     8-GPU node,
  2. models f for small shapes on the simulated R9 Nano, calibrated on our own
     measurements, and
  3. proposes simulated configurations (shape, CUs per GPU, link bandwidth)
     whose predicted f matches each real layer, with an estimated wall time.

Everything printed as "predicted" is a model. Step 4 measures one proposal.

Usage:  python3 scripts/ratio_model.py [--tokens 4096] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Target system: AMD Instinct MI300X, 8 GPUs fully connected.
# ---------------------------------------------------------------------------
# Peak dense BF16: 1307.4 TFLOP/s per GPU (AMD MI300X platform specs; also
#   SemiAnalysis chip page). https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html
# Interconnect: seven Infinity Fabric links per GPU form a fully connected
#   8-GPU node; 896 GB/s total aggregate bidirectional peer-to-peer bandwidth
#   per GPU (AMD platform page; ROCm MI300 microarchitecture page), i.e.
#   128 GB/s bidirectional = 64 GB/s per direction per link.
#   https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
# Efficiencies are assumptions, not measurements: large BF16 GEMMs on MI300X
#   commonly reach well under peak, and collectives under link peak.
REAL_PEAK_FLOPS = 1307.4e12
REAL_LINK_BW = 64e9          # bytes/s per direction per link
REAL_GEMM_EFF = 0.70         # assumed achieved fraction of peak FLOP/s
REAL_LINK_EFF = 0.80         # assumed achieved fraction of link bandwidth
BF16 = 2                     # bytes per element


def real_out_bw(p: int) -> float:
    """Per-GPU outbound bandwidth to the other p-1 members of a TP/EP group.
    In a fully connected node each peer is one direct link away."""
    return (p - 1) * REAL_LINK_BW * REAL_LINK_EFF


@dataclass
class Model:
    name: str
    hidden: int
    ffn: int


# Hugging Face config.json values (Llama-3.x 70B: 8192 / 28672, 80 layers;
# Llama-3.1 405B: 16384 / 53248, 126 layers; Mixtral-8x7B: 4096 / 14336,
# 8 experts, top-2).
LLAMA70 = Model("Llama-3 70B", 8192, 28672)
LLAMA405 = Model("Llama-3.1 405B", 16384, 53248)
MIXTRAL = Model("Mixtral 8x7B", 4096, 14336)
MIXTRAL_EXPERTS, MIXTRAL_TOPK = 8, 2


@dataclass
class RealLayer:
    layer: str
    gemm: str            # the GEMM producing (or consuming) the collective
    collective: str
    p: int
    t_gemm: float        # seconds
    t_comm: float        # seconds

    @property
    def frac(self) -> float:
        return self.t_comm / (self.t_gemm + self.t_comm)


def gemm_time(m: int, n: int, k: int) -> float:
    return 2 * m * n * k / (REAL_PEAK_FLOPS * REAL_GEMM_EFF)


def real_layers(tokens: int) -> list[RealLayer]:
    out = []
    for model in (LLAMA70, LLAMA405):
        h, f = model.hidden, model.ffn
        for p in (2, 4, 8):
            msg = tokens * h * BF16                     # T x H activations
            ar = 2 * (p - 1) / p * msg / real_out_bw(p)  # all-reduce
            rs = (p - 1) / p * msg / real_out_bw(p)      # reduce-scatter (SP)
            # Row-parallel GEMMs that end each sublayer: K is split by p.
            for gemm, k in (("attn out-proj", h // p), ("MLP down-proj", f // p)):
                t = gemm_time(tokens, h, k)
                out.append(RealLayer(model.name, gemm, "all-reduce", p, t, ar))
                out.append(RealLayer(model.name, gemm, "reduce-scatter", p, t, rs))

    # MoE: Mixtral 8x7B, expert parallel over 8 GPUs, uniform routing.
    p, h, f = 8, MIXTRAL.hidden, MIXTRAL.ffn
    routed = tokens * MIXTRAL_TOPK                   # token-expert pairs per GPU
    a2a = (p - 1) / p * routed * h * BF16 / real_out_bw(p)
    t_up = gemm_time(routed, 2 * f, h)              # gate + up projections
    t_down = gemm_time(routed, h, f)                # down projection
    out.append(RealLayer(MIXTRAL.name, "expert gate+up (consumer)",
                         "dispatch all-to-all", p, t_up, a2a))
    out.append(RealLayer(MIXTRAL.name, "expert down-proj (producer)",
                         "combine all-to-all", p, t_down, a2a))
    return out


# ---------------------------------------------------------------------------
# Simulated system: MGPUSim R9 Nano (64 CUs, 1 GHz), PCIe 4.0 x16 links.
# ---------------------------------------------------------------------------
# Calibration: measured one-band GEMM times (simulated us, tileSize 8, per GPU)
# from results/overlap_sweep*/ and results/stage2_h2d/: (M, N, K per GPU, us).
CALIBRATION = [
    (256, 256, 512, 266.0),
    (256, 256, 1024, 530.0),
    (256, 256, 2048, 1247.0),
    (512, 512, 256, 192.0),
]
SIM_CUS = 64
SIM_TILE = 32               # tileSize 8 -> 32x32 output tile per work-group
SIM_LINK_NOMINAL = 32 * 2**30  # Akita PCIe 4.0 x16: 16 Gb/s x 16 / 8 per direction
# Measured: a 1 MB All-Reduce on 2 GPUs (16384 threads) wrote 1 MB each way in
# 52.7 us of link-busy time = 19.9 GB/s per direction, 58% of nominal. The
# fused engine's efficiency is unknown; this is the best measured stand-in.
SIM_LINK_EFF = 19.9e9 / SIM_LINK_NOMINAL
FP32 = 4


# Measured fixed cost of one GEMM launch: three host-to-device setup copies
# of 7.58 us each at the default -h2d-cycles (results/allreduce_breakdown/).
SIM_LAUNCH_US = 22.75


def fit_sim_gemm():
    """t = t0 + Kg * r * load, load = max(1, WGs-per-CU / w_sat).

    Simulated GEMM time grows faster than linearly in K (larger B, more L2
    misses), so a straight-line fit over all points has a negative intercept.
    The proposals all sit at small K, so t0 is pinned to the measured launch
    cost and r comes from the smallest-K point with one work-group per CU;
    the model then underestimates large-K points. w_sat comes from the 512^2
    point (four work-groups per CU)."""
    pts = [(kg, t) for m, n, kg, t in CALIBRATION if m * n // SIM_TILE**2 == SIM_CUS]
    kg_min, t_min = min(pts)
    t0 = SIM_LAUNCH_US
    r = (t_min - t0) / kg_min
    m, nn, kg, t = next(c for c in CALIBRATION if c[0] == 512)
    wgs_per_cu = m * nn / SIM_TILE**2 / SIM_CUS
    load = (t - t0) / (r * kg)
    w_sat = wgs_per_cu / max(load, 1.0) if load > 1 else wgs_per_cu
    return t0, r, w_sat


T0, R_PER_K, W_SAT = fit_sim_gemm()


def sim_gemm_us(m: int, n: int, kg: int, cus: int) -> float:
    wgs_per_cu = m * n / SIM_TILE**2 / cus
    return T0 + R_PER_K * kg * max(1.0, wgs_per_cu / W_SAT)


def sim_comm_us(m: int, n: int, p: int, link_bw: float) -> float:
    """Fused reduce-scatter + all-gather: each GPU sends 2(p-1)/p of C."""
    return 2 * (p - 1) / p * m * n * FP32 / (link_bw * SIM_LINK_EFF) * 1e6


def sim_wall_s(m: int, n: int, k: int) -> float:
    """Uncontended wall time, fit to standalone runs: 256x256x1024 (2 GPUs)
    about 70 s, 512^3 (2 GPUs) about 180 s. Runs sharing the machine take
    2-10x longer."""
    mmac = m * n * k / 1e6
    return 10 + 1.25 * mmac


def needed_link_bw(m, n, kg, cus, p, frac):
    """Nominal link bandwidth (bytes/s) that makes the predicted f equal frac."""
    t_g = sim_gemm_us(m, n, kg, cus)
    t_c = frac / (1 - frac) * t_g
    return 2 * (p - 1) / p * m * n * FP32 / (t_c * 1e-6) / SIM_LINK_EFF


def akita_link(bw: float):
    """Akita's PCIe model moves round(bw / freq) bytes per cycle. Return the
    network clock and flit size that realize bw, preferring the 1 GHz default."""
    for freq in (1e9, 500e6, 250e6, 100e6):
        flit = round(bw / freq)
        if flit >= 4 and abs(flit * freq - bw) / bw < 0.15:
            return freq, flit, flit * freq
    flit = max(1, round(bw / 100e6))
    return 100e6, flit, flit * 100e6


# Candidate small shapes: 2 GPUs, tileSize 8, few minutes of wall time.
CANDIDATES = [
    # (M, N, K total, CUs per GPU)
    (128, 128, 256, 16),
    (256, 256, 512, 64),
    (256, 256, 1024, 64),
    (256, 256, 512, 16),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096,
                    help="tokens per GPU per step (fractions are nearly independent of it)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    layers = real_layers(args.tokens)
    print(f"Simulated GEMM model: t = {T0:.1f} us + {R_PER_K:.3f} us x K-per-GPU "
          f"x max(1, WGs-per-CU / {W_SAT:.1f})")
    for m, n, kg, t in CALIBRATION:
        print(f"  calibration {m}x{n}, K/GPU {kg}: measured {t:.0f} us, "
              f"model {sim_gemm_us(m, n, kg, SIM_CUS):.0f} us")
    print(f"Simulated link: nominal {SIM_LINK_NOMINAL/1e9:.1f} GB/s per direction, "
          f"efficiency {SIM_LINK_EFF:.2f} (measured)\n")

    print("Candidate simulated shapes at the default link (2 GPUs, tileSize 8):")
    for m, n, k, cus in CANDIDATES:
        tg = sim_gemm_us(m, n, k // 2, cus)
        tc = sim_comm_us(m, n, 2, SIM_LINK_NOMINAL)
        print(f"  {m}x{n}x{k}, {cus} CUs: GEMM {tg:.0f} us, comm {tc:.1f} us, "
              f"f = {tc/(tg+tc):.3f}, wall ~{sim_wall_s(m, n, k):.0f} s")
    print()

    # Proposals use one base shape so the only knob that moves is the link.
    bm, bn, bk, bcus = 256, 256, 512, 64
    rows = []
    for L in layers:
        bw = needed_link_bw(bm, bn, bk // 2, bcus, 2, L.frac)
        freq, flit, real_bw = akita_link(bw)
        tg = sim_gemm_us(bm, bn, bk // 2, bcus)
        tc = sim_comm_us(bm, bn, 2, real_bw)
        rows.append({
            "layer": L.layer, "gemm": L.gemm, "collective": L.collective, "tp": L.p,
            "real_gemm_us": round(L.t_gemm * 1e6, 1),
            "real_comm_us": round(L.t_comm * 1e6, 1),
            "real_frac": round(L.frac, 3),
            "sim_shape": f"{bm}x{bn}x{bk}", "sim_gpus": 2, "sim_cus": bcus,
            "sim_link_gbps": round(real_bw / 1e9, 2),
            "sim_net_freq_mhz": int(freq / 1e6), "sim_flit_bytes": flit,
            "sim_frac_pred": round(tc / (tg + tc), 3),
            "wall_s_est": round(sim_wall_s(bm, bn, bk)),
        })

    # In a fully connected node, per-GPU bandwidth grows with p-1 while the
    # per-GPU GEMM shrinks with p, so f does not depend on the TP size here;
    # rows that differ only in TP are printed once.
    hdr = ("layer", "gemm", "collective", "tp", "real_frac", "sim_shape",
           "sim_cus", "sim_link_gbps", "sim_net_freq_mhz", "sim_flit_bytes",
           "sim_frac_pred", "wall_s_est")
    print(" | ".join(hdr))
    seen = {}
    for r in rows:
        key = (r["layer"], r["gemm"], r["collective"])
        seen.setdefault(key, []).append(r)
    for group in seen.values():
        r = dict(group[0])
        r["tp"] = ",".join(str(g["tp"]) for g in group)
        print(" | ".join(str(r[h]) for h in hdr))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
