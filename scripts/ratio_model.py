#!/usr/bin/env python3
"""
Ratio-matched workload sizing for the fused GEMM + collective study.

We can only simulate small shapes, so we match ratios, not sizes. Three
ratios shape the results we care about, and each is matched separately:

  f          comm fraction t_comm / (t_gemm + t_comm) of the GEMM that produces
             (or consumes) a collective, both phases serialized;
  L / X      per-message latency over per-message transfer time, for a message
             of one output tile (sets whether merging tiles, -tiles-per-msg,
             pays off);
  waves      output tiles per GPU / (CUs x resident work-groups per CU) (sets
             when tiles become ready and how long the tail is).

The script prints

  1. f and waves for real tensor-parallel layers on an MI300X 8-GPU node,
  2. the three representative points, real vs simulated, with the simulator
     knobs (-link-bw, -link-latency, and the send engine's -msg-overhead)
     derived from the ratios,
  3. sensitivity of real f to the unsourced efficiencies, and of the knobs to
     the latency range,
  4. MoE all-to-all rows, including small-token (decode-like) and skewed
     routing. Analysis only.

Simulated GEMM times are measured (1 GPU, tile tracker, -lds-occupancy) and
read from results/step1_tracker/. Everything else is a model.

Usage:  python3 scripts/ratio_model.py [--tokens 4096] [--outdir results/step0_ratio]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Target system: AMD Instinct MI300X, 8 GPUs fully connected.
# ---------------------------------------------------------------------------
# Peak dense BF16: 1307.4 TFLOP/s per GPU (AMD MI300X platform specs).
#   https://www.amd.com/en/products/accelerators/instinct/mi300/platform.html
# Interconnect: seven Infinity Fabric links per GPU, fully connected; 896 GB/s
#   aggregate bidirectional peer bandwidth per GPU, i.e. 64 GB/s per direction
#   per link (AMD platform page; ROCm MI300 microarchitecture page).
#   https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html
# HBM3: 5.3 TB/s per GPU (AMD MI300X product page).
# GEMM, link and HBM efficiencies are assumptions, not measurements; see the
#   sensitivity section.
REAL_PEAK_FLOPS = 1307.4e12
REAL_LINK_BW = 64e9          # bytes/s per direction per link
REAL_HBM_BW = 5.3e12
REAL_GEMM_EFF = 0.70
REAL_LINK_EFF = 0.80
REAL_HBM_EFF = 0.80
BF16 = 2

# Waves: output tiles per GPU / (CUs x resident work-groups per CU).
# MI300X has 304 CUs, 64 KiB LDS and a 512 KiB VGPR file per CU (ROCm "GPU
#   hardware specifications" table).
#   https://rocm.docs.amd.com/en/docs-6.3.3/reference/gpu-arch-specs.html
# Residency of a 256x256 BF16 output tile is our estimate, not a measurement:
#   its FP32 accumulators are 256 KiB, half the CU's VGPR file, and staging a
#   256x64 BF16 slice of A and of B takes 64 KiB, all of the LDS. So one
#   work-group per CU (two at most, with a smaller K slice).
REAL_CUS = 304
REAL_TILE = 256
REAL_RESIDENT_WGS = 1

# Per-message latency. MI300A shares MI300X's Infinity Fabric generation.
#   "Inter-APU Communication on AMD MI300A Systems via Infinity Fabric: a Deep
#   Dive" (MEMSYS 2025, arXiv 2508.11298), Sec. 5.1-5.2:
#   - a GPU pointer-chasing load takes 346 ns from local HBM and 690 ns from a
#     neighbour's HBM, so one Infinity Fabric crossing adds (690-346)/2 = 172 ns
#     one way: the wire latency;
#   - a hipMemcpy (DMA) between APUs costs about 1 us for any size from 1 B to
#     128 KB: the per-message cost of a DMA transfer, wire included.
#   DMA-Latte (arXiv 2511.06605, Sec. III-B3) agrees in kind: DMA command
#   creation, scheduling and sync are up to ~60% of a small transfer's time.
# The per-message engine overhead is the rest: 1000 - 172 = 828 ns. The range
#   for sensitivity is 690 ns (the remote-load round trip, a lower bound for a
#   hardware-triggered engine) to 2 us.
REAL_WIRE_NS = (690 - 346) / 2
REAL_MSG_LATENCY_NS = 1000.0
REAL_MSG_LATENCY_RANGE_NS = (690.0, 1000.0, 2000.0)


def real_out_bw(p: int, link_eff: float = REAL_LINK_EFF) -> float:
    """Per-GPU outbound bandwidth to the other p-1 members of a TP/EP group.
    In a fully connected node each peer is one direct link away."""
    return (p - 1) * REAL_LINK_BW * link_eff


def gemm_time(m: int, n: int, k: int, eff: float = REAL_GEMM_EFF) -> float:
    return 2 * m * n * k / (REAL_PEAK_FLOPS * eff)


def real_waves(m: int, n: int, resident: int = REAL_RESIDENT_WGS) -> float:
    tiles = math.ceil(m / REAL_TILE) * math.ceil(n / REAL_TILE)
    return tiles / (REAL_CUS * resident)


def first_tile_share(waves: float) -> float:
    """Share of the GEMM's time before the first tiles finish, if every
    round of resident work-groups takes the same time."""
    return 1 / math.ceil(waves)


def real_tile_transfer_ns() -> float:
    """One message of one output tile: 256x256 BF16 partial sums, over one
    link."""
    return REAL_TILE * REAL_TILE * BF16 / (REAL_LINK_BW * REAL_LINK_EFF) * 1e9


@dataclass
class Model:
    name: str
    hidden: int
    ffn: int


# Hugging Face config.json values (Llama-3.x 70B: 8192 / 28672; Llama-3.1
# 405B: 16384 / 53248; Mixtral-8x7B: 4096 / 14336, 8 experts, top-2).
LLAMA70 = Model("Llama-3 70B", 8192, 28672)
LLAMA405 = Model("Llama-3.1 405B", 16384, 53248)
MIXTRAL = Model("Mixtral 8x7B", 4096, 14336)
MIXTRAL_EXPERTS, MIXTRAL_TOPK = 8, 2


@dataclass
class RealLayer:
    model: Model
    gemm: str            # "attn out-proj" or "MLP down-proj"
    collective: str      # "all-reduce" or "reduce-scatter"
    p: int
    tokens: int

    @property
    def k(self) -> int:
        full = self.model.hidden if self.gemm == "attn out-proj" else self.model.ffn
        return full // self.p

    def t_gemm(self, gemm_eff: float = REAL_GEMM_EFF) -> float:
        return gemm_time(self.tokens, self.model.hidden, self.k, gemm_eff)

    def t_comm(self, link_eff: float = REAL_LINK_EFF) -> float:
        msg = self.tokens * self.model.hidden * BF16
        share = 2 * (self.p - 1) / self.p if self.collective == "all-reduce" \
            else (self.p - 1) / self.p
        return share * msg / real_out_bw(self.p, link_eff)

    def frac(self, gemm_eff=REAL_GEMM_EFF, link_eff=REAL_LINK_EFF) -> float:
        g, c = self.t_gemm(gemm_eff), self.t_comm(link_eff)
        return c / (g + c)

    @property
    def waves(self) -> float:
        # Row-parallel GEMM: every GPU produces the full tokens x hidden output.
        return real_waves(self.tokens, self.model.hidden)


def real_layers(tokens: int) -> list[RealLayer]:
    return [RealLayer(m, g, c, p, tokens)
            for m in (LLAMA70, LLAMA405)
            for p in (2, 4, 8)
            for g in ("attn out-proj", "MLP down-proj")
            for c in ("all-reduce", "reduce-scatter")]


# ---------------------------------------------------------------------------
# Simulated system: MGPUSim R9 Nano CUs at 1 GHz, 2 GPUs, tileSize 8.
# ---------------------------------------------------------------------------
SIM_TILE_SIZE = 8
SIM_TILE = 4 * SIM_TILE_SIZE  # 32x32 FP32 output tile = one 4 KiB message
FP32 = 4
SIM_GPUS = 2
# Resident work-groups per CU for mmmKernel_local with -lds-occupancy, from
# TestGEMMResidency (timing/cp/internal/resource): 74 VGPRs limit a SIMD to 3
# wavefronts, before LDS does. tileSize 4 and 8 -> 12, tileSize 16 -> 3.
SIM_RESIDENT_WGS = {4: 12, 8: 12, 16: 3}
# Measured fixed cost of one kernel launch (three setup copies of 7.58 us at
# the default -h2d-cycles, results/allreduce_breakdown/). The tracker times
# start before the launch, so this is subtracted to get the GEMM's own time.
SIM_LAUNCH_US = 22.75
# Efficiency of the fused send engine on the peer fabric. Unknown until the
# engine exists (Step 2); assumed equal to the real link's, to be measured.
SIM_FABRIC_EFF = 0.80
SIM_FABRIC_FLIT = 16          # bytes; fabric clock = bandwidth / flit
TRACKER_DIR = os.path.join(ROOT, "results", "step1_tracker")


def sim_waves(m: int, n: int, cus: int, tile_size: int = SIM_TILE_SIZE) -> float:
    edge = 4 * tile_size
    return (m // edge) * (n // edge) / (cus * SIM_RESIDENT_WGS[tile_size])


def tracker_times(m: int, n: int, kg: int, cus: int):
    """(GEMM us, first-tile share) measured by the tile tracker on 1 GPU."""
    path = os.path.join(TRACKER_DIR,
                        f"tiles_cu{cus}_M{m}_N{n}_K{kg}_t{SIM_TILE_SIZE}.csv")
    with open(path) as f:
        done = sorted(float(r["complete_us"]) for r in csv.DictReader(f))
    gemm = done[-1] - SIM_LAUNCH_US
    return gemm, (done[0] - SIM_LAUNCH_US) / gemm


@dataclass
class Point:
    label: str
    layer: RealLayer
    m: int
    n: int
    kg: int      # K per GPU
    cus: int


def points(tokens: int) -> list[Point]:
    return [
        Point("f~0.8", RealLayer(LLAMA70, "attn out-proj", "all-reduce", 8, tokens),
              384, 256, 256, 4),
        Point("f~0.5", RealLayer(LLAMA70, "MLP down-proj", "all-reduce", 8, tokens),
              384, 256, 256, 4),
        Point("f~0.25", RealLayer(LLAMA405, "MLP down-proj", "reduce-scatter", 8, tokens),
              512, 320, 256, 4),
    ]


def sim_comm_bytes(pt: Point) -> float:
    """Bytes each simulated GPU sends: the engine's reduce-scatter sends
    (p-1)/p of C, and the all-gather sends as much again."""
    c = pt.m * pt.n * FP32
    share = (SIM_GPUS - 1) / SIM_GPUS
    return c * share * (2 if pt.layer.collective == "all-reduce" else 1)


def derive_knobs(pt: Point, f: float, latency_ns: float = REAL_MSG_LATENCY_NS):
    """Simulator knobs that match f and the per-message latency ratio.

    1. t_comm = f / (1 - f) x measured GEMM time; the link must move this
       GPU's bytes in that time at SIM_FABRIC_EFF -> -link-bw.
    2. One 4 KiB tile message then takes X_sim = 4096 / (bw x eff). The real
       ratios are wire / X_real and overhead / X_real for one 128 KiB tile;
       multiplying each by X_sim gives -link-latency and -msg-overhead.
    3. The fabric clock is bw / 16 B flit; -link-latency is realized in whole
       cycles of it."""
    t_g, first_share = tracker_times(pt.m, pt.n, pt.kg, pt.cus)
    t_c = f / (1 - f) * t_g
    bw_eff = sim_comm_bytes(pt) / (t_c * 1e-6)
    bw = bw_eff / SIM_FABRIC_EFF
    x_sim = SIM_TILE * SIM_TILE * FP32 / bw_eff * 1e9
    x_real = real_tile_transfer_ns()
    wire = REAL_WIRE_NS / x_real * x_sim
    overhead = (latency_ns - REAL_WIRE_NS) / x_real * x_sim
    clock = bw / SIM_FABRIC_FLIT
    cycles = max(1, round(wire * clock / 1e9))
    return dict(t_gemm_us=t_g, first_share=first_share, t_comm_us=t_c,
                link_bw_gbps=bw / 1e9, msg_transfer_ns=x_sim,
                link_latency_ns=wire, msg_overhead_ns=overhead,
                fabric_clock_mhz=clock / 1e6, latency_cycles=cycles,
                realized_latency_ns=cycles / clock * 1e9,
                ratio=(wire + overhead) / x_sim)


# ---------------------------------------------------------------------------
# Tile-size mapping and the matrix-core compute curve (Step 3).
# ---------------------------------------------------------------------------
# Real GEMM efficiency vs output tile. The only measured AMD matrix-core tile
# comparison found: HipKittens (arXiv 2511.08083, Table 2), BF16 8192^3 on
# MI355X (CDNA4, not MI300X): 256x256 -> 1610 TFLOPS, 192x256 -> 1281,
# 128x256 -> 893 (different producer/consumer split). Efficiency relative to
# 256x256 follows the tile's arithmetic intensity I = MT*NT/(MT+NT) as
# (I / I_256)^alpha with alpha = 1.48 and 1.45 from the two points; we use
# 1.47. 128x128 is an extrapolation, so the curve counts as assumed and is
# bracketed by alpha = 1 (flatter) and 2 (steeper).
HK_POINTS = [((192, 256), 1281 / 1610), ((128, 256), 893 / 1610)]
CURVE_ALPHAS = {"flatter": 1.0, "matrix-core": 1.47, "steeper": 2.0}


def intensity(mt: int, nt: int) -> float:
    return mt * nt / (mt + nt)


def fitted_alphas():
    ref = intensity(256, 256)
    return [math.log(eff) / math.log(intensity(*t) / ref) for t, eff in HK_POINTS]


def rel_time(edge: int, alpha: float) -> float:
    """GEMM time with edge x edge tiles over time with 256x256 tiles."""
    return (intensity(256, 256) / intensity(edge, edge)) ** alpha


# Sim tileSize -> real tile edge. tileSize 8 = 256x256 is the matched point;
# tileSize 4 = 128x128 halves the edge in both. tileSize 16 would be 512x512,
# whose FP32 accumulators (1 MiB) exceed the CU's 512 KiB VGPR file, so no
# MI300X kernel uses it; tileSize 2 = 64x64 runs about 16x slower than
# tileSize 8 natively (4 of 64 lanes busy), beyond the run budget.
TILE_MAP = {4: 128, 8: 256}
# Real resident work-groups per CU (estimate, as for 256x256 above): a
# 128x128 tile has 64 KiB of accumulators and stages 2 x 128x64 BF16 = 32 KiB
# in LDS, so LDS allows 2.
REAL_RESIDENT_BY_EDGE = {128: 2, 256: 1}
# Measured simulated GEMM times (1 GPU, 384x256, K 256, 4 CUs,
# -lds-occupancy; results/step3_tiles/calib/): tileSize -> {valu scale: us}.
SIM_GEMM_US = {4: {1.0: 2700.3, 0.9: 2430.4, 0.6: 1643.3, 0.4: 1183.8},
               8: {1.0: 646.3}, 16: {1.0: 624.0}}


def valu_scale_for(tile_size: int, target_us: float) -> float:
    """Linear fit of measured GEMM time against -valu-cycles-scale."""
    pts = sorted(SIM_GEMM_US[tile_size].items())
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
    return (target_us - (my - slope * mx)) / slope


def waves_at_edge(m: int, n: int, edge: int) -> float:
    tiles = math.ceil(m / edge) * math.ceil(n / edge)
    return tiles / (REAL_CUS * REAL_RESIDENT_BY_EDGE[edge])


def tile_rows(tokens: int, pt: "Point"):
    """Per mapped tile size and curve: waves, one-tile L/X and f, real vs
    simulated, with the f~0.5 point's link knobs held fixed (they describe
    the network, not the tile)."""
    k = derive_knobs(pt, pt.layer.frac())
    lat_ns = k["link_latency_ns"] + k["msg_overhead_ns"]
    bw_eff = k["link_bw_gbps"] * 1e9 * SIM_FABRIC_EFF
    rows = []
    for ts, edge in TILE_MAP.items():
        real_x = edge * edge * BF16 / (REAL_LINK_BW * REAL_LINK_EFF) * 1e9
        sim_x = ((4 * ts) ** 2 * FP32 + 16) / bw_eff * 1e9
        tiles_sim = (pt.m // (4 * ts)) * (pt.n // (4 * ts))
        msgs = tiles_sim  # per direction at N=1 (all-reduce, 2 GPUs)
        comm_us = msgs * max(sim_x, k["msg_overhead_ns"]) / 1e3
        for curve in ("native",) + tuple(CURVE_ALPHAS):
            if curve == "native":
                scale, rel = 1.0, SIM_GEMM_US[ts][1.0] / SIM_GEMM_US[8][1.0]
            else:
                rel = rel_time(edge, CURVE_ALPHAS[curve])
                scale = 1.0 if ts == 8 else valu_scale_for(ts, rel * SIM_GEMM_US[8][1.0])
            alpha = CURVE_ALPHAS.get(curve, CURVE_ALPHAS["matrix-core"])
            g_real = pt.layer.t_gemm() * rel_time(edge, alpha)
            c_real = pt.layer.t_comm()
            g_sim = rel * SIM_GEMM_US[8][1.0] - SIM_LAUNCH_US
            rows.append(dict(
                sim_tile_size=ts, real_tile=f"{edge}x{edge}", curve=curve,
                valu_scale=round(scale, 3), sim_gemm_rel=round(rel, 2),
                real_waves=round(waves_at_edge(pt.layer.tokens, pt.layer.model.hidden,
                                               edge), 2),
                sim_waves=round(sim_waves(pt.m, pt.n, pt.cus, ts), 2),
                real_LX=round(REAL_MSG_LATENCY_NS / real_x, 2),
                sim_LX=round(lat_ns / sim_x, 2),
                real_f=round(c_real / (g_real + c_real), 3) if curve != "native" else "",
                sim_f_n1=round(comm_us / (g_sim + comm_us), 3)))
    return rows


# ---------------------------------------------------------------------------
# MoE all-to-all (analysis only).
# ---------------------------------------------------------------------------
# Expert parallel over 8 GPUs, Mixtral 8x7B: one expert per GPU. Each GPU
# routes tokens x top-2 token-expert pairs; uniform routing sends 7/8 of them
# to other GPUs. Under skew s the busiest GPU receives s times the mean, and
# it sets the layer's time. The expert GEMM time is the larger of compute and
# reading the expert's weights from HBM: decode-sized batches are weight-bound.
# All-to-all latency per collective: 1 us for a hardware-triggered DMA (the
# MI300A per-copy cost above) or 20 us for a host-launched library collective
# (MI300A RCCL point-to-point minimum, same paper, Sec. 6.1.1). The inter-node
# rows assume one 400 Gb/s NIC per GPU (50 GB/s) and 5 us latency: an
# assumption for contrast, not sourced.
MOE_TOKENS = (4096, 256, 64, 16)
MOE_SKEWS = (1, 4)
MOE_LATENCIES_US = (1.0, 20.0)


def moe_rows(gemm_eff=REAL_GEMM_EFF):
    p, h, ffn, k = 8, MIXTRAL.hidden, MIXTRAL.ffn, MIXTRAL_TOPK
    hbm = REAL_HBM_BW * REAL_HBM_EFF
    flops = REAL_PEAK_FLOPS * gemm_eff
    fabrics = [("intra-node IF", (p - 1) * REAL_LINK_BW * REAL_LINK_EFF, MOE_LATENCIES_US),
               ("inter-node 400G", 50e9 * REAL_LINK_EFF, (5.0,))]
    rows = []
    for fabric, in_bw, lats in fabrics:
        for tokens in MOE_TOKENS:
            for skew in MOE_SKEWS:
                for lat in lats:
                    pairs = tokens * k * skew              # into the busiest GPU
                    t_a2a = lat * 1e-6 + pairs * h * BF16 * (p - 1) / p / in_bw
                    up_compute = 2 * pairs * 2 * ffn * h / flops
                    t_up = max(up_compute, 2 * h * ffn * BF16 / hbm)
                    t_down = max(2 * pairs * h * ffn / flops, h * ffn * BF16 / hbm)
                    rows.append(dict(
                        fabric=fabric, tokens_per_gpu=tokens, skew=skew,
                        a2a_latency_us=lat,
                        a2a_us=round(t_a2a * 1e6, 1),
                        gate_up_us=round(t_up * 1e6, 1),
                        down_us=round(t_down * 1e6, 1),
                        weight_bound=t_up > up_compute,
                        f_dispatch=round(t_a2a / (t_a2a + t_up), 3),
                        f_combine=round(t_a2a / (t_a2a + t_down), 3)))
    return rows


# ---------------------------------------------------------------------------

def table(rows, cols):
    out = [" | ".join(cols)]
    for r in rows:
        out.append(" | ".join(str(r[c]) for c in cols))
    return "\n".join(out)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096,
                    help="tokens per GPU per step (sets waves; f is independent of it)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "step0_ratio"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Real layers.
    real = []
    for L in real_layers(args.tokens):
        real.append(dict(layer=L.model.name, gemm=L.gemm, collective=L.collective,
                         tp=L.p, real_gemm_us=round(L.t_gemm() * 1e6, 1),
                         real_comm_us=round(L.t_comm() * 1e6, 1),
                         real_frac=round(L.frac(), 3),
                         real_waves=round(L.waves, 2),
                         real_waves_2res=round(real_waves(args.tokens, L.model.hidden, 2), 2)))
    write_csv(os.path.join(args.outdir, "real_layers.csv"), real)
    print(f"1. Real layers, MI300X, {args.tokens} tokens per GPU "
          f"(f does not depend on TP in a fully connected node; TP 8 shown)")
    print(table([r for r in real if r["tp"] == 8],
                ["layer", "gemm", "collective", "real_frac", "real_waves",
                 "real_waves_2res"]))

    x_real = real_tile_transfer_ns()
    print(f"\nReal one-tile message: {REAL_TILE}x{REAL_TILE} BF16 = "
          f"{REAL_TILE * REAL_TILE * BF16 // 1024} KiB, {x_real:.0f} ns at "
          f"{REAL_LINK_BW * REAL_LINK_EFF / 1e9:.1f} GB/s; latency "
          f"{REAL_MSG_LATENCY_NS:.0f} ns = wire {REAL_WIRE_NS:.0f} + engine "
          f"{REAL_MSG_LATENCY_NS - REAL_WIRE_NS:.0f}; L/X = "
          f"{REAL_MSG_LATENCY_NS / x_real:.2f}")

    # 2. Representative points.
    rep = []
    for pt in points(args.tokens):
        f = pt.layer.frac()
        k = derive_knobs(pt, f)
        rep.append(dict(
            point=pt.label, layer=f"{pt.layer.model.name} {pt.layer.gemm}",
            collective=pt.layer.collective,
            real_f=round(f, 3),
            sim_f=round(k["t_comm_us"] / (k["t_comm_us"] + k["t_gemm_us"]), 3),
            real_LX=round(REAL_MSG_LATENCY_NS / x_real, 2), sim_LX=round(k["ratio"], 2),
            real_waves=round(pt.layer.waves, 2),
            sim_waves=round(sim_waves(pt.m, pt.n, pt.cus), 2),
            real_first_share=round(first_tile_share(pt.layer.waves), 2),
            sim_first_share=round(k["first_share"], 2),
            sim_shape=f"{pt.m}x{pt.n}, K/GPU {pt.kg}", sim_cus=pt.cus,
            sim_gemm_us=round(k["t_gemm_us"], 1), sim_comm_us=round(k["t_comm_us"], 1),
            link_bw_gbps=round(k["link_bw_gbps"], 4),
            fabric_clock_mhz=round(k["fabric_clock_mhz"], 3),
            msg_transfer_ns=round(k["msg_transfer_ns"]),
            link_latency_ns=round(k["link_latency_ns"]),
            latency_cycles=k["latency_cycles"],
            realized_latency_ns=round(k["realized_latency_ns"]),
            msg_overhead_ns=round(k["msg_overhead_ns"])))
    write_csv(os.path.join(args.outdir, "points.csv"), rep)
    print("\n2. Representative points, real vs simulated (2 GPUs, tileSize 8, N=1)")
    print(table(rep, ["point", "layer", "collective", "real_f", "sim_f", "real_LX",
                      "sim_LX", "real_waves", "sim_waves", "real_first_share",
                      "sim_first_share"]))
    print("\nKnobs:")
    print(table(rep, ["point", "sim_shape", "sim_cus", "sim_gemm_us", "sim_comm_us",
                      "link_bw_gbps", "fabric_clock_mhz", "msg_transfer_ns",
                      "link_latency_ns", "latency_cycles", "realized_latency_ns",
                      "msg_overhead_ns"]))

    # 3. Sensitivity.
    effs_g, effs_l = (0.50, 0.70, 0.85), (0.60, 0.80, 0.90)
    sens = [dict(point=pt.label, gemm_eff=ge, link_eff=le,
                 real_f=round(pt.layer.frac(ge, le), 3))
            for pt in points(args.tokens) for ge in effs_g for le in effs_l]
    write_csv(os.path.join(args.outdir, "sensitivity_eff.csv"), sens)
    print("\n3a. Real f vs assumed efficiencies (rows: GEMM eff; columns: link eff)")
    for pt in points(args.tokens):
        print(f"  {pt.label} ({pt.layer.model.name} {pt.layer.gemm} {pt.layer.collective})")
        print("    gemm\\link  " + "   ".join(f"{le:.2f}" for le in effs_l))
        for ge in effs_g:
            vals = [pt.layer.frac(ge, le) for le in effs_l]
            print(f"    {ge:.2f}       " + "  ".join(f"{v:.3f}" for v in vals))
        allv = [pt.layer.frac(ge, le) for ge in effs_g for le in effs_l]
        print(f"    range {min(allv):.3f}-{max(allv):.3f}")

    lat = []
    for pt in points(args.tokens):
        for total in REAL_MSG_LATENCY_RANGE_NS:
            k = derive_knobs(pt, pt.layer.frac(), total)
            lat.append(dict(point=pt.label, real_latency_ns=total,
                            LX=round(total / x_real, 2),
                            link_latency_ns=round(k["link_latency_ns"]),
                            msg_overhead_ns=round(k["msg_overhead_ns"])))
    write_csv(os.path.join(args.outdir, "sensitivity_latency.csv"), lat)
    print("\n3b. Knobs over the real per-message latency range (wire fixed at "
          f"{REAL_WIRE_NS:.0f} ns real)")
    print(table(lat, ["point", "real_latency_ns", "LX", "link_latency_ns",
                      "msg_overhead_ns"]))

    # 3c. Tile-size mapping (Step 3).
    a1, a2 = fitted_alphas()
    print(f"\n3c. Tile-size mapping at the f~0.5 point (link knobs fixed). "
          f"Matrix-core curve: alpha fitted {a1:.2f} and {a2:.2f} from HipKittens "
          f"MI355X data, used {CURVE_ALPHAS['matrix-core']}")
    trows = tile_rows(args.tokens, points(args.tokens)[1])
    write_csv(os.path.join(args.outdir, "tile_mapping.csv"), trows)
    print(table(trows, ["sim_tile_size", "real_tile", "curve", "valu_scale",
                        "sim_gemm_rel", "real_waves", "sim_waves", "real_LX",
                        "sim_LX", "real_f", "sim_f_n1"]))

    # 4. MoE.
    moe = moe_rows()
    write_csv(os.path.join(args.outdir, "moe.csv"), moe)
    print("\n4. MoE all-to-all, Mixtral 8x7B, EP 8 (analysis only; busiest GPU)")
    print(table(moe, ["fabric", "tokens_per_gpu", "skew", "a2a_latency_us", "a2a_us",
                      "gate_up_us", "down_us", "weight_bound", "f_dispatch",
                      "f_combine"]))
    print(f"\nwrote CSVs to {args.outdir}")


if __name__ == "__main__":
    main()
