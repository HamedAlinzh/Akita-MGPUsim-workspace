#!/usr/bin/env bash
# Step 7 (batch 1): software-overlap baseline, direct software collective on
# 4 and 8 GPUs, 8-GPU engine runs, and the model rank test. f~0.5 shape,
# capped GEMM, 4 CUs, tileSize 8, -lds-occupancy. 8-GPU runs go one at a
# time (about 2 GB each) beside two others. Resumable: a run with a
# "Passed!" stdout is skipped. ONLY="a b" picks runs.
set -u
cd "$(dirname "$0")/.."
out=results/step7_baselines/runs
bin=results/step7_baselines/allreducegemm
mkdir -p "$out"
base="-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40"
x2="$base -gpus=1,2 -k=512 -link-bw=0.7807 -link-latency=441 -msg-overhead=2121"
x4="$base -gpus=1,2,3,4 -k=1024 -peer-mesh -link-bw=0.3903 -link-latency=881 -msg-overhead=4243"
x8="$base -gpus=1,2,3,4,5,6,7,8 -k=2048 -peer-mesh -link-bw=0.1952 -link-latency=1763 -msg-overhead=8485"
sw="-collective-threads=16384"
plan="-splitk-plan=4,4,4,4,1,1,2,2"
trace="-wg-trace=wgs.csv"

big=(); small=()
addbig() { big+=("$*"); }
add() { small+=("$*"); }
run() {
  local name=$1; shift
  if [[ -n ${ONLY:-} && " $ONLY " != *" $name "* ]]; then return; fi
  local dir="$out/$name"
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" "$@" > stdout.txt 2>&1)
}

addbig x8_tile_s1 $x8 -engine-mode=tile -engine-peer-queues $trace
addbig x8_tile_plan $x8 -engine-mode=tile -engine-peer-queues $plan $trace
addbig x8_direct $x8 -algorithm=direct $sw
add sw_s1 $x2 -sw-overlap=128,128 $sw $trace
add sw_plan $x2 -sw-overlap=128,64,64 $plan $sw $trace
add x4_bulk $x4 -engine-mode=bulk
add x4_direct $x4 -algorithm=direct $sw
add rank_2nd $x2 -engine-mode=tile -splitk-plan=4,4,4,4,4,4,4,1 $trace
add rank_10th $x2 -engine-mode=tile -splitk-plan=1,2,4,1,1,1,1,1 $trace
add rank_median $x2 -engine-mode=tile -splitk-plan=4,1,4,4,2,1,2,4 $trace
add rank_worst $x2 -engine-mode=tile -splitk-plan=1,1,1,1,4,4,4,4 $trace
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384

( for job in "${big[@]}"; do run $job; done ) &
running=0
for job in "${small[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 2 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
