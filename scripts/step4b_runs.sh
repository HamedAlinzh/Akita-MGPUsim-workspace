#!/usr/bin/env bash
# Capped baseline (-gemm-wg-cap 8) at f~0.5 and f~0.8, ramp on top of it,
# and small-tiles-first. 2 GPUs, 4 CUs, 256x256, K/GPU 256, tileSize 8,
# -lds-occupancy; knobs from ratio_model.py (points_capped.csv). Resumable.
set -u
cd "$(dirname "$0")/.."
out=results/step4b_capped/runs
bin=results/step4b_capped/allreducegemm
mkdir -p "$out"
base="-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8"
two="$base -gpus=1,2 -k=512"
f05="-link-bw=0.7807 -link-latency=441 -msg-overhead=2121"
f08="-link-bw=0.223 -link-latency=1542 -msg-overhead=7424"
small="-small-valu-scale=0.738 -small-tiles-per-msg=2"

queue=()
add() { queue+=("$*"); }
run() {
  local name=$1; shift
  local dir="$out/$name"
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" "$@" > stdout.txt 2>&1)
}

for p in f05 f08; do
  knobs=$f05; [[ $p == f08 ]] && knobs=$f08
  add ${p}_mccl $two $knobs -collective-threads=16384
  add ${p}_bulk $two $knobs -engine-mode=bulk
  add ${p}_tile $two $knobs -engine-mode=tile
  add ${p}_tile_ramp1 $two $knobs -engine-mode=tile -wg-ramp=1
  for r in 32 64 128; do
    add ${p}_small_r$r $two $knobs -engine-mode=tile -small-rows=$r $small
  done
done
for r in 32 64 128; do
  add gemm_small_r$r $base -gpus=1 -k=256 $f05 -engine-mode=tile -small-rows=$r $small
done
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384

running=0
for job in "${queue[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
