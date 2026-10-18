#!/usr/bin/env bash
# Step 2 correctness: the communication engine on 2 GPUs (4 CUs, 384x256,
# K per GPU 256, tileSize 8) at N = 1, 2, 4 in tile and bulk mode, plus
# reduce-scatter at N = 1 and one 4-GPU all-reduce at N = 1. Link knobs are
# the f~0.5 point's. Resumable: a run with a "Passed!" stdout is skipped.
set -u
cd "$(dirname "$0")/.."
out=results/step2_engine/correctness
bin=results/step2_engine/allreducegemm
mkdir -p "$out"
common="-timing -verify -m=384 -n=256 -tile-size=8 -cus-per-gpu=4 -lds-occupancy \
 -link-bw=0.6325 -link-latency=544 -msg-overhead=2618"

run() {  # name, extra flags
  local dir="$out/$1"; shift
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" $common "$@" > stdout.txt 2>&1)
}

jobs=()
for mode in tile bulk; do
  for n in 1 2 4; do
    run "ar_${mode}_n${n}" -gpus=1,2 -k=512 -engine-mode=$mode -tiles-per-msg=$n &
    jobs+=($!)
    if (( ${#jobs[@]} >= 3 )); then wait "${jobs[0]}"; jobs=("${jobs[@]:1}"); fi
  done
  run "rs_${mode}_n1" -gpus=1,2 -k=512 -engine-mode=$mode -tiles-per-msg=1 -reduce-scatter &
  jobs+=($!)
  if (( ${#jobs[@]} >= 3 )); then wait "${jobs[0]}"; jobs=("${jobs[@]:1}"); fi
done
run "ar_tile_n1_4gpu" -gpus=1,2,3,4 -k=1024 -engine-mode=tile -tiles-per-msg=1 &
wait
echo done
