#!/usr/bin/env bash
# Step 2 early timing: mccl baselines with the representative points' link
# knobs (the engine versions at f~0.5, N=1 are the correctness runs
# ar_tile_n1 and ar_bulk_n1). Resumable.
set -u
cd "$(dirname "$0")/.."
out=results/step2_engine/timing
bin=results/step2_engine/allreducegemm
mkdir -p "$out"
common="-timing -verify -gpus=1,2 -k=512 -tile-size=8 -cus-per-gpu=4 -lds-occupancy \
 -collective-threads=16384"

run() {  # name, extra flags
  local dir="$out/$1"; shift
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" $common "$@" > stdout.txt 2>&1)
}

run mccl_ring_ar_f05 -m=384 -n=256 -link-bw=0.6325 -link-latency=544 &
run mccl_ring_rs_f025 -m=512 -n=320 -link-bw=1.1504 -link-latency=299 -reduce-scatter &
wait
echo done
