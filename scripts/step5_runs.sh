#!/usr/bin/env bash
# Step 5: split-K for the first and last bands (s = 2 at 64 rows, s = 4 at
# 32 rows) on the capped baseline (2 GPUs, 4 CUs, 256x256, K/GPU 256,
# tileSize 8, -gemm-wg-cap 8, -lds-occupancy), knobs from points_capped.csv.
# 16 CP dispatchers so s = 4's nine kernels per GPU all start at launch, and
# ordered dispatch so the first band's work-groups go before the middle's
# and the middle's before the last band's.
# Resumable: a run with a "Passed!" stdout is skipped.
set -u
cd "$(dirname "$0")/.."
out=results/step5_splitk/runs
bin=results/step5_splitk/allreducegemm
mkdir -p "$out"
base="-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -cp-dispatchers=16 -ordered-dispatch -wg-trace=wgs.csv"
two="$base -gpus=1,2 -k=512"
f05="-link-bw=0.7807 -link-latency=441 -msg-overhead=2121"
f08="-link-bw=0.223 -link-latency=1542 -msg-overhead=7424"
s2="-splitk-ways=2 -splitk-rows=64"
s4="-splitk-ways=4 -splitk-rows=32"

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

add gemm_s2 $base -gpus=1 -k=256 $f05 -engine-mode=tile $s2
add gemm_s4 $base -gpus=1 -k=256 $f05 -engine-mode=tile $s4
add f05_tile_s2 $two $f05 -engine-mode=tile $s2
add f05_tile_s4 $two $f05 -engine-mode=tile $s4
add f05_tile_s2_nopresum $two $f05 -engine-mode=tile $s2 -splitk-presum=off
add f08_tile_s2 $two $f08 -engine-mode=tile $s2
add f08_tile_s4 $two $f08 -engine-mode=tile $s4
add f05_bulk_s2 $two $f05 -engine-mode=bulk $s2
# Added after the first results: s = 1 with the same flags and a trace, to
# check the flags leave the baseline unchanged and to time plain work-groups.
add f05_tile_s1 $two $f05 -engine-mode=tile
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384

only=${ONLY:-}
running=0
for job in "${queue[@]}"; do
  name=${job%% *}
  if [[ -n $only && " $only " != *" $name "* ]]; then continue; fi
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
