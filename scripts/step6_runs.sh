#!/usr/bin/env bash
# Step 6: warm vs cold at f~0.5, the f~0.25 point (reduce-scatter, 512x320),
# and 4 GPUs on the per-pair mesh. Capped GEMM (-gemm-wg-cap 8), 4 CUs,
# tileSize 8, K/GPU 256, -lds-occupancy, ordered dispatch and launch.
# Knobs: results/step0_ratio/points_capped.csv and points_step6.csv.
# Resumable: a run with a "Passed!" stdout is skipped. ONLY="a b" picks runs;
# F025_PLAN sets the f~0.25 plan.
set -u
cd "$(dirname "$0")/.."
out=results/step6_paper/runs
bin=results/step6_paper/allreducegemm
mkdir -p "$out"
base="-timing -verify -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -cp-dispatchers=32 -ordered-dispatch -wg-trace=wgs.csv"
f05="-m=256 -n=256 -link-bw=0.7807 -link-latency=441 -msg-overhead=2121"
f025="-m=512 -n=320 -reduce-scatter -link-bw=1.632 -link-latency=211 -msg-overhead=1015"
f05x4="-m=256 -n=256 -gpus=1,2,3,4 -k=1024 -peer-mesh -link-bw=0.3903 -link-latency=881 -msg-overhead=4243"
plan="-splitk-plan=4,4,4,4,1,1,2,2"
f025plan="-splitk-plan=${F025_PLAN:-}"

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

add warm_f05_s1 $base $f05 -gpus=1,2 -k=512 -engine-mode=tile -warmup
add warm_f05_plan $base $f05 -gpus=1,2 -k=512 -engine-mode=tile -warmup $plan
add f025_tile_s1 $base $f025 -gpus=1,2 -k=512 -engine-mode=tile
add f025_bulk_s1 $base $f025 -gpus=1,2 -k=512 -engine-mode=bulk
add f05_4gpu_tile_s1 $base $f05x4 -engine-mode=tile
add f05_4gpu_tile_plan $base $f05x4 -engine-mode=tile $plan
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384
if [[ -n ${F025_PLAN:-} ]]; then
  add f025_tile_plan $base $f025 -gpus=1,2 -k=512 -engine-mode=tile $f025plan
fi

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
