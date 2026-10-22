#!/usr/bin/env bash
# Step 5b B: is the slow first GEMM round a simulator artifact? 1 GPU, GEMM
# alone (engine tile mode), capped baseline shape, each run launches the GEMM
# twice (-warmup: the first untimed) and records work-group times and a
# memory timeline. Resumable: a run with a "Passed!" stdout is skipped.
set -u
cd "$(dirname "$0")/.."
out=results/step5b_bounds/runs
bin=results/step5b_bounds/allreducegemm
mkdir -p "$out"
base="-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -cp-dispatchers=16 -ordered-dispatch -wg-trace=wgs.csv -mem-timeline=mem.csv"
one="$base -gpus=1 -k=256 -link-bw=0.7807 -link-latency=441 -msg-overhead=2121 -engine-mode=tile -warmup"

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

add warm_plain $one
add warm_s2 $one -splitk-ways=2 -splitk-rows=64
# C: the per-row plan the queue model picked (rows 0-3 split 4 ways, 4-5
# plain, 6-7 split 2 ways; 21 launches per GPU), 2 GPUs.
two="-timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -cp-dispatchers=32 -ordered-dispatch -wg-trace=wgs.csv -gpus=1,2 -k=512 -engine-mode=tile"
plan="-splitk-plan=4,4,4,4,1,1,2,2"
add c_f05_plan $two -link-bw=0.7807 -link-latency=441 -msg-overhead=2121 $plan
add c_f08_plan $two -link-bw=0.223 -link-latency=1542 -msg-overhead=7424 $plan
# Step 5's gemm_s2 again with this binary (expect gemm_wait 334.490 us).
add repro_gemm_s2 -timing -verify -m=256 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -cp-dispatchers=16 -ordered-dispatch -wg-trace=wgs.csv -gpus=1 -k=256 -link-bw=0.7807 -link-latency=441 -msg-overhead=2121 -engine-mode=tile -splitk-ways=2 -splitk-rows=64
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
