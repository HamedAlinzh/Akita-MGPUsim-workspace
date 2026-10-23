#!/usr/bin/env bash
# Step 8 batch B (items 3-4): shapes, K per GPU, 9 tile rows and
# reduce-scatter on 2 GPUs, each s = 1 and the model's best plan, knobs and
# plans from results/step8_edges/batchb.txt (scripts/step8_batchb.py). Capped
# GEMM, 4 CUs, tileSize 8, -lds-occupancy, ordered dispatch and launch.
# Resumable: a run with a "Passed!" stdout is skipped. ONLY="a b" picks runs.
set -u
cd "$(dirname "$0")/.."
out=results/step8_edges/runs
bin=results/step8_edges/allreducegemm
mkdir -p "$out"
base="-timing -verify -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv -engine-mode=tile"

jobs=()
add() { jobs+=("$*"); }
run() {
  local name=$1; shift
  if [[ -n ${ONLY:-} && " $ONLY " != *" $name "* ]]; then return; fi
  local dir="$out/$name"
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" "$@" > stdout.txt 2>&1)
}

add w15_s1 $base -gpus=1,2 -m=192 -n=256 -k=512 -link-bw=0.7349 -link-latency=468 -msg-overhead=2253
add w15_plan $base -gpus=1,2 -m=192 -n=256 -k=512 -link-bw=0.7349 -link-latency=468 -msg-overhead=2253 -splitk-plan=4,4,1,1,2,1
add w337_s1 $base -gpus=1,2 -m=384 -n=288 -k=512 -link-bw=0.8446 -link-latency=407 -msg-overhead=1961
add w337_plan $base -gpus=1,2 -m=384 -n=288 -k=512 -link-bw=0.8446 -link-latency=407 -msg-overhead=1961 -splitk-plan=4,4,2,1,1,1,1,1,1,2,2,2
add r9_s1 $base -gpus=1,2 -m=288 -n=256 -k=512 -link-bw=0.7970 -link-latency=432 -msg-overhead=2078
add r9_plan $base -gpus=1,2 -m=288 -n=256 -k=512 -link-bw=0.7970 -link-latency=432 -msg-overhead=2078 -splitk-plan=4,4,4,2,2,1,1,2,1
add k128_s1 $base -gpus=1,2 -m=256 -n=256 -k=256 -link-bw=1.3959 -link-latency=246 -msg-overhead=1186
add k128_plan $base -gpus=1,2 -m=256 -n=256 -k=256 -link-bw=1.3959 -link-latency=246 -msg-overhead=1186 -splitk-plan=4,4,4,1,1,2,1,1
add k512_s1 $base -gpus=1,2 -m=256 -n=256 -k=1024 -link-bw=0.4148 -link-latency=829 -msg-overhead=3992
add k512_plan $base -gpus=1,2 -m=256 -n=256 -k=1024 -link-bw=0.4148 -link-latency=829 -msg-overhead=3992 -splitk-plan=4,4,4,2,2,1,1,1
add rs05_s1 $base -gpus=1,2 -m=256 -n=256 -k=512 -link-bw=0.3903 -link-latency=881 -msg-overhead=4243 -reduce-scatter
add rs05_plan $base -gpus=1,2 -m=256 -n=256 -k=512 -link-bw=0.3903 -link-latency=881 -msg-overhead=4243 -reduce-scatter -splitk-plan=4,4,4,2,2,1,1,1
add rs08_s1 $base -gpus=1,2 -m=256 -n=256 -k=512 -link-bw=0.1112 -link-latency=3093 -msg-overhead=14890 -reduce-scatter
add rs08_plan $base -gpus=1,2 -m=256 -n=256 -k=512 -link-bw=0.1112 -link-latency=3093 -msg-overhead=14890 -reduce-scatter -splitk-plan=4,1,1,1,1,1,1,1
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384

running=0
for job in "${jobs[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
