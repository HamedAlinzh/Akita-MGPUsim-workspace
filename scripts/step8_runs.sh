#!/usr/bin/env bash
# Step 8 runs (batch 2): 4-GPU engine runs with per-peer queues, the f sweep,
# and (batch B) shapes and reduce-scatter. Capped GEMM, 4 CUs, tileSize 8,
# -lds-occupancy, ordered dispatch and launch. Resumable: a run with a
# "Passed!" stdout is skipped. ONLY="a b" picks runs.
set -u
cd "$(dirname "$0")/.."
out=results/step8_edges/runs
bin=results/step8_edges/allreducegemm
mkdir -p "$out"
base="-timing -verify -cus-per-gpu=4 -lds-occupancy -tile-size=8 -gemm-wg-cap=8 -ordered-dispatch -cp-dispatchers=40 -wg-trace=wgs.csv -engine-mode=tile"
sq="-m=256 -n=256"
x2="$base $sq -gpus=1,2 -k=512"
x4="$base $sq -gpus=1,2,3,4 -k=1024 -peer-mesh -engine-peer-queues -link-bw=0.3903 -link-latency=881 -msg-overhead=4243"

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

# Batch A.
add x4_s1_pq $x4
add x4_plan_pq $x4 -splitk-plan=4,4,4,4,1,1,2,2
add f035_s1 $x2 -link-bw=1.8077 -link-latency=190 -msg-overhead=916
add f035_plan $x2 -link-bw=1.8077 -link-latency=190 -msg-overhead=916 -splitk-plan=2,2,2,2,2,2,4,4
add f065_s1 $x2 -link-bw=0.5241 -link-latency=656 -msg-overhead=3160
add f065_plan $x2 -link-bw=0.5241 -link-latency=656 -msg-overhead=3160 -splitk-plan=4,1,1,2,1,1,2,4
add f090_s1 $x2 -link-bw=0.1082 -link-latency=3181 -msg-overhead=15312
add f090_plan $x2 -link-bw=0.1082 -link-latency=3181 -msg-overhead=15312 -splitk-plan=4,1,1,1,1,2,2,4
# Batch B (items 3 and 4): scripts/step8_runs_b.sh.

running=0
for job in "${jobs[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
