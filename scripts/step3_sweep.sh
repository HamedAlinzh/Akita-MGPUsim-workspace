#!/usr/bin/env bash
# Step 3: placements, tile size x N with two compute curves, 4-GPU
# baselines. 2 GPUs, 384x256, K 512, 4 CUs, -lds-occupancy, f~0.5 link
# knobs unless noted. Resumable: runs whose stdout has "Passed!" are
# skipped. Usage: step3_sweep.sh [main|variants]
set -u
cd "$(dirname "$0")/.."
out=results/step3_tiles/runs
bin=results/step3_tiles/allreducegemm
mkdir -p "$out"
link="-link-bw=0.6325 -link-latency=544 -msg-overhead=2618"
base="-timing -verify -m=384 -n=256 -cus-per-gpu=4 -lds-occupancy $link"
two="$base -gpus=1,2 -k=512"
four="$base -gpus=1,2,3,4 -k=1024"
MC=0.647 FLAT=0.451 STEEP=0.96

queue=()
add() { queue+=("$*"); }

run() {  # name, flags...
  local name=$1; shift
  local dir="$out/$name"
  if grep -q "Passed!" "$dir/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$dir"
  (cd "$dir" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" "$@" > stdout.txt 2>&1)
}

if [[ "${1:-main}" == main ]]; then
  add nm_t8_tile_n1 $two -tile-size=8 -engine-mode=tile -placement=near-memory
  add nm_t8_bulk_n1 $two -tile-size=8 -engine-mode=bulk -placement=near-memory
  add nm_t8_rs_tile_n1 $two -tile-size=8 -engine-mode=tile -placement=near-memory -reduce-scatter
  for c in native:1 mc:$MC; do
    name=${c%%:*} s=${c##*:}
    t4="$two -tile-size=4 -valu-cycles-scale=$s"
    for n in 1 2 4; do add own_t4_${name}_tile_n$n $t4 -engine-mode=tile -tiles-per-msg=$n; done
    add own_t4_${name}_bulk_n1 $t4 -engine-mode=bulk -tiles-per-msg=1
    add own_t4_${name}_bulk_n4 $t4 -engine-mode=bulk -tiles-per-msg=4
    add mccl_t4_${name} $t4 -collective-threads=16384
    add nm_t4_${name}_tile_n1 $t4 -engine-mode=tile -placement=near-memory
  done
  add mccl_t8_4gpu $four -tile-size=8 -collective-threads=16384
  add own_t8_bulk_n1_4gpu $four -tile-size=8 -engine-mode=bulk
else
  # tileSize 8 owner tile runs again, now logging first_ready (Step 2's
  # binary did not).
  for n in 1 2 4; do add own_t8_tile_n$n $two -tile-size=8 -engine-mode=tile -tiles-per-msg=$n; done
  n=${BEST_N:?set BEST_N}
  for c in flat:$FLAT steep:$STEEP; do
    name=${c%%:*} s=${c##*:}
    t4="$two -tile-size=4 -valu-cycles-scale=$s"
    add own_t4_${name}_tile_n$n $t4 -engine-mode=tile -tiles-per-msg=$n
    add own_t4_${name}_bulk_n$n $t4 -engine-mode=bulk -tiles-per-msg=$n
  done
fi

running=0
for job in "${queue[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo "done ${1:-main}"
