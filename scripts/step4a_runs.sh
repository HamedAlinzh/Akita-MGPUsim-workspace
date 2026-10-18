#!/usr/bin/env bash
# Dispatch-shaping spike, 2- and 4-GPU runs (tile mode N=1 unless noted).
# 4 CUs, 384x256, K/GPU 256, tileSize 8, -lds-occupancy. Resumable.
set -u
cd "$(dirname "$0")/.."
out=results/step4a_shaping/runs
bin=results/step4a_shaping/allreducegemm
mkdir -p "$out"
base="-timing -verify -m=384 -n=256 -cus-per-gpu=4 -lds-occupancy -tile-size=8"
f05="-link-bw=0.6325 -link-latency=544 -msg-overhead=2618"
f08="-link-bw=0.1807 -link-latency=1904 -msg-overhead=9163"
two="$base -gpus=1,2 -k=512"
four="$base -gpus=1,2,3,4 -k=1024 -peer-mesh $f05"

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

add f08_tile_none $two $f08 -engine-mode=tile
for p in f05 f08; do
  knobs=$f05; [[ $p == f08 ]] && knobs=$f08
  add ${p}_tile_ramp1 $two $knobs -engine-mode=tile -wg-ramp=1
  add ${p}_tile_cap4 $two $knobs -engine-mode=tile -wg-cap=4
done
add f08_tile_t4_mc $base -gpus=1,2 -k=512 $f08 -tile-size=4 -valu-cycles-scale=0.647 -engine-mode=tile
add mesh4_mccl $four -collective-threads=16384
add mesh4_bulk $four -engine-mode=bulk
add mesh4_tile $four -engine-mode=tile
add defaults_433 -timing --report-all -verify -gpus=1,2 -m=256 -k=1024 -n=256 -algorithm=ring -band-rows=0 -overlap=false -allreduce-chunk=0 -collective-threads=16384

running=0
for job in "${queue[@]}"; do
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
