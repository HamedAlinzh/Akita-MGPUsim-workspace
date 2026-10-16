#!/usr/bin/env bash
# Runs every pending experiment for the GEMM/All-Reduce overlap study, in two
# lanes, and is safe to rerun: each step skips work that already finished.
#
#   lane 1: All-Reduce cost breakdown on the fixed mccl (2 GPUs, then 4 GPUs)
#           -> results/allreduce_breakdown/after/g{2,4}.csv
#   lane 2: tile x band x overlap sweeps, 256x256x4096 (4 and 2 GPUs), then
#           256x256x1024 (2 GPUs)
#           -> results/overlap_sweep_k4096/, results/overlap_sweep_k1024/
#
# Start it detached, so it survives Claude Code or the terminal closing:
#   setsid nohup scripts/run_pending_experiments.sh \
#       > results/pending_experiments.log 2>&1 < /dev/null &
#
# Check on it:
#   tail -f results/pending_experiments.log
#   ps -eo etime,args | grep -E 'allreducegemm_t|allreducebench' | grep -v grep
#
# Stop it (finished runs are kept; rerun the start command to resume):
#   pkill -f run_pending_experiments.sh; pkill -f sweep_overlap.py
#   pkill -f allreducegemm_t; pkill -f allreducebench
#
# JOBS (default 3) is the number of sweep simulations run at once in lane 2.

set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MGPU=$ROOT/simulator/mgpusim-3.0.3
JOBS=${JOBS:-3}

exec 9>"$ROOT/results/.pending_experiments.lock"
if ! flock -n 9; then
    echo "run_pending_experiments.sh is already running" >&2
    exit 1
fi

stamp() { echo "[$(date '+%F %T')] $*"; }

lane_breakdown() {
    local out=$ROOT/results/allreduce_breakdown/after
    mkdir -p "$out"
    stamp "lane 1: building allreducebench"
    (cd "$MGPU" && go build -o "$out/allreducebench" ./samples/allreducebench) || {
        stamp "lane 1: build failed"
        return 1
    }
    for g in 2 4; do
        stamp "lane 1: All-Reduce breakdown, $g GPUs"
        (cd "$out" && ./allreducebench -ring-gpus "$g" -out "g$g.csv" >> "g$g.log" 2>&1)
        stamp "lane 1: $g GPUs finished (exit $?)"
    done
}

lane_sweeps() {
    stamp "lane 2: 256x256x4096 sweep"
    python3 "$ROOT/scripts/sweep_overlap.py" \
        --m 256 --k 4096 --n 256 --gpus 4,2 --tiles 16,8,4 --bands 0,128,64 \
        --jobs "$JOBS" --outdir "$ROOT/results/overlap_sweep_k4096"
    stamp "lane 2: 256x256x4096 finished (exit $?)"

    stamp "lane 2: 256x256x1024 sweep"
    python3 "$ROOT/scripts/sweep_overlap.py" \
        --m 256 --k 1024 --n 256 --gpus 2 --tiles 16,8,4 --bands 0,128,64 \
        --jobs "$JOBS" --outdir "$ROOT/results/overlap_sweep_k1024"
    stamp "lane 2: 256x256x1024 finished (exit $?)"
}

stamp "starting (JOBS=$JOBS)"
lane_breakdown &
lane1=$!
lane_sweeps &
lane2=$!

trap 'kill $lane1 $lane2 2>/dev/null; wait' TERM HUP INT
wait $lane1 $lane2
stamp "all pending experiments finished"
