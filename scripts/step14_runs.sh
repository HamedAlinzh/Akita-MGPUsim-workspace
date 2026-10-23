#!/usr/bin/env bash
# Step 14: runs listed in results/step14_allgather_sweep/runs.txt, one per
# line: "<name> <flags...>" (written by scripts/step14_allgather_sweep.py). At most 3 at
# a time. Resumable: a run with a "Passed!" stdout is skipped. ONLY="a b"
# picks runs. The binary is results/step14_allgather_sweep/allreducegemm.
set -u
cd "$(dirname "$0")/.."
dir=results/step14_allgather_sweep
out=$dir/runs
bin=$dir/allreducegemm
mkdir -p "$out"

run() {
  local name=$1; shift
  if [[ -n ${ONLY:-} && " $ONLY " != *" $name "* ]]; then return; fi
  local d="$out/$name"
  if grep -q "Passed!" "$d/stdout.txt" 2>/dev/null; then return; fi
  mkdir -p "$d"
  echo "$(date +%T) start $name"
  (cd "$d" && /usr/bin/time -f "wall_s=%e maxrss_kb=%M" \
    "../../../../$bin" "$@" > stdout.txt 2>&1)
  echo "$(date +%T) end $name $(grep -c 'Passed!' "$d/stdout.txt")"
}

mapfile -t jobs < <(grep -v '^\s*#' "$dir/runs.txt" | grep -v '^\s*$')
running=0
for job in "${jobs[@]}"; do
  # shellcheck disable=SC2086
  run $job &
  running=$((running + 1))
  if (( running >= 3 )); then wait -n; running=$((running - 1)); fi
done
wait
echo done
