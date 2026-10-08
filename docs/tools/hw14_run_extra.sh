#!/bin/bash
# HW14 experiment E (user decision 2026-10-08), run after hw14_run_grid.sh, one at a time.
# Run from HW14/:  bash ../docs/tools/hw14_run_extra.sh <logdir>
set -u
LOG=${1:-logs}
mkdir -p "$LOG"
PY=../.venv/bin/python
EXTRA=../docs/tools/hw14_extra.py
OUT=../docs/tools/hw14_runs.jsonl

run() {  # tag, args...
  local tag=$1; shift
  if grep -q "\"tag\": \"$tag\"" "$OUT" 2>/dev/null; then echo "skip $tag"; return; fi
  echo "$(date +%T) start $tag $*"
  $PY -u $EXTRA --tag "$tag" --out "$OUT" "$@" > "$LOG/$tag.out" 2> "$LOG/$tag.err" \
    || { echo "$(date +%T) FAILED $tag"; tail -5 "$LOG/$tag.err"; exit 1; }
  echo "$(date +%T) done $tag $(tail -1 "$LOG/$tag.out" | grep -o '[0-9.]*\]$')"
}

# E1: all five tasks mixed (upper bound)
run E1_joint --mode joint
# E2: replay with 50 / 200 / 1000 kept images per finished task
for m in 50 200 1000; do run "E2_replay_m$m" --mode replay --mem $m; done
echo "$(date +%T) EXTRA DONE"
