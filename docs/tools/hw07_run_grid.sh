#!/bin/bash
# Run HW07 experiments one after another (deterministic mode). Each line of the spec file:
#   <tag> <hw07_exp.py options...>
# usage: bash docs/tools/hw07_run_grid.sh <spec file> <scratch dir>
# Results append to docs/tools/hw07_runs.jsonl, logs to docs/tools/hw07_logs/<tag>.log,
# checkpoints to <scratch dir>/ckpt/<tag>. A tag already in the jsonl is skipped.
R=$(cd "$(dirname "$0")/../.." && pwd)
SPEC=$(realpath "$1"); W=$2
[ -f "$SPEC" ] || { echo "RUN_FAILED no spec $1"; exit 1; }
mkdir -p "$W/ckpt" "$R/docs/tools/hw07_logs"
cd "$W" && ln -sf "$R"/HW07/hw7_*.json .
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$R/HW07"
while read -r tag opts; do
  [ -z "$tag" ] || [[ $tag == \#* ]] && continue
  if grep -q "\"tag\": \"$tag\"" "$R/docs/tools/hw07_runs.jsonl" 2>/dev/null; then echo "skip $tag"; continue; fi
  echo "$(date +%H:%M:%S) start $tag $opts"
  "$R/.venv/bin/python" "$R/docs/tools/hw07_det.py" "$R/docs/tools/hw07_exp.py" --tag "$tag" $opts \
    --save "$W/ckpt/$tag" --jsonl "$R/docs/tools/hw07_runs.jsonl" > "$R/docs/tools/hw07_logs/$tag.log" 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) end $tag rc=$rc $(grep -o '"dev_by_epoch": \[[^]]*\]' "$R/docs/tools/hw07_logs/$tag.log")"
  [ $rc -ne 0 ] && { echo "RUN_FAILED $tag"; exit 1; }
done < "$SPEC"
echo GRID_DONE
