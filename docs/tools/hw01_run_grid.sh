#!/bin/bash
# Run a list of hw01_exp.py configurations in parallel and collect the JSON results.
#
# usage: docs/tools/hw01_run_grid.sh <runs.txt> <out.jsonl> [parallel=6] [extra args...]
#   runs.txt: one run per line, "<name> <hw01_exp.py args>"; blank lines are skipped.
#   Every run gets --fix 1 (honest validation) unless its own args override it.
#
# Example (reproduces every run behind FACTS.md「ch07 實測」, about 10 minutes on one GPU):
#   docs/tools/hw01_run_grid.sh docs/tools/hw01_ch07_runs.txt /tmp/ch07.jsonl 6
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(realpath "$1"); OUT=$(realpath -m "$2"); P=${3:-6}; shift 3 || shift $#
EXTRA="$*"
cd "$ROOT/HW01"
: > "$OUT"
grep -v '^\s*$' "$RUNS" | while read -r name args; do
  echo "PYTHONPATH=. ../.venv/bin/python $ROOT/docs/tools/hw01_exp.py --name $name --fix 1 $args $EXTRA 2>/dev/null | grep '^{' >> $OUT"
done | xargs -P "$P" -I{} bash -c "{}"
echo "$(wc -l < "$OUT") runs -> $OUT"
