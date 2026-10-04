#!/bin/bash
# Run a list of hw02_exp.py configurations, a few at a time, and collect the JSON results.
#
# usage: docs/tools/hw02_run_grid.sh <runs.txt> <out.jsonl> [parallel=2]
#   runs.txt: one run per line, "<name> <hw02_exp.py args>"; blank lines and # comments are skipped.
#   Each run's per-epoch lines go to <out.jsonl dir>/<name>.log.
#   Every concat-11 run holds about 11 GB of RAM (utils.py preallocates 3,000,000 rows per split),
#   so keep parallel low.
#
# Example (reproduces every run behind FACTS.md「ch07 實測」):
#   docs/tools/hw02_run_grid.sh docs/tools/hw02_ch07_runs.txt /tmp/hw02_ch07/runs.jsonl 2
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=$(realpath "$1"); OUT=$(realpath -m "$2"); P=${3:-2}
DIR=$(dirname "$OUT"); mkdir -p "$DIR"
cd "$ROOT/HW02"
touch "$OUT"
grep -v -e '^\s*$' -e '^\s*#' "$RUNS" | while read -r name args; do
  echo "PYTHONPATH=. ../.venv/bin/python $ROOT/docs/tools/hw02_exp.py --name $name $args 2>/dev/null | tee $DIR/$name.log | grep '^{' >> $OUT"
done | xargs -P "$P" -I{} bash -c "{}"
echo "$(wc -l < "$OUT") runs -> $OUT"
