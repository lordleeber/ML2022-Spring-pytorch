#!/bin/bash
# HW12 grid runner: reads one argument line per run from a spec file, runs 20 at a time on the CPU,
# appends one JSON line per run to docs/tools/hw12_runs.jsonl.
# usage: docs/tools/hw12_grid.sh <spec file> <log name>
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
P=.venv/bin/python
LOG=docs/tools/hw12_logs/$2.log
grep -v '^#' "$1" | grep -v '^$' | xargs -P ${HW12_JOBS:-20} -L 1 sh -c "$P docs/tools/hw12_exp.py --out docs/tools/hw12_runs.jsonl \$0 \"\$@\" >> $LOG 2>&1 || echo FAILED: \$0 \$@ >> $LOG"
echo DONE >> $LOG
