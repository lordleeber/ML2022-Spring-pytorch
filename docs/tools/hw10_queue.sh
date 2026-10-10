#!/bin/bash
# Wait for the previous queue to finish (its .progress ends with ALLDONE), then run the given specs one after another.
# usage: setsid nohup docs/tools/hw10_queue.sh <wait_for_progress_file> <spec>... &
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
cd $R
W=$1; shift
while [ -n "$W" ] && ! grep -q ALLDONE "$W" 2>/dev/null; do sleep 20; done
for spec in "$@"; do
  docs/tools/hw10_run_grid.sh $spec docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
done
