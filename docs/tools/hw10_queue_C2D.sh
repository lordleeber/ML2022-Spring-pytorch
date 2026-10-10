#!/bin/bash
# After the B2 reruns: combination grid C2 (ch06), then the JPEG-aware grid D (ch07), one runner at a time.
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
cd $R
while ! grep -q ALLDONE docs/tools/hw10_logs/hw10_grid_B2.progress 2>/dev/null; do sleep 20; done
docs/tools/hw10_run_grid.sh docs/tools/hw10_grid_C2.txt docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
docs/tools/hw10_run_grid_bpda.sh docs/tools/hw10_grid_D.txt docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
