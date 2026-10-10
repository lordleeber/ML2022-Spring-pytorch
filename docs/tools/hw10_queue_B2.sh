#!/bin/bash
# The 16-model logits-sum ensembles saturated (zero input gradients, ch05); rerun them with other aggregations
# after grid U, keeping a single attack runner at a time.
R=/home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch
cd $R
while ! grep -q ALLDONE docs/tools/hw10_logs/hw10_grid_U.progress 2>/dev/null; do sleep 20; done
docs/tools/hw10_run_grid.sh docs/tools/hw10_grid_B2.txt docs/tools/hw10_runs.jsonl /tmp/claude-1000/-home-valtec-poyi-GitHubLL-ML2022-Spring-pytorch/9cfb0b8b-5321-4271-908f-9fbd7802d111/scratchpad/runs
