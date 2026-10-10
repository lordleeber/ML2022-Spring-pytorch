#!/bin/bash
cd /home/valtec/poyi/GitHubLL/ML2022-Spring-pytorch/HW10
../.venv/bin/python ../docs/tools/hw10_zoo.py ../docs/tools/hw10_zoo.jsonl > ../docs/tools/hw10_zoo.log 2>&1
echo DONE >> ../docs/tools/hw10_zoo.log
