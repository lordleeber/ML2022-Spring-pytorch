#!/bin/bash
# Run grid specs one after another (each waits for the previous to finish).
# usage: docs/tools/hw12_queue.sh <spec name> [<spec name> ...]   (spec name = docs/tools/hw12_specs/<name>.txt)
cd "$(dirname "$0")/../.."
for name in "$@"; do
  rm -f docs/tools/hw12_logs/$name.log
  docs/tools/hw12_grid.sh docs/tools/hw12_specs/$name.txt $name
done
