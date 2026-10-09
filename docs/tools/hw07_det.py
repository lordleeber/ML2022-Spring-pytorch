# Run a script with PyTorch's deterministic algorithms turned on, without editing it.
# BERT training on CUDA is not reproducible by default (atomic adds in the embedding backward):
# two runs of the same train.py differ after 100 steps. This wrapper makes runs bit-identical,
# so the split train.py can be compared with the notebook reference.
# usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw07_det.py <script.py> [args...]
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
