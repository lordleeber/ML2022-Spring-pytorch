# Run a script with PyTorch's deterministic algorithms turned on, without editing it.
# Attacks need input gradients; cuDNN convolution backward is not deterministic by default,
# so the sign of near-zero gradients flips between runs. This wrapper makes runs bit-identical,
# so the split hw10.py can be compared with the notebook reference.
# usage: CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/tools/hw10_det.py <script.py> [args...]
import os
import runpy
import sys

import torch

torch.use_deterministic_algorithms(True)
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))  # as `python script.py` would
runpy.run_path(sys.argv[0], run_name='__main__')
