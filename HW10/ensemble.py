# Ensemble of several pytorchcv models as one proxy model (HW10.ipynb cell [24]; the TODO is filled in).
# Ensemble multiple models as your proxy model to increase the black-box transferability
# (paper: https://arxiv.org/abs/1611.02770)
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model

class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        for i, m in enumerate(self.models):
        # TODO: sum up logits from multiple models  
        # return ensemble_logits
            ensemble_logits = m(x) if i == 0 else ensemble_logits + m(x)
        return ensemble_logits
