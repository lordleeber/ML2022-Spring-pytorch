import torch.nn as nn


# Baseline
class baseline(object):
  """
  baseline technique: do nothing in regularization term [initialize and all weight is zero]
  """
  def __init__(self, model, dataloader, device):
    self.model = model
    self.dataloader = dataloader
    self.device = device
    # extract all parameters in models
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
    
    # store current parameters
    self.p_old = {} 

    # generate weight matrix
    self._precision_matrices = self._calculate_importance()  

    for n, p in self.params.items():
      # keep the old parameter in self.p_old
      self.p_old[n] = p.clone().detach() 

  def _calculate_importance(self):
    precision_matrices = {} 
    # initialize weight matrix（fill zero）
    for n, p in self.params.items(): 
      precision_matrices[n] = p.clone().detach().fill_(0)

    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
      loss += _loss.sum()
    return loss
  
  def update(self, model):
    # do nothing
    return
