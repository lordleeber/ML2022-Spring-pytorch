import numpy as np
import torch
import torch.nn as nn


def sample_spherical(npoints, ndim=3):
  vec = np.random.randn(ndim, npoints)
  vec /= np.linalg.norm(vec, axis=0)
  return torch.from_numpy(vec)

class scp(object):
  """
  OPEN REVIEW VERSION:
  https://openreview.net/forum?id=BJge3TNKwH
  """
  def __init__(self, model: nn.Module, dataloader, L: int, device, prev_guards=[None]):
    self.model = model 
    self.dataloader = dataloader
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
    self._state_parameters = {}
    self.L= L
    self.device = device
    self.previous_guards_list = prev_guards
    self._precision_matrices = self.calculate_importance()
    for n, p in self.params.items():
      self._state_parameters[n] = p.clone().detach()
  
  def calculate_importance(self):
    precision_matrices = {}
    for n, p in self.params.items():
      precision_matrices[n] = p.clone().detach().fill_(0)
      for i in range(len(self.previous_guards_list)):
        if self.previous_guards_list[i]:
          precision_matrices[n] += self.previous_guards_list[i][n]

    self.model.eval()
    if self.dataloader is not None:
      num_data = len(self.dataloader)
      for data in self.dataloader:
        self.model.zero_grad()
        output = self.model(data[0].to(self.device))
          
        mean_vec = output.mean(dim=0)

        L_vectors = sample_spherical(self.L, output.shape[-1])
        L_vectors = L_vectors.transpose(1,0).to(self.device).float()
                    
        total_scalar = 0
        for vec in L_vectors:
          scalar=torch.matmul(vec, mean_vec)
          total_scalar += scalar
        total_scalar /= L_vectors.shape[0] 
        total_scalar.backward()     

        for n, p in self.model.named_parameters():                      
          precision_matrices[n].data += p.grad**2 / num_data      
              
    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      _loss = self._precision_matrices[n] * (p - self._state_parameters[n]) ** 2
      loss += _loss.sum()
    return loss
  
  def update(self, model):
    # do nothing
    return
