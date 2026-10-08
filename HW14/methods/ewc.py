import torch.nn as nn
import torch.nn.functional as F


# EWC
class ewc(object):
  """
  @article{kirkpatrick2017overcoming,
      title={Overcoming catastrophic forgetting in neural networks},
      author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
      journal={Proceedings of the national academy of sciences},
      year={2017},
      url={https://arxiv.org/abs/1612.00796}
  }
  """
  def __init__(self, model, dataloader, device, prev_guards=[None]):
    self.model = model
    self.dataloader = dataloader
    self.device = device
    # extract all parameters in models
    self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
    
    # initialize parameters
    self.p_old = {}
    # save previous guards
    self.previous_guards_list = prev_guards

    # generate Fisher (F) matrix for EWC
    self._precision_matrices = self._calculate_importance()                   

    # keep the old parameter in self.p_old
    for n, p in self.params.items():
      self.p_old[n] = p.clone().detach()       

  def _calculate_importance(self):
    precision_matrices = {}
    # initialize Fisher (F) matrix（all fill zero）and add previous guards
    for n, p in self.params.items(): 
      precision_matrices[n] = p.clone().detach().fill_(0)                 
      for i in range(len(self.previous_guards_list)):
        if self.previous_guards_list[i]:
          precision_matrices[n] += self.previous_guards_list[i][n]

    self.model.eval()
    if self.dataloader is not None:
      number_data = len(self.dataloader)
      for data in self.dataloader:
        self.model.zero_grad()
        # get image data
        input = data[0].to(self.device)
          
        # image data forward model
        output = self.model(input)
          
        # Simply use groud truth label of dataset.  
        label = data[1].to(self.device)
          
        # generate Fisher(F) matrix for EWC     
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward()   

        for n, p in self.model.named_parameters():
          # get the gradient of each parameter and square it, then average it in all validation set.                          
          precision_matrices[n].data += p.grad.data ** 2 / number_data   
                                                                
      precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices

  def penalty(self, model: nn.Module):
    loss = 0
    for n, p in model.named_parameters():
      # generate the final regularization term by the ewc weight (self._precision_matrices[n]) and the square of weight difference ((p - self.p_old[n]) ** 2).  
      _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
      loss += _loss.sum()
    return loss
  
  def update(self, model):
    # do nothing
    return
