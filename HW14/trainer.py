import numpy as np
import torch
import torch.nn as nn
import tqdm
import tqdm.auto


def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, log_step=1):
  model.train()
  model.zero_grad()
  objective = nn.CrossEntropyLoss()
  acc_per_epoch = []
  loss = 1.0
  bar = tqdm.auto.trange(epochs_per_task, leave=False, desc=f"Epoch 1, Loss: {loss:.7f}")
  for epoch in bar:
    for imgs, labels in tqdm.auto.tqdm(dataloader, leave=False):            
      imgs, labels = imgs.to(device), labels.to(device)
      outputs = model(imgs)
      loss = objective(outputs, labels)
      total_loss = loss
      lll_loss = lll_object.penalty(model)
      total_loss += lll_lambda * lll_loss 
      lll_object.update(model)
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

      loss = total_loss.item()
      bar.set_description_str(desc=f"Epoch {epoch+1:2}, Loss: {loss:.7f}", refresh=True)
    acc_average  = []
    for test_dataloader in test_dataloaders: 
      acc_test = evaluate(model, test_dataloader, device)
      acc_average.append(acc_test)
    average=np.mean(np.array(acc_average))
    acc_per_epoch.append(average*100.0)
    bar.set_description_str(desc=f"Epoch {epoch+2:2}, Loss: {loss:.7f}", refresh=True)
                
  return model, optimizer, acc_per_epoch


def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total
