# HW14 LifeLong Learning: runs the six methods of HW14.ipynb in the notebook's order.
# All methods share one random stream (as in the notebook), so running a subset
# with --methods gives different numbers from a full run.
import argparse
import json
import os

import torch
import tqdm
import tqdm.auto
from torch.utils.data import DataLoader

from config import args, angle_list
from dataset import Data
from model import Model
from trainer import train, evaluate
from utils import same_seeds
from methods.baseline import baseline
from methods.ewc import ewc
from methods.mas import mas
from methods.si import si
from methods.rwalk import rwalk
from methods.scp import scp


def run_baseline(train_dataloaders, test_dataloaders, device):
  # Baseline
  print("RUN BASELINE")
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # initialize lifelong learning object (baseline class) without adding any regularization term.
  lll_object=baseline(model=model, dataloader=None, device=device)
  lll_lambda=0.0
  baseline_acc=[]
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

  # iterate training on each task continually.
  for train_indexes in task_bar:
    # Train each task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, 
                    lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
    # get model weight to baseline class and do nothing!
    lll_object=baseline(model=model, dataloader=train_dataloaders[train_indexes],device=device)
    
    # new a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Collect average accuracy in each epoch
    baseline_acc.extend(acc_list)
    
    # display the information of the next task.
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch! 
  print(baseline_acc)
  print("==================================================================================================")
  return baseline_acc


def run_ewc(train_dataloaders, test_dataloaders, device):
  # EWC
  print("RUN EWC")
  model = Model()
  model = model.to(device)
  # initialize optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  # initialize lifelong learning object for EWC
  lll_object=ewc(model=model, dataloader=None, device=device)

  # setup the coefficient value of regularization term.
  lll_lambda=100
  ewc_acc= []
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
  prev_guards = []

  # iterate training on each task continually.
  for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
    # get model weight and calculate guidance for each weight
    prev_guards.append(lll_object._precision_matrices)
    lll_object=ewc(model=model, dataloader=train_dataloaders[train_indexes], device=device, prev_guards=prev_guards)

    # new a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # collect average accuracy in each epoch
    ewc_acc.extend(acc_list)

    # Update tqdm displayer
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch!     
  print(ewc_acc)
  print("==================================================================================================")
  return ewc_acc


def run_mas(train_dataloaders, test_dataloaders, device):
  # MAS
  print("RUN MAS")
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  lll_object=mas(model=model, dataloader=None, device=device)
  lll_lambda=0.1
  mas_acc= []
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
  prev_guards = []

  for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
    # get model weight and calculate guidance for each weight
    prev_guards.append(lll_object._precision_matrices)
    lll_object=mas(model=model, dataloader=train_dataloaders[train_indexes], device=device, prev_guards=prev_guards)

    # New a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    mas_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch!     
  print(mas_acc)
  print("==================================================================================================")
  return mas_acc


def run_si(train_dataloaders, test_dataloaders, device):
  # SI
  print("RUN SI")
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  lll_object=si(model=model, dataloader=None, epsilon=0.1, device=device)
  lll_lambda=1
  si_acc = []
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

  for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
    # get model weight and calculate guidance for each weight
    lll_object=si(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device)

    # New a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    si_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch!     
  print(si_acc)
  print("==================================================================================================")
  return si_acc


def run_rwalk(train_dataloaders, test_dataloaders, device):
  # RWalk
  print("RUN Rwalk")
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  lll_object=rwalk(model=model, dataloader=None, epsilon=0.1, device=device)
  lll_lambda=100
  rwalk_acc = []
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
  prev_guards = []

  for train_indexes in task_bar:
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    prev_guards.append(lll_object._precision_matrices)
    lll_object=rwalk(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device, prev_guards=prev_guards)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rwalk_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch!     
  print(rwalk_acc)
  print("==================================================================================================")
  return rwalk_acc


def run_scp(train_dataloaders, test_dataloaders, device):
  # SCP
  print("RUN SLICE CRAMER PRESERVATION")
  model = Model()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  lll_object=scp(model=model, dataloader=None, L=100, device=device)
  lll_lambda=100
  scp_acc= []
  task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
  prev_guards = []

  for train_indexes in task_bar:
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    prev_guards.append(lll_object._precision_matrices)
    lll_object=scp(model=model, dataloader=train_dataloaders[train_indexes], L=100, device=device, prev_guards=prev_guards)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scp_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

  # average accuracy in each task per epoch!     
  print(scp_acc)
  print("==================================================================================================")
  return scp_acc


METHODS = {
  'baseline': run_baseline,
  'EWC': run_ewc,
  'MAS': run_mas,
  'SI': run_si,
  'RWALK': run_rwalk,
  'SCP': run_scp,
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
  parser.add_argument('--out', default='output')
  opt = parser.parse_args()

  same_seeds(0)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # prepare rotated MNIST datasets.
  train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
  train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]

  test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
  test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]

  # the notebook builds one example model here, which also draws from the random stream
  example = Model()
  print(example)

  os.makedirs(opt.out, exist_ok=True)
  results = {}
  for name in opt.methods:
    results[name] = METHODS[name](train_dataloaders, test_dataloaders, device)
    with open(os.path.join(opt.out, 'acc.json'), 'w') as f:
      json.dump(results, f, indent=1)


if __name__ == '__main__':
  main()
