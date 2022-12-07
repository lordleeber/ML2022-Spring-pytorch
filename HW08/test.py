import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
from dataset import CustomTensorDataset
from conv_autoencoder import CONV_AUTOENCODER
from fcn_autoencoder import FCN_AUTOENCODER
from vae import VAE


"""# Inference
Model is loaded and generates its anomaly score predictions.

## Initialize
- dataloader
- model
- prediction file
"""


if __name__ == "__main__":
    test = np.load('data/testingset.npy', allow_pickle=True)
    print(test.shape)
    print(f"len(test) = {len(test)}")

    eval_batch_size = 200

    # build testing dataloader
    data = torch.tensor(test, dtype=torch.float32)
    test_dataset = CustomTensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
    eval_loss = nn.MSELoss(reduction='none')

    # load trained model
    model_type = 'cnn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
    checkpoint_path = f'last_model_{model_type}.pt'
    model = torch.load(checkpoint_path)
    model.eval()

    # prediction file
    out_file = 'prediction.csv'

    anomality = list()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            img = data.float().cuda()
            if model_type in ['fcn']:
                img = img.view(img.shape[0], -1)
            output = model(img)
            if model_type in ['vae']:
                output = output[0]
            if model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])
            anomality.append(loss)
    anomality = torch.cat(anomality, axis=0)
    # anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()
    anomality1 = torch.sqrt(anomality)
    anomality2 = anomality1.reshape(len(anomality), 1)
    anomality3 = anomality2.cpu()
    anomality4 = anomality3.numpy()

    df = pd.DataFrame(anomality4, columns=['score'])
    df.to_csv(out_file, index_label='ID')
