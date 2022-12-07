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


"""## Random seed
Set the random seed to a certain value for reproducibility.
"""

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


"""# Training

## Configuration
"""

if __name__ == "__main__":
    same_seeds(48763)
    train = np.load('data/trainingset.npy', allow_pickle=True)
    print(train.shape)

    # Training hyperparameters
    num_epochs = 50
    batch_size = 2000
    learning_rate = 1e-3

    # Build training dataloader
    x = torch.from_numpy(train)
    train_dataset = CustomTensorDataset(x)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    # Model
    model_type = 'cnn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
    model_classes = {'fcn': FCN_AUTOENCODER(), 'cnn': CONV_AUTOENCODER(), 'vae': VAE()}
    model = model_classes[model_type].cuda()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """## Training loop"""

    best_loss = np.inf
    model.train()

    qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
    for epoch in qqdm_train:
        tot_loss = list()
        for data in train_dataloader:

            # ===================loading=====================
            img = data.float().cuda()
            if model_type in ['fcn']:
                img = img.view(img.shape[0], -1)

            # ===================forward=====================
            output = model(img)
            if model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)

            tot_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, 'best_model_{}.pt'.format(model_type))
        # ===================log========================
        qqdm_train.set_infos({
            'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
            'loss': f'{mean_loss:.4f}',
        })
        # ===================save_last========================
        torch.save(model, 'last_model_{}.pt'.format(model_type))
