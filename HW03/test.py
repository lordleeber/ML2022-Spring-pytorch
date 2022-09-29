import os
import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from classifier import Classifier
from dataset import FoodDataset
from config import *
from torchvision.datasets import DatasetFolder, VisionDataset
import torch
import torchvision.transforms as transforms


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

test_set = FoodDataset(os.path.join(dataset_dir, "test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
device = "cuda"


"""# Testing and generate prediction CSV"""

model_best = Classifier().to(device)
model_best.load_state_dict(torch.load(f"{exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)


df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv", index=False)
