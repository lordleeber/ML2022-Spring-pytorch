# HW13 ch05: the teacher on AUGMENTED training images (what it sees during grid B's distillation).
# usage (from HW13/): python ../docs/tools/hw13_teacher_aug.py   (one pass, torch.manual_seed(0), 8 workers)
import io, os, sys
from contextlib import redirect_stdout
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import FoodDataset, train_tfm, normalize
import torchvision.transforms as transforms
from model import get_teacher_model

hw03 = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), transforms.ToTensor(), normalize])
t = get_teacher_model(cfg['dataset_root']).cuda().eval()
for name, tfm in [('sample train_tfm (flip)', train_tfm), ('hw03 augmentation', hw03)]:
  torch.manual_seed(0)
  with redirect_stdout(io.StringIO()):
    ds = FoodDataset(f"{cfg['dataset_root']}/training", tfm=tfm)
  zs, ys = [], []
  with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=128, num_workers=8):
      zs.append(t(x.cuda()).cpu()); ys.append(y)
  z, y = torch.cat(zs), torch.cat(ys)
  p = z.softmax(-1); q2 = (z / 2).softmax(-1)
  print(f'{name}: acc {(p.argmax(-1) == y).float().mean().item():.5f}  mean max prob T=1 {p.max(-1).values.mean().item():.4f}  '
        f'>0.99 {(p.max(-1).values > 0.99).float().mean().item():.4f}  mass on other classes T=1 {1 - p[torch.arange(len(y)), y].mean().item():.4f}  '
        f'T=2 {1 - q2[torch.arange(len(y)), y].mean().item():.4f}', flush=True)
sys.stdout.flush(); os._exit(0)
