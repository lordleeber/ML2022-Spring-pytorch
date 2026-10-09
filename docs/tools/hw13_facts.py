# HW13 facts: parameter counts, teacher accuracy, metric check of the best student, KD loss check.
# usage (from HW13/): python ../docs/tools/hw13_facts.py [exp_name]
import io
import sys
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

sys.path.insert(0, '.')
from config import cfg
from dataset import FoodDataset, test_tfm
from model import get_student_model, get_teacher_model
from kd import loss_fn_kd

exp_name = sys.argv[1] if len(sys.argv) > 1 else cfg['exp_name']
device = 'cuda'


def counts(model):
  params = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  buffers = sum(b.numel() for name, b in model.named_buffers())
  buf_float = sum(b.numel() for name, b in model.named_buffers() if b.is_floating_point())
  buf = io.StringIO()
  with redirect_stdout(buf):
    summary(model, (3, 224, 224), device='cpu')
  total_line = [l for l in buf.getvalue().splitlines() if l.startswith('Total params')][0]
  return dict(params=params, trainable=trainable, buffers=buffers, float_buffers=buf_float, torchsummary=total_line)


@torch.no_grad()
def evaluate(model, loader):
  model.eval().to(device)
  correct, loss_sum, n, logits_all = 0, 0.0, 0, []
  for imgs, labels in loader:
    imgs, labels = imgs.to(device), labels.to(device)
    logits = model(imgs)
    loss_sum += F.cross_entropy(logits, labels, reduction='sum').item()
    correct += (logits.argmax(-1) == labels).sum().item()
    n += len(imgs)
    logits_all.append(logits.cpu())
  return correct / n, loss_sum / n, n, torch.cat(logits_all)


student = get_student_model()
teacher = get_teacher_model(cfg['dataset_root'])
print('student', counts(student))
print('teacher', counts(teacher))

valid_set = FoodDataset(f"{cfg['dataset_root']}/validation", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)

acc, loss, n, t_logits = evaluate(teacher, valid_loader)
print(f'teacher valid: acc={acc:.5f} ({round(acc * n)}/{n}) ce={loss:.5f}')

student.load_state_dict(torch.load(f"{cfg['save_dir']}/{exp_name}/student_best.ckpt", map_location='cpu'))
acc, loss, n, s_logits = evaluate(student, valid_loader)
print(f'student best ({exp_name}) valid: acc={acc:.5f} ({round(acc * n)}/{n}) ce={loss:.5f}')

# KD loss check against nn.KLDivLoss (the doc example: input = log-probs of the model, target = probs)
labels = torch.tensor([int(f.split('/')[-1].split('_')[0]) for f in valid_set.files])
s, t = s_logits[:64], t_logits[:64]
for alpha, T in [(0.5, 1.0), (0.5, 4.0), (0.0, 1.0), (1.0, 1.0)]:
  mine = loss_fn_kd(s, labels[:64], t, alpha=alpha, temperature=T).item()
  ref = (alpha * T * T * nn.KLDivLoss(reduction='batchmean')(F.log_softmax(s / T, -1), F.softmax(t / T, -1))
         + (1 - alpha) * nn.CrossEntropyLoss()(s, labels[:64])).item()
  print(f'loss_fn_kd alpha={alpha} T={T}: {mine:.6f} ref {ref:.6f}')
