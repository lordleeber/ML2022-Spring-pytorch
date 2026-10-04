"""Replicates HW02 train.py's trainer with switchable variants.

Run from HW02/ with PYTHONPATH=. so utils/model/data_loader import from the repo:
    cd HW02 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw02_exp.py --name orig
Prints one line per epoch (same format as train.py) and one JSON line at the end.
Nothing is written to disk unless --save is given, so model.ckpt and prediction.csv are left alone.

Default arguments are the original train.py (10-layer LSTM, concat 11, batch 64, 20 epochs,
loss on all 11 positions, accuracy on the middle one). The RNG order follows train.py:
preprocess (python random, seed 1337) -> DataLoaders -> same_seeds(seed) -> model -> AdamW.

--arch dnn is model_dnn.py (the official sample code's model); with --concat 1 --bs 512
--epochs 5 --hidden 256 --layers 1 --ratio 0.8 it is the official notebook's setting.
"""
import argparse, copy, gc, json, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from utils import preprocess_data, same_seeds
from data_loader import LibriDataset

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--arch', default='lstm')          # lstm (model.py) | bilstm | dnn (model_dnn.py)
ap.add_argument('--layers', type=int, default=10)  # lstm/bilstm: num_layers; dnn: hidden_layers
ap.add_argument('--hidden', type=int, default=512)
ap.add_argument('--dropout', type=float, default=None)  # lstm/bilstm: between layers (default 0.5); dnn: after each block (default 0)
ap.add_argument('--bn', type=int, default=0)       # dnn only: BatchNorm1d after each Linear
ap.add_argument('--concat', type=int, default=11)
ap.add_argument('--ratio', type=float, default=0.9)
ap.add_argument('--loss', default='all')           # all: 11 positions (train.py) | mid: middle frame only
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--max_batches', type=int, default=0)  # >0: stop each training epoch early (smoke test)
ap.add_argument('--save', default='')              # save the best state_dict here
a = ap.parse_args()
if a.dropout is None:
    a.dropout = 0.0 if a.arch == 'dnn' else 0.5
dev = 'cuda'
t0 = time.time()
C, mid = a.concat, a.concat // 2

train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=C, train_ratio=a.ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                               concat_nframes=C, train_ratio=a.ratio)
train_set, val_set = LibriDataset(train_X, train_y), LibriDataset(val_X, val_y)
del train_X, train_y, val_X, val_y
gc.collect()
train_loader = DataLoader(train_set, batch_size=a.bs, shuffle=True)
val_loader = DataLoader(val_set, batch_size=a.bs, shuffle=False)
t_data = time.time() - t0

same_seeds(a.seed)

if a.arch == 'lstm':
    if a.layers == 10 and a.dropout == 0.5:
        from model import Classifier
        model = Classifier(input_dim=39 * C, hidden_layers=1, hidden_dim=a.hidden)
    else:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=39, hidden_size=a.hidden, num_layers=a.layers,
                                    batch_first=True, dropout=a.dropout)
                self.out = nn.Linear(a.hidden, 41)
            def forward(self, x):
                return self.out(self.lstm(x, None)[0].contiguous())
        model = M()
elif a.arch == 'bilstm':
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=39, hidden_size=a.hidden, num_layers=a.layers,
                                batch_first=True, dropout=a.dropout, bidirectional=True)
            self.out = nn.Linear(2 * a.hidden, 41)
        def forward(self, x):
            return self.out(self.lstm(x, None)[0].contiguous())
    model = M()
else:
    if a.dropout == 0 and not a.bn:
        from model_dnn import Classifier
        model = Classifier(input_dim=39 * C, hidden_layers=a.layers, hidden_dim=a.hidden)
    else:
        def block(i, o):
            m = [nn.Linear(i, o)] + ([nn.BatchNorm1d(o)] if a.bn else []) + [nn.ReLU()]
            return m + ([nn.Dropout(a.dropout)] if a.dropout else [])
        layers = block(39 * C, a.hidden)
        for _ in range(a.layers):
            layers += block(a.hidden, a.hidden)
        model = nn.Sequential(*layers, nn.Linear(a.hidden, 41))
model = model.to(dev)
nparams = sum(p.numel() for p in model.parameters())
seq = a.arch != 'dnn'
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)


def step(features, labels):
    """Returns (loss, middle-frame logits, middle-frame labels), as train.py computes them."""
    if seq:
        features = features.view(-1, C, 39)
        outputs = model(features)
        if a.loss == 'all':
            loss = criterion(outputs.view(-1, 41), labels.view(-1))
        else:
            loss = criterion(outputs[:, mid, :], labels[:, mid])
        return loss, outputs[:, mid, :].view(outputs.shape[0], -1), labels[:, mid]
    outputs = model(features)
    labels = labels[:, mid]
    return criterion(outputs, labels), outputs, labels


best_acc, best_ep, best_sd, hist = 0.0, 0, None, []
for epoch in range(a.epochs):
    te = time.time()
    train_acc = train_loss = val_acc = val_loss = 0.0
    model.train()
    nb = 0
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(dev), labels.to(dev)
        optimizer.zero_grad()
        loss, outputs, labels = step(features, labels)
        loss.backward()
        optimizer.step()
        _, train_pred = torch.max(outputs, 1)
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
        nb += 1
        if a.max_batches and nb >= a.max_batches:
            break
    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(dev), labels.to(dev)
            loss, outputs, labels = step(features, labels)
            _, val_pred = torch.max(outputs, 1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()
    row = dict(epoch=epoch + 1, train_acc=train_acc / len(train_set), train_loss=train_loss / len(train_loader),
               val_acc=val_acc / len(val_set), val_loss=val_loss / len(val_loader), secs=round(time.time() - te, 1))
    hist.append(row)
    print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}  ({:.0f}s)'.format(
        epoch + 1, a.epochs, row['train_acc'], row['train_loss'], row['val_acc'], row['val_loss'], row['secs']), flush=True)
    if val_acc > best_acc:
        best_acc, best_ep = val_acc, epoch + 1
        best_sd = copy.deepcopy(model.state_dict())

# re-evaluate the best state on the whole validation set in one pass, per position
model.load_state_dict(best_sd)
model.eval()
X, Y = val_set.data, val_set.label
correct, pos = 0, torch.zeros(C if seq else 1)
with torch.no_grad():
    for i in range(0, len(X), 4096):
        x, y = X[i:i + 4096].to(dev), Y[i:i + 4096]
        if seq:
            pred = model(x.view(-1, C, 39)).argmax(-1).cpu()
            pos += (pred == y).sum(0)
            correct += (pred[:, mid] == y[:, mid]).sum().item()
        else:
            correct += (model(x).argmax(-1).cpu() == y[:, mid]).sum().item()
if a.save:
    torch.save(best_sd, a.save)
out = dict(vars(a), nparams=nparams, n_train=len(train_set), n_val=len(val_set),
           best_epoch=best_ep, printed_best=round(best_acc / len(val_set), 6),
           true_val_acc=round(correct / len(X), 6),
           pos_acc=[round(v, 4) for v in (pos / len(X)).tolist()] if seq else None,
           hist=hist, data_secs=round(t_data, 1), secs=round(time.time() - t0, 1))
print(json.dumps(out))
