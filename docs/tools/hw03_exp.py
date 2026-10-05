"""Replicates HW03 train.py with switchable variants.

Run from HW03/ with PYTHONPATH=. so classifier/dataset/config import from the repo:
    cd HW03 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw03_exp.py --name orig
Prints the same lines as train.py (without tqdm and the log-file lines) and one JSON line at the end.
Nothing is written to disk unless --save / --dump is given, so sample_best.ckpt and submission.csv
are left alone.

Default arguments are the original train.py (Classifier, no augmentation, batch 256, 5 epochs,
Adam lr 3e-4 wd 1e-5, clip 10, seed 6666, both loaders shuffle=True). The RNG order follows
train.py: seeds -> datasets/loaders -> model -> optimizer -> epochs (each epoch draws from the
global torch RNG for the train shuffle, the train transforms and the valid shuffle). Extra
bookkeeping (exact accuracy over the whole validation set) adds no RNG draws.
Both loops are wrapped in tqdm (disabled) because train.py's `from tqdm.auto import tqdm` is
tqdm_asyncio outside notebooks, whose __init__ calls iter(loader) once more: a throwaway DataLoader
iterator that draws one extra number from the torch RNG per loop. Without it the shuffle differs.

Variants:
  --aug A       train_tfm with augmentation (see AUG below; test_tfm unchanged)
  --arch res    others.py Residual_Network with the skip connections of slides p.39 added
  --arch res0   others.py Residual_Network as given (forward without skips)
  --ls 0.1      CrossEntropyLoss(label_smoothing=0.1) (literal change of train.py:78)
  --resplit 1   move 2/3 of validation into training; validate on the other 1/3 (HOLDOUT)
"""
import argparse, copy, json, os, random, time
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from classifier import Classifier
from dataset import FoodDataset

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--arch', default='cnn')        # cnn (classifier.py) | res | res0
ap.add_argument('--aug', default='none')        # none (train.py) | A
ap.add_argument('--ls', type=float, default=0.0)
ap.add_argument('--resplit', type=int, default=0)
ap.add_argument('--seed', type=int, default=6666)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--max_batches', type=int, default=0)  # >0: stop each training epoch early (smoke test)
ap.add_argument('--save', default='')           # save the best state_dict here
ap.add_argument('--dump', default='')           # save best model's val/test logits (.npz) here
a = ap.parse_args()
dev = 'cuda'
t0 = time.time()
DS = './food11'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.cuda.manual_seed_all(a.seed)

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
AUG = {
    'none': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
    'A': transforms.Compose([
        transforms.RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
    ]),
}
train_tfm = AUG[a.aug]


def jpgs(d):
    return sorted([os.path.join(d, x) for x in os.listdir(d) if x.endswith('.jpg')])


val_files = jpgs(os.path.join(DS, 'validation'))
if a.resplit:
    # fixed split, independent of the torch RNG: 2/3 of validation -> training, 1/3 -> holdout
    idx = list(range(len(val_files)))
    random.Random(0).shuffle(idx)
    hold = sorted(idx[:len(idx) // 3])
    move = sorted(idx[len(idx) // 3:])
    train_files = jpgs(os.path.join(DS, 'training')) + [val_files[i] for i in move]
    train_set = FoodDataset(os.path.join(DS, 'training'), tfm=train_tfm, files=train_files)
    valid_set = FoodDataset(os.path.join(DS, 'validation'), tfm=test_tfm, files=[val_files[i] for i in hold])
else:
    train_set = FoodDataset(os.path.join(DS, 'training'), tfm=train_tfm)
    valid_set = FoodDataset(os.path.join(DS, 'validation'), tfm=test_tfm)
train_loader = DataLoader(train_set, batch_size=a.bs, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=a.bs, shuffle=True, num_workers=0, pin_memory=True)

if a.arch == 'cnn':
    model = Classifier()
else:
    # others.py cannot be imported (it uses `transforms` without importing it); exec the model part
    src = open('others.py').read()
    ns = {}
    exec(src[src.index('from torch import nn'):], ns)
    R = ns['Residual_Network']
    if a.arch == 'res':
        def forward(self, x):
            x1 = self.relu(self.cnn_layer1(x))
            x2 = self.relu(self.cnn_layer2(x1) + x1)
            x3 = self.relu(self.cnn_layer3(x2))
            x4 = self.relu(self.cnn_layer4(x3) + x3)
            x5 = self.relu(self.cnn_layer5(x4))
            x6 = self.relu(self.cnn_layer6(x5) + x5)
            return self.fc_layer(x6.flatten(1))
        R.forward = forward
    model = R()
model = model.to(dev)
nparams = sum(p.numel() for p in model.parameters())
criterion = nn.CrossEntropyLoss(label_smoothing=a.ls)
plain_ce = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)

best_acc, best_ep, best_sd, hist = 0, 0, None, []
for epoch in range(a.epochs):
    te = time.time()
    model.train()
    train_loss, train_accs = [], []
    for i, (imgs, labels) in enumerate(tqdm(train_loader, disable=True)):
        logits = model(imgs.to(dev))
        labels_on_device = labels.to(dev)
        loss = criterion(logits, labels_on_device)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = torch.eq(logits.argmax(dim=-1), labels_on_device).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
        if a.max_batches and i + 1 >= a.max_batches:
            break
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{a.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}", flush=True)
    t_train = time.time() - te

    model.eval()
    valid_loss, valid_accs = [], []
    n_ok, ce_sum = 0, 0.0
    for imgs, labels in tqdm(valid_loader, disable=True):
        with torch.no_grad():
            logits = model(imgs.to(dev))
            labels_on_device = labels.to(dev)
        loss = criterion(logits, labels_on_device)
        compare_result = torch.eq(logits.argmax(dim=-1), labels_on_device)
        acc = compare_result.float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        n_ok += compare_result.sum().item()
        ce_sum += plain_ce(logits, labels_on_device).item()
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{a.epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}", flush=True)
    row = dict(epoch=epoch + 1, train_loss=round(train_loss, 5), train_acc=round(train_acc.item(), 5),
               valid_loss=round(valid_loss, 5), valid_acc=round(valid_acc.item(), 5),
               true_acc=round(n_ok / len(valid_set), 5), true_ce=round(ce_sum / len(valid_set), 5),
               train_secs=round(t_train, 1), secs=round(time.time() - te, 1))
    hist.append(row)
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model", flush=True)
        best_sd = copy.deepcopy(model.state_dict())
        best_acc, best_ep = valid_acc, epoch + 1

# the best state on the whole validation set (and test set), in sorted file order, one pass
model.load_state_dict(best_sd)
model.eval()


def logits_of(ds):
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=a.bs, shuffle=False):
            out.append(model(x.to(dev)).cpu())
    return torch.cat(out)


full_val = FoodDataset(os.path.join(DS, 'validation'), tfm=test_tfm)
Lv = logits_of(full_val)
Yv = torch.tensor([int(os.path.basename(f).split('_')[0]) for f in full_val.files])
# with --resplit, 2/3 of full_val was trained on, so full_val_* is not a validation number there
res = dict(full_val_acc=round((Lv.argmax(1) == Yv).float().mean().item(), 5),
           full_val_ce=round(nn.functional.cross_entropy(Lv, Yv).item(), 5))
if a.dump:
    Lt = logits_of(FoodDataset(os.path.join(DS, 'test'), tfm=test_tfm))
    np.savez(a.dump, val=Lv.numpy(), val_y=Yv.numpy(), test=Lt.numpy())
if a.save:
    torch.save(best_sd, a.save)
out = dict(vars(a), nparams=nparams, n_train=len(train_set), n_val=len(valid_set),
           best_epoch=best_ep, printed_best=round(best_acc.item(), 5),
           best_true_acc=hist[best_ep - 1]['true_acc'], **res,
           hist=hist, secs=round(time.time() - t0, 1))
print(json.dumps(out), flush=True)
