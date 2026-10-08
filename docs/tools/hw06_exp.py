"""Replicates HW06 train.py with switchable GAN / WGAN / WGAN-GP variants.

Run from HW06/ with PYTHONPATH=. so config/utils/trainer_gan import from the repo:
    cd HW06 && PYTHONPATH=. ../.venv/bin/python ../docs/tools/hw06_exp.py --name gan --out DIR
`import train` would start training at import time, so this file repeats train.py's module level
instead: utils.same_seeds(2022) -> utils.get_dataset (glob only, no RNG) -> TrainerGAN(config)
(G init, D init, z_samples) -> prepare_environment's DataLoader (batch 64, shuffle, 2 workers) ->
the loop of TrainerGAN.train. The loop below keeps every RNG draw of trainer_gan.py in the same
order (z for D, z for G, one iter() per epoch, the eval sample on z_samples), so with the defaults
the checkpoints are bit-identical to train.py's. Variant changes are applied after TrainerGAN() has
built the models, so they draw nothing and every variant starts from the same initial weights:
  --model_type WGAN     drop the last Sigmoid of D, loss_D = -mean(D(r)) + mean(D(f)),
                        loss_G = -mean(D(f)), clip D weights to +-clip after each D step
  --model_type WGANGP   same losses, no clipping, + gp_lambda * (||grad D(interp)||_2 - 1)^2
                        (one torch.rand per D step for the interpolation; nothing else changes)
  --sigmoid 1           keep D's Sigmoid with a WGAN loss (the "only the loss was changed" case)
  --clip 0              WGAN without weight clipping
  --norm in|none        replace D's BatchNorm2d by InstanceNorm2d(affine=True) / Identity
  --detach 1            D step on f_imgs.detach(): G's graph is not backpropagated by loss_D
  --workers N           DataLoader num_workers (train.py: 2)
  --opt rmsprop --lr --beta1 --beta2 --n_critic --clip --gp_lambda --n_epoch
Logged without any RNG draw: loss_D, loss_G, mean D(real), mean D(fake) every --log_every steps,
and the L2 norm of every D conv weight's gradient (5 convs, input -> output) every --gn_every steps.
Writes to --out: G_{e}.pth / D_{e}.pth on train.py's schedule (e == 0 or (e+1) % 5 == 0),
Epoch_XXX.jpg sample grids, log.jsonl, and prints one JSON summary line at the end.
HW06/logs, checkpoints and output are left alone.
"""
import argparse, json, os, time
import torch, torch.nn as nn, torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from config import config
from trainer_gan import TrainerGAN

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='run')
ap.add_argument('--out', required=True)
ap.add_argument('--model_type', default='GAN')      # GAN (train.py) | WGAN | WGANGP
ap.add_argument('--sigmoid', type=int, default=-1)  # -1: GAN keeps it, WGAN/WGANGP drop it
ap.add_argument('--norm', default='bn')             # bn (discriminator.py) | in | none
ap.add_argument('--opt', default='adam')            # adam (trainer_gan.py) | rmsprop
ap.add_argument('--lr', type=float, default=config['lr'])
ap.add_argument('--beta1', type=float, default=0.5)
ap.add_argument('--beta2', type=float, default=0.999)
ap.add_argument('--n_epoch', type=int, default=config['n_epoch'])
ap.add_argument('--n_critic', type=int, default=config['n_critic'])
ap.add_argument('--clip', type=float, default=0.01)
ap.add_argument('--gp_lambda', type=float, default=10.0)
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--detach', type=int, default=0)     # 1: D step uses f_imgs.detach() (no backward into G)
ap.add_argument('--log_every', type=int, default=10)
ap.add_argument('--gn_every', type=int, default=100)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cfg = dict(config, lr=args.lr, n_epoch=args.n_epoch, n_critic=args.n_critic, model_type=args.model_type)

# --- train.py module level ---
utils.same_seeds(2022)
dataset = utils.get_dataset(os.path.join(cfg['workspace_dir'], 'faces'))
trainer = TrainerGAN(cfg)
G, D = trainer.G, trainer.D

# --- variants (no RNG draws) ---
wgan = args.model_type in ('WGAN', 'WGANGP')
keep_sigmoid = (not wgan) if args.sigmoid < 0 else bool(args.sigmoid)
if not keep_sigmoid:
    D.l1[-1] = nn.Identity()
if args.norm != 'bn':
    for blk in D.l1:
        if isinstance(blk, nn.Sequential):
            c = blk[1].num_features
            blk[1] = nn.InstanceNorm2d(c, affine=True) if args.norm == 'in' else nn.Identity()
# Rebuilt in every case so that replaced norm layers are optimized; a fresh Adam with the same
# settings is exactly trainer_gan.py's (constructing an optimizer draws no RNG).
if args.opt == 'rmsprop':
    opt_D = torch.optim.RMSprop(D.parameters(), lr=args.lr)
    opt_G = torch.optim.RMSprop(G.parameters(), lr=args.lr)
else:
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
convs = [m for m in D.modules() if isinstance(m, nn.Conv2d)]

# --- prepare_environment (same DataLoader, models to GPU) ---
dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=args.workers)
G, D = G.cuda(), D.cuda()
trainer.G, trainer.D = G, D
G.train(); D.train()


def gradient_penalty(r_imgs, f_imgs):
    alpha = torch.rand(r_imgs.size(0), 1, 1, 1).cuda()
    interp = (alpha * r_imgs + (1 - alpha) * f_imgs.detach()).requires_grad_(True)
    out = D(interp)
    grad = torch.autograd.grad(out, interp, grad_outputs=torch.ones_like(out), create_graph=True)[0]
    return args.gp_lambda * ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


loss = trainer.loss
logf = open(os.path.join(args.out, 'log.jsonl'), 'w')
steps, t0 = 0, time.time()
for e, epoch in enumerate(range(cfg['n_epoch'])):
    te = time.time()
    for i, data in enumerate(tqdm(dataloader, disable=True)):
        imgs = data.cuda()
        bs = imgs.size(0)

        # Train D
        z = Variable(torch.randn(bs, cfg['z_dim'])).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()
        r_logit = D(r_imgs)
        f_logit = D(f_imgs.detach() if args.detach else f_imgs)
        if args.model_type == 'GAN':
            r_loss = loss(r_logit, r_label)
            f_loss = loss(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2
        else:
            loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
            if args.model_type == 'WGANGP':
                loss_D = loss_D + gradient_penalty(r_imgs, f_imgs)
        D.zero_grad()
        loss_D.backward()
        if steps % args.gn_every == 0:
            gn = [c.weight.grad.norm(2).item() for c in convs]
        opt_D.step()
        if args.model_type == 'WGAN' and args.clip > 0:
            for p in D.parameters():
                p.data.clamp_(-args.clip, args.clip)

        # Train G
        if steps % cfg['n_critic'] == 0:
            z = Variable(torch.randn(bs, cfg['z_dim'])).cuda()
            f_imgs = G(z)
            f_logit = D(f_imgs)
            loss_G = loss(f_logit, r_label) if args.model_type == 'GAN' else -torch.mean(D(f_imgs))
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if steps % args.log_every == 0:
            rec = dict(step=steps, epoch=e + 1, loss_D=loss_D.item(), loss_G=loss_G.item(),
                       d_real=r_logit.mean().item(), d_fake=f_logit.mean().item())
            if steps % args.gn_every == 0:
                rec['gn'] = gn
            logf.write(json.dumps(rec) + '\n')
        steps += 1

    G.eval()
    f_imgs_sample = (G(trainer.z_samples).data + 1) / 2.0
    torchvision.utils.save_image(f_imgs_sample, os.path.join(args.out, f'Epoch_{epoch + 1:03d}.jpg'), nrow=10)
    G.train()
    if (e + 1) % 5 == 0 or e == 0:
        torch.save(G.state_dict(), os.path.join(args.out, f'G_{e}.pth'))
        torch.save(D.state_dict(), os.path.join(args.out, f'D_{e}.pth'))
    logf.write(json.dumps(dict(epoch_end=e + 1, steps=steps, sec=round(time.time() - te, 2))) + '\n')
    logf.flush()
    print(f'epoch {e + 1} steps {steps} loss_D {loss_D.item():.4f} loss_G {loss_G.item():.4f} '
          f'{time.time() - te:.1f}s', flush=True)

logf.close()
print(json.dumps(dict(name=args.name, args=vars(args), steps=steps, sec=round(time.time() - t0, 1),
                      final_loss_D=loss_D.item(), final_loss_G=loss_G.item())))
