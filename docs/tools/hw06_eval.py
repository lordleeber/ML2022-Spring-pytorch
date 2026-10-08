"""FID and AFD for HW06 generators (the JudgeBoi evaluator is not public; this is an approximation).

Needs pytorch-fid 0.3.0 and opencv-python-headless<5 (5.x has no CascadeClassifier) outside the venv (PYTHONPATH=<pylib>), the
Inception weights pt_inception-2015-12-05-6726825d.pth in ~/.cache/torch/hub/checkpoints, and
nagadomi's lbpcascade_animeface.xml (--cascade). Run from HW06/ with PYTHONPATH=.:<pylib>.

  real --stats DIR      Inception (pool3, 2048-d) mean/cov of all 71,314 faces, twice:
                        real64 = utils.get_dataset's transform (64x64, what G learns to imitate)
                        real96 = the original 96x96 files. Also the AFD rate of both.
  gen --stats DIR G...  for each generator checkpoint: --n images from a fixed z
                        (torch.Generator seeded --seed), written as JPEG with
                        torchvision.utils.save_image exactly like TrainerGAN.inference, read back
                        (what a submission contains), then FID vs real64 / real96 and AFD.
FID: pytorch_fid.fid_score.calculate_frechet_distance copied below, because scipy 1.18's sqrtm no
longer takes disp= (TypeError in pytorch-fid 0.3.0); the arithmetic is unchanged. AFD: fraction of images where
cv2.CascadeClassifier.detectMultiScale(equalizeHist(gray), 1.1, 5, minSize=(24, 24)) finds >= 1
face (the parameters of the lbpcascade_animeface README example).
  sg2 --stats DIR N...  StyleGAN2 checkpoints models/<sg2_name>/model_N.pt under --sg2_dir: --n images
                        from the EMA mapping/generator (SE, GE) with truncation --psi, the package's
                        own generate_truncated; global RNG seeded --seed (z, per-pixel noise and the
                        2000 z of the truncation mean all come from it). Needs <sg2lib> on PYTHONPATH.
One JSON line per result on stdout.
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np
import torch, torchvision
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from pytorch_fid.inception import InceptionV3
from scipy import linalg

ap = argparse.ArgumentParser()
ap.add_argument('cmd', choices=['real', 'gen', 'sg2'])
ap.add_argument('G', nargs='*')
ap.add_argument('--stats', required=True)
ap.add_argument('--cascade', required=True)
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--keep', default='')    # gen: keep the JPEGs of each checkpoint under this dir
ap.add_argument('--bs', type=int, default=250)
ap.add_argument('--sg2_dir', default='')       # sg2: the work dir of hw06_sg2_train.sh (holds models/<name>)
ap.add_argument('--sg2_name', default='sg2')
ap.add_argument('--sg2_every', type=int, default=2500)  # sg2: --save_every of hw06_sg2_train.sh
ap.add_argument('--psi', type=float, default=0.75)  # sg2: truncation psi (0.75 = the package's CLI default)
args = ap.parse_intermixed_args()

inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
cascade = cv2.CascadeClassifier(args.cascade)
assert not cascade.empty()


@torch.no_grad()
def activations(batches):
    out = []
    for x in batches:  # float in [0, 1], N x 3 x H x W
        out.append(inception(x.cuda())[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(out).numpy()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        print('fid calculation produces singular product; adding %s to diagonal of cov estimates' % eps,
              file=sys.stderr)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError('Imaginary component {}'.format(np.max(np.abs(covmean.imag))))
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def stats(act):
    return act.mean(0), np.cov(act, rowvar=False)


def has_face(rgb_uint8):  # H x W x 3, uint8
    gray = cv2.equalizeHist(cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY))
    return len(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))) > 0


def load(path):
    return np.asarray(Image.open(path).convert('RGB'))


def file_batches(paths, fn):
    for i in range(0, len(paths), args.bs):
        yield torch.stack([fn(p) for p in paths[i:i + args.bs]])


os.makedirs(args.stats, exist_ok=True)
if args.cmd == 'real':
    import utils
    ds = utils.get_dataset('faces')
    paths = sorted(ds.fnames)
    to64 = lambda p: ds.transform(torchvision.io.read_image(p)) * 0.5 + 0.5   # back to [0, 1]
    to96 = lambda p: TF.to_tensor(Image.open(p).convert('RGB'))
    for name, fn in (('real64', to64), ('real96', to96)):
        act = activations(file_batches(paths, fn))
        mu, sigma = stats(act)
        np.savez(os.path.join(args.stats, f'{name}.npz'), mu=mu, sigma=sigma)
        faces = sum(has_face((fn(p).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)) for p in paths)
        print(json.dumps(dict(set=name, n=len(paths), afd=faces / len(paths), afd_count=int(faces))), flush=True)
    sys.exit()

ref = {k: np.load(os.path.join(args.stats, f'{k}.npz')) for k in ('real64', 'real96')}


def score(imgs, tag, **info):  # imgs: n x 3 x 64 x 64 in [0, 1] on the CPU
    d = os.path.join(args.keep, tag) if args.keep else tempfile.mkdtemp()
    os.makedirs(d, exist_ok=True)
    paths = [os.path.join(d, f'{i + 1}.jpg') for i in range(args.n)]
    for i, p in enumerate(paths):
        torchvision.utils.save_image(imgs[i], p)
    act = activations(file_batches(paths, lambda p: TF.to_tensor(Image.open(p).convert('RGB'))))
    mu, sigma = stats(act)
    faces = sum(has_face(load(p)) for p in paths)
    rec = dict(info, n=args.n, seed=args.seed,
               fid64=float(calculate_frechet_distance(mu, sigma, ref['real64']['mu'], ref['real64']['sigma'])),
               fid96=float(calculate_frechet_distance(mu, sigma, ref['real96']['mu'], ref['real96']['sigma'])),
               afd=faces / args.n, afd_count=int(faces))
    print(json.dumps(rec), flush=True)
    if not args.keep:
        for p in paths:
            os.remove(p)
        os.rmdir(d)


if args.cmd == 'sg2':
    from stylegan2_pytorch.stylegan2_pytorch import Trainer, noise_list, image_noise
    for num in args.G:
        T = Trainer(name=args.sg2_name, base_dir=args.sg2_dir, image_size=64, batch_size=args.bs)
        T.load(int(num))
        T.GAN.eval()
        L = T.GAN.GE.num_layers
        torch.manual_seed(args.seed)  # noise() / image_noise() / truncate_style's 2000 z use the global RNG
        with torch.no_grad():
            imgs = torch.cat([T.generate_truncated(T.GAN.SE, T.GAN.GE, noise_list(min(args.bs, args.n - i), L, 512, 0),
                                                   image_noise(min(args.bs, args.n - i), 64, 0), trunc_psi=args.psi)
                              for i in range(0, args.n, args.bs)]).cpu()
        score(imgs, f'{args.sg2_name}_model_{num}_psi{args.psi}', model=f'{args.sg2_name}/model_{num}.pt',
              step=int(num) * args.sg2_every, psi=args.psi)
    sys.exit()

from generator import Generator
for gpath in args.G:
    G = Generator(100).cuda()
    G.load_state_dict(torch.load(gpath, map_location='cuda'))
    G.eval()
    z = torch.randn(args.n, 100, generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        imgs = torch.cat([(G(z[i:i + args.bs].cuda()).data + 1) / 2.0 for i in range(0, args.n, args.bs)]).cpu()
    tag = os.path.basename(os.path.dirname(os.path.abspath(gpath))) + '_' + os.path.basename(gpath)[:-4]
    score(imgs, tag, G=gpath)
