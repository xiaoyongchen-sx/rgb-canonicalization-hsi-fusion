"""Second backbone (UAFL, CVPR 2026) — feed-forward, non-diffusion.

Plan B integration: UAFL runs as a standalone feed-forward fusion backbone,
REUSING the project's dataloader (P_N_V2 synthesis, degradation D, Harvard
split) so only the model differs from the EMR-Diff diffusion baseline.

  input : x = bicubic-upsampled LR-HSI (31ch), ref = HR-RGB (3ch)
  output: HR-HSI (31ch)   [UAFL: SVD unmixing -> abundance fusion -> mix]
  loss  : SmoothL1 (as in the official UAFL)

The deformable-alignment (DCN) module is replaced by RefRefine (registered
setting; see arch/uafl_model.py). `--canonicalize` inserts the LR-HSI-anchored
closed-form RGB canonicalizer BEFORE the ref input (step-2 plug-and-play test).

Usage:
  # train
  python scripts/train_test_uafl.py --mode train --epochs 2000 \
    --name Harvard/uafl_baseline
  # test (raw) / test with canonicalizer
  python scripts/train_test_uafl.py --mode test  --ckpt <path> [--canonicalize]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import (
    PeakSignalNoiseRatio, SpectralAngleMapper,
    MultiScaleStructuralSimilarityIndexMeasure,
    ErrorRelativeGlobalDimensionlessSynthesis,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arch.uafl_model import UAFL
from arch.BAFUnet import FixedSRFRGBCanonicalizer
from dataset_loader.dataloader import DataloaderSimpleTrain, DataloaderSimpleTest
from scripts.rgb_filters_citable import apply_rgb_filter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('train', 'test'), default='train')
    p.add_argument('--name', default='Harvard/uafl_baseline')
    p.add_argument('--ckpt', default=None, help='checkpoint for test / resume')
    p.add_argument('--srf', default='SRF/P_N_V2.mat')
    p.add_argument('--sf', type=int, default=8)
    p.add_argument('--gt-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--dim', type=int, default=28)
    p.add_argument('--stage', type=int, default=2)
    p.add_argument('--numend', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=20260721)
    p.add_argument('--max-samples', type=int, default=20)
    # test-time shift + canonicalizer
    p.add_argument('--canonicalize', action='store_true')
    p.add_argument('--filter', default=None,
                   help='apply a single RGB filter at test time (e.g. gamma_encode)')
    p.add_argument('--profile', default=None,
                   help='apply a full-chain composite profile (A_indoor/B_outdoor/C_stress)')
    p.add_argument('--camera', default=None,
                   help='layer-2 SRF shift: rebuild the guide RGB by projecting GT '
                        'with this camera SRF (name in SRF/test, or a .mat path). '
                        'The canonicalizer target stays --srf, as in the EMR-Diff '
                        'cross-camera protocol.')
    p.add_argument('--camera-sweep', action='store_true',
                   help='evaluate every SRF/test/*_harvard.mat in one process')
    p.add_argument('--val-every', type=int, default=200)
    p.add_argument('--eval-crop', type=int, default=0,
                   help='center-crop HR size for eval (0=full image). '
                        'Full-res UAFL eval is slow; use e.g. 512 for speed.')
    return p.parse_args()


def build_model(args, device):
    return UAFL(dim=args.dim, stage=args.stage,
               num_blocks=[2] * (args.stage + 1), numend=args.numend).to(device)


def build_canonicalizer(args, device):
    srf = torch.from_numpy(sio.loadmat(ROOT / args.srf)['P_20N']).float()
    canon = FixedSRFRGBCanonicalizer(
        rgb_channels=3, hsi_channels=31, scale=args.sf,
        mode='global_forward_polynomial', polynomial_degree=3,
        normalize_target_srf=True, allow_signed_target_srf=True,
        clamp_output=True,
    ).to(device)
    canon.set_target_srf(srf.to(device))
    return canon


def make_loaders(args):
    common = dict(sf=args.sf, srf_path=str(ROOT / args.srf), srf_mode='file',
                  band_dim=31)
    train_opt = dict(common, paths=[str(ROOT / 'dataset/Harvard/train')],
                     gt_size=args.gt_size)
    test_opt = dict(common, paths=[str(ROOT / 'dataset/Harvard/test')],
                    gt_size=args.gt_size, use_full_img=True)
    train_ds = DataloaderSimpleTrain(train_opt)
    test_ds = DataloaderSimpleTest(test_opt)
    if args.max_samples:
        test_ds.data = test_ds.data[:args.max_samples]
    return train_ds, test_ds


def upsample_lr(lq, size):
    return F.interpolate(lq, size=size, mode='bicubic', align_corners=False)


def load_srf_file(path):
    srf = torch.from_numpy(sio.loadmat(str(path))['P_20N']).float()
    assert tuple(srf.shape) == (3, 31), srf.shape
    return srf


def srf_to_rgb(hsi, srf):
    """Same projection the dataloader uses (plain matmul, no normalization)."""
    b, c, h, w = hsi.shape
    return torch.matmul(srf.to(hsi.device, hsi.dtype),
                        hsi.reshape(b, c, -1)).reshape(b, 3, h, w)


def resolve_camera(name):
    p = Path(name)
    if p.suffix == '.mat':
        return p if p.is_absolute() else ROOT / p
    return ROOT / 'SRF' / 'test' / f'{name}_harvard.mat'


@torch.no_grad()
def evaluate(model, test_ds, args, device, canon=None, profile_fn=None,
             camera_srf=None):
    model.eval()
    psnr = PeakSignalNoiseRatio().to(device)
    sam = SpectralAngleMapper().to(device)
    ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
    ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(device)
    tot = {'psnr': 0, 'ssim': 0, 'sam': 0, 'ergas': 0}
    n = len(test_ds)
    for i in range(n):
        gt, lq, rgb = test_ds[i]
        gt = gt.unsqueeze(0).to(device)
        lq = lq.unsqueeze(0).to(device)
        rgb = rgb.unsqueeze(0).to(device)
        if args.eval_crop:
            ch = min(args.eval_crop, gt.shape[-2])
            cw = min(args.eval_crop, gt.shape[-1])
            ch -= ch % 8; cw -= cw % 8  # align to LR grid (scale 8)
            sf = args.sf
            gt = gt[..., :ch, :cw]
            rgb = rgb[..., :ch, :cw]
            lq = lq[..., :ch // sf, :cw // sf]
        if camera_srf is not None:
            # layer-2 shift: guide RGB comes from a different camera's SRF
            rgb = srf_to_rgb(gt, camera_srf)
        if args.filter:
            rgb = apply_rgb_filter(rgb.squeeze(0), args.filter).unsqueeze(0)
        if profile_fn is not None:
            # full-chain composite: illuminant (spectral) x SRF x correction x post
            rgb = profile_fn(gt.squeeze(0)).unsqueeze(0)
        if canon is not None:
            rgb, _, _ = canon(rgb, lq)
        x = upsample_lr(lq, gt.shape[-2:])
        # pad to a multiple of window_size * 2**stage (window attention needs it)
        mult = 8 * (2 ** args.stage)
        H, W = x.shape[-2:]
        ph, pw = (-H) % mult, (-W) % mult
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='reflect')
            rgb = F.pad(rgb, (0, pw, 0, ph), mode='reflect')
        out = model(x, rgb)[..., :H, :W].clamp(0, 1)
        tot['psnr'] += psnr(out, gt).item()
        tot['ssim'] += ssim(out, gt).item()
        try:
            tot['sam'] += sam(out, gt).item()
        except Exception:
            pass
        tot['ergas'] += ergas(out, gt).item()
    return {k: v / n for k, v in tot.items()}


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out_dir = ROOT / 'output' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    canon = build_canonicalizer(args, device) if args.canonicalize else None
    train_ds, test_ds = make_loaders(args)
    print(f'UAFL params: {sum(p.numel() for p in model.parameters())}')

    if args.mode == 'test':
        if not args.ckpt:
            raise ValueError('--ckpt required for test')
        model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device)['model'])
        profile_fn = None
        if args.profile:
            from scripts.composite_pipeline import build_profile
            profile_fn = build_profile(args.profile, seed=args.seed)
            profile_fn.srf = profile_fn.srf.to(device)
            if profile_fn.illuminant is not None:
                profile_fn.illuminant = profile_fn.illuminant.to(device)
        if args.camera_sweep:
            cams = sorted((ROOT / 'SRF' / 'test').glob('*_harvard.mat'))
        elif args.camera:
            cams = [resolve_camera(args.camera)]
        else:
            cams = [None]
        rows = []
        for cam_path in cams:
            csrf = None if cam_path is None else load_srf_file(cam_path).to(device)
            m = evaluate(model, test_ds, args, device, canon, profile_fn, csrf)
            base = (cam_path.stem.replace('_harvard', '') if cam_path is not None
                    else (args.profile or args.filter or 'clean'))
            tag = base + ('+canon' if canon else '')
            print(f'[{tag}] PSNR={m["psnr"]:.4f} SSIM={m["ssim"]:.4f} '
                  f'SAM={m["sam"]:.4f} ERGAS={m["ergas"]:.4f}', flush=True)
            rows.append([tag, m['psnr'], m['ssim'], m['sam'], m['ergas']])
        with (out_dir / 'uafl_test.csv').open('a', newline='') as f:
            w = csv.writer(f)
            w.writerows(rows)
        return

    # ── train ──
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.SmoothL1Loss()
    best = 0.0
    start = args.ckpt is not None
    if start:
        ck = torch.load(ROOT / args.ckpt, map_location=device)
        model.load_state_dict(ck['model'])
        best = ck.get('best', 0.0)
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            gt, lq, rgb = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            x = upsample_lr(lq, gt.shape[-2:])
            out = model(x, rgb)
            loss = crit(out, gt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            m = evaluate(model, test_ds, args, device)
            print(f'epoch {epoch+1}/{args.epochs} loss={ep_loss/len(loader):.5f} '
                  f'val PSNR={m["psnr"]:.4f} SSIM={m["ssim"]:.4f}')
            if m['psnr'] > best:
                best = m['psnr']
                torch.save({'model': model.state_dict(), 'best': best,
                            'epoch': epoch + 1}, out_dir / 'best.pth.tar')
            model.train()
    torch.save({'model': model.state_dict(), 'best': best, 'epoch': args.epochs},
               out_dir / 'last.pth.tar')
    print(f'Done. Best val PSNR={best:.4f}')


if __name__ == '__main__':
    main()
