"""Full-chain composite RGB synthesis (end-to-end imaging pipeline).

Unlike the single-factor conditions (each opens ONE pipeline layer), a
composite profile walks the whole imaging chain at once, producing a realistic
"as-captured" guidance RGB. Four monotone colour-mapping layers:

    HSI reflectance R(lambda)
      (1) x illuminant SED L(lambda)        [SPECTRAL domain]  -- Illumination
      (2) @ camera SRF -> linear RGB                            -- Sensor (SRF)
      (4) exposure gain -> sRGB gamma                           -- Correction (ISP)
      (5) saturation / contrast enhancement                     -- Post-filter
      -> final RGB

The non-monotone sensor-degradation layer (Bayer/noise/JPEG) is out of scope
for the canonicalizer and is excluded from the benchmark.

Physical correctness: the illuminant is applied in the SPECTRAL domain
(HSI x L(lambda)), not as an RGB white-balance gain -- this is stricter than
the single-factor illuminant diagnostic in rgb_filters_citable.py.

Citations per layer: see docs/AAAI_submission_experiment_inventory.md 0.1b.
  illuminant SED: CIE 015:2018 (via colour-science)
  SRF:            Jiang et al., ICCV 2013
  correction:     Brooks et al., CVPR 2019 (gain) / IEC 61966-2-1 (sRGB)
  post-filter:    Cubuk et al., CVPR 2019 (AutoAugment)
"""

from pathlib import Path

import colour
import numpy as np
import scipy.io as sio
import torch
from colour import SpectralShape

from scripts.rgb_filters_citable import (
    apply_color_saturation,
    apply_contrast,
    apply_srgb_encode,
)

ROOT = Path(__file__).resolve().parents[1]

# Harvard HSI spectral grid: 420-720 nm at 10 nm (31 bands).
SPECTRAL_SHAPE = SpectralShape(420, 720, 10)


def load_illuminant(name):
    """CIE standard illuminant SED aligned to the 31-band Harvard grid.

    Returns a (31,) tensor, normalised to mean 1 so it acts as a relative
    spectral reweighting of reflectance (preserves overall scale).
    """
    sd = colour.SDS_ILLUMINANTS[name].copy().align(SPECTRAL_SHAPE)
    v = torch.from_numpy(np.asarray(sd.values, dtype=np.float32))
    return v / v.mean().clamp_min(1e-8)


def load_srf(name):
    """Camera SRF (3, 31), row-normalised (rows sum to 1) as in the dataloader."""
    path = ROOT / 'SRF' / 'test' / f'{name}_harvard.mat'
    srf = torch.from_numpy(sio.loadmat(str(path))['P_20N']).float()
    if tuple(srf.shape) != (3, 31):
        raise ValueError(f'{name}: expected SRF (3,31), got {tuple(srf.shape)}')
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)


# ═══════════════════════════════════════════════════════════════════════
# Composite profiles (see doc 2.6)
# ═══════════════════════════════════════════════════════════════════════
# Each profile is a dict describing the imaging chain. None/absent = skip layer.
# Four monotone colour-mapping layers only: illumination / SRF / correction /
# post-filter. The non-monotone sensor-degradation layer (Bayer/noise/JPEG) is
# out of scope for the canonicalizer and excluded from the benchmark.
PROFILES = {
    'A_indoor': {
        'illuminant': 'A',                 # tungsten
        'srf': 'nokia_n900',               # consumer camera
        'exposure': 0.6,                   # under-expose
        'gamma': True,                     # sRGB encode
        'saturation': 1.4,                 # mild
        'contrast': None,
    },
    'B_outdoor': {
        'illuminant': 'D65',               # daylight
        'srf': 'canon_5dmarkii',           # DSLR
        'exposure': None,
        'gamma': True,
        'saturation': 1.8,                 # vivid
        'contrast': 1.5,
    },
    # Profile C: worst single-factor config per layer (full-20 single-factor
    # results), across the four colour-mapping layers. Maximal stress.
    #   (1) illuminant  -> FL2         (worst: -28.98 dB)
    #   (2) SRF         -> nikon_d40   (worst camera: -8.69 dB)
    #   (4) correction  -> gamma + over-expose 1.4 (over > under: -17.71 dB)
    #   (5) post-filter -> contrast 1.5 (contrast > saturation: -17.73 dB)
    'C_stress': {
        'illuminant': 'FL2',
        'srf': 'nikon_d40',
        'exposure': 1.4,
        'gamma': True,
        'saturation': None,
        'contrast': 1.5,
    },
}


class CompositeProfile:
    """Callable that maps a (C,H,W) HSI reflectance tensor to a composite RGB."""

    def __init__(self, spec, seed=20260723):
        self.spec = spec
        self.seed = seed
        self.illuminant = (
            load_illuminant(spec['illuminant']) if spec.get('illuminant') else None
        )
        self.srf = load_srf(spec['srf'])

    def __call__(self, hsi):
        # hsi: (C, H, W) reflectance in [0, 1]
        x = hsi
        # (1) Illumination -- spectral-domain reweighting
        if self.illuminant is not None:
            x = x * self.illuminant.to(x.device, x.dtype).view(-1, 1, 1)
        # (2) Sensor SRF -> linear RGB
        c, h, w = x.shape
        rgb = torch.matmul(
            self.srf.to(x.device, x.dtype), x.reshape(c, -1)
        ).reshape(3, h, w).clamp(0.0, 1.0)
        # (Sensor degradation layer intentionally omitted: non-monotone,
        #  out of scope for the canonicalizer.)
        # (4) Correction: exposure gain then sRGB gamma
        if self.spec.get('exposure'):
            rgb = (rgb * float(self.spec['exposure'])).clamp(0.0, 1.0)
        if self.spec.get('gamma'):
            rgb = apply_srgb_encode(rgb)
        # (5) Post-filter: saturation then contrast
        if self.spec.get('saturation'):
            rgb = apply_color_saturation(rgb, factor=float(self.spec['saturation']))
        if self.spec.get('contrast'):
            rgb = apply_contrast(rgb, factor=float(self.spec['contrast']))
        return rgb.clamp(0.0, 1.0)


def build_profile(name, seed=20260723):
    if name not in PROFILES:
        raise ValueError(f'Unknown profile {name}. Available: {list(PROFILES)}')
    return CompositeProfile(PROFILES[name], seed=seed)
