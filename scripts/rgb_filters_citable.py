"""Paper-citable RGB shift transforms for HSI-SR cross-camera evaluation.

Transforms are organised along the physical imaging pipeline (the order in
which a real RGB is formed), NOT by ad-hoc mathematical grouping. The full
pipeline has five stages:

    illumination -> SRF -> sensor_degradation -> correction -> post_filter

This script implements the three RGB-domain stages (illumination, correction,
post_filter). The other two are produced elsewhere:
  - SRF stage:                 28 camera SRFs (Jiang et al., ICCV 2013),
                               tested via the 'camera' category.
  - sensor_degradation stage:  Bayer + Poisson-Gaussian noise + JPEG
                               (scripts/isp_pipeline.py), 'isp' category.

Canonical domain = P_N_V2 SRF (which the EMR-Diff authors identify as the
Nikon D700 sensor), neutral illuminant, linear RGB, no post-processing.

Stages and citations (this script):
  - illumination (scene light):
      white_balance_tungsten / white_balance_fluorescent
      RGB white-balance gains induced by a non-neutral illuminant.
      gain_c = (SRF @ illuminant)[green] / (SRF @ illuminant)[c]
      Illuminant spectra: CIE 015:2018 (Illuminant A 2856K, FL2 4230K)
      via colour-science 0.4.7, aligned to 420-720nm @ 10nm (31-band Harvard
      grid). Camera SRF: P_N_V2.
      (For the composite / end-to-end condition the illuminant is applied in
       the spectral domain, HSI x L(lambda); the RGB-domain gain here is the
       single-factor white-balance-error diagnostic.)
  - correction (in-camera colour rendering: exposure/CCM/gamma/tone curve):
      exposure_underexpose / exposure_overexpose -- digital gain, following
        the unprocessing pipeline (Brooks et al., CVPR 2019).
      gamma_encode -- linear sensor RGB -> sRGB (IEC 61966-2-1).
  - post_filter (post-capture enhancement):
      color_saturation / contrast -- PIL ImageEnhance operators standardised in
        AutoAugment (Cubuk et al., CVPR 2019) / RandAugment (NeurIPS 2020).

All transforms here are monotone, information-preserving colour maps.
"""

import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════
# Illuminant layer -- white balance under non-neutral illuminants
# ═══════════════════════════════════════════════════════════════════════
# Gains computed from P_N_V2 SRF x CIE standard illuminants.
# Formula: w = SRF @ illuminant_spectrum; gain_c = w[green] / w[c] (green=1).
# Illuminant spectra from CIE 015:2018 via colour-science, interpolated to
# 400-700nm at 10nm steps (31 bands) to match Harvard HSI.

# Verified with colour-science 0.4.7, P_N_V2 SRF, CIE 015:2018 illuminants
# aligned to SpectralShape(420, 720, 10) -- the 31-band Harvard HSI grid.
# gain_c = w[green] / w[c], w = P_N_V2 @ illuminant_SD.

# CIE Standard Illuminant A (tungsten, 2856K):
#   Camera response w_A = [161.14, 103.76, 49.62]
#   WB gains (green-normalized): [0.6439, 1.0, 2.0911]
WB_GAINS_TUNGSTEN_A = (0.6439, 1.0000, 2.0911)

# CIE Standard Illuminant FL2 (cool white fluorescent, 4230K):
#   Camera response w_FL2 = [5.80, 12.92, 7.00]
#   WB gains (green-normalized): [2.2281, 1.0, 1.8462]
WB_GAINS_FLUORESCENT_FL2 = (2.2281, 1.0000, 1.8462)

# ═══════════════════════════════════════════════════════════════════════
# Exposure layer -- digital gain
# ═══════════════════════════════════════════════════════════════════════
# Brooks et al. (CVPR 2019) sample a digital gain to model exposure variation.
# We use fixed representative under/over-exposure factors within that regime.
EXPOSURE_UNDER = 0.6
EXPOSURE_OVER = 1.4

# ═══════════════════════════════════════════════════════════════════════
# Sensor encoding layer -- sRGB transfer function (IEC 61966-2-1)
# ═══════════════════════════════════════════════════════════════════════


def apply_srgb_encode(rgb):
    """Linear RGB -> sRGB encoding (IEC 61966-2-1)."""
    # Piecewise: x <= 0.0031308 -> 12.92*x, else -> 1.055*x^(1/2.4) - 0.055
    linear = rgb.clamp(0.0, 1.0)
    low_mask = linear <= 0.0031308
    out = torch.zeros_like(linear)
    out[low_mask] = 12.92 * linear[low_mask]
    out[~low_mask] = 1.055 * linear[~low_mask].pow(1.0 / 2.4) - 0.055
    return out.clamp(0.0, 1.0)


def apply_srgb_decode(rgb):
    """sRGB -> linear RGB decoding (IEC 61966-2-1)."""
    srgb = rgb.clamp(0.0, 1.0)
    low_mask = srgb <= 0.04045
    out = torch.zeros_like(srgb)
    out[low_mask] = srgb[low_mask] / 12.92
    out[~low_mask] = ((srgb[~low_mask] + 0.055) / 1.055).pow(2.4)
    return out.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Post-processing layer -- AutoAugment/RandAugment colour operators
# ═══════════════════════════════════════════════════════════════════════
# PIL ImageEnhance definition: out = blend(degenerate, image, factor)
#   blend(a, b, f) = a + f * (b - a); factor > 1 enhances, factor < 1 reduces.
# These are monotone, information-preserving colour maps.

# ITU-R BT.601 luma weights (PIL uses these for grayscale conversion)
_LUMA = (0.299, 0.587, 0.114)

# Enhancement factors (AutoAugment magnitude range is [0.1, 1.9] around 1.0)
COLOR_SATURATION_FACTOR = 1.8   # boost saturation ("vivid")
CONTRAST_FACTOR = 1.5           # boost contrast


def _luma(rgb):
    """Per-pixel grayscale luminance, shape (1, H, W)."""
    w = rgb.new_tensor(_LUMA).view(3, 1, 1)
    return (rgb * w).sum(dim=0, keepdim=True)


def apply_color_saturation(rgb, factor=COLOR_SATURATION_FACTOR):
    """PIL ImageEnhance.Color: blend grayscale with original (AutoAugment 'Color')."""
    gray = _luma(rgb).expand_as(rgb)
    out = gray + factor * (rgb - gray)
    return out.clamp(0.0, 1.0)


def apply_contrast(rgb, factor=CONTRAST_FACTOR):
    """PIL ImageEnhance.Contrast: blend mean-gray with original (AutoAugment 'Contrast')."""
    # PIL uses the mean of the grayscale image as the degenerate constant.
    mean = _luma(rgb).mean()
    out = mean + factor * (rgb - mean)
    return out.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

FILTERS = (
    'clean',
    'white_balance_tungsten',     # illuminant: CIE Illuminant A, 2856K
    'white_balance_fluorescent',  # illuminant: CIE FL2, 4230K
    'exposure_underexpose',       # exposure: 0.6x digital gain
    'exposure_overexpose',        # exposure: 1.4x digital gain
    'gamma_encode',               # sensor encoding: linear -> sRGB
    'color_saturation',           # post-processing: saturation boost
    'contrast',                   # post-processing: contrast boost
)

# Which imaging-pipeline stage each transform belongs to.
# Pipeline order: illumination -> SRF -> sensor_degradation -> correction -> post_filter
# (SRF and sensor_degradation stages are produced by the camera/isp categories.)
FILTER_LAYERS = {
    'clean': 'control',
    'white_balance_tungsten': 'illumination',
    'white_balance_fluorescent': 'illumination',
    'exposure_underexpose': 'correction',
    'exposure_overexpose': 'correction',
    'gamma_encode': 'correction',
    'color_saturation': 'post_filter',
    'contrast': 'post_filter',
}

FILTER_CITATIONS = {
    'clean': 'Control (no transformation)',
    'white_balance_tungsten': 'WB gains [0.644, 1.0, 2.091] for CIE Illuminant A (2856K). gain_c = (SRF@ill_A)[green]/(SRF@ill_A)[c] using P_N_V2 SRF, illuminant from CIE 015:2018 via colour-science 0.4.7 (420-720nm @ 10nm).',
    'white_balance_fluorescent': 'WB gains [2.228, 1.0, 1.846] for CIE FL2 (4230K). Same computation as tungsten.',
    'exposure_underexpose': 'Digital gain 0.6x, following the unprocessing pipeline (Brooks et al., CVPR 2019).',
    'exposure_overexpose': 'Digital gain 1.4x, following the unprocessing pipeline (Brooks et al., CVPR 2019).',
    'gamma_encode': 'Linear RGB -> sRGB encoding (IEC 61966-2-1). Standard camera output gamma compression.',
    'color_saturation': 'Saturation boost (factor 1.8), PIL ImageEnhance.Color as used in AutoAugment (Cubuk et al., CVPR 2019) / RandAugment (NeurIPS 2020).',
    'contrast': 'Contrast boost (factor 1.5), PIL ImageEnhance.Contrast as used in AutoAugment (Cubuk et al., CVPR 2019) / RandAugment (NeurIPS 2020).',
}


def apply_rgb_filter(rgb, filter_name, index=0):
    """Apply a single citable RGB transform.

    Args:
        rgb: torch.Tensor of shape (3, H, W), values in [0, 1]
        filter_name: one of FILTERS
        index: sample index (unused, kept for API compatibility)

    Returns:
        torch.Tensor of shape (3, H, W), values clamped to [0, 1]
    """
    if filter_name == 'clean':
        return rgb

    if filter_name == 'white_balance_tungsten':
        gains = rgb.new_tensor(WB_GAINS_TUNGSTEN_A).view(3, 1, 1)
        return (rgb * gains).clamp(0.0, 1.0)

    if filter_name == 'white_balance_fluorescent':
        gains = rgb.new_tensor(WB_GAINS_FLUORESCENT_FL2).view(3, 1, 1)
        return (rgb * gains).clamp(0.0, 1.0)

    if filter_name == 'exposure_underexpose':
        return (EXPOSURE_UNDER * rgb).clamp(0.0, 1.0)

    if filter_name == 'exposure_overexpose':
        return (EXPOSURE_OVER * rgb).clamp(0.0, 1.0)

    if filter_name == 'gamma_encode':
        return apply_srgb_encode(rgb)

    if filter_name == 'gamma_decode':
        return apply_srgb_decode(rgb)

    if filter_name == 'color_saturation':
        return apply_color_saturation(rgb)

    if filter_name == 'contrast':
        return apply_contrast(rgb)

    raise ValueError(f'Unknown RGB filter: {filter_name}')
