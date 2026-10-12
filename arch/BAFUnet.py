from abc import abstractmethod
import math
from CBMA import *
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from arch.swin_transformer import Mlp
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .basic_ops import (
    linear,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

try:
    import xformers
    import xformers.ops as xop
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
size = 3
padding = 1
num = 32
band = 34
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, size, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)
class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, size, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class EdgeControlBlock(nn.Module):
    def __init__(self, out_channels, hidden_channels=64, zero_init=True, scale_init=1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            LayerNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            LayerNorm2d(hidden_channels),
        )
        self.zero_conv = nn.Conv2d(hidden_channels, out_channels, 1)
        if zero_init:
            nn.init.zeros_(self.zero_conv.weight)
            nn.init.zeros_(self.zero_conv.bias)
        self.scale = nn.Parameter(th.tensor(float(scale_init), dtype=th.float32))

    def forward(self, edge, size_hw, dtype):
        if edge.shape[-2:] != size_hw:
            edge = F.interpolate(edge, size=size_hw, mode='bilinear', align_corners=False)
        residual = self.zero_conv(self.body(edge.to(dtype=dtype)))
        return self.scale.to(dtype=dtype) * residual

class RGBMultiScaleControlAdapter(nn.Module):
    def __init__(
            self,
            layer_ids,
            hidden_channels=64,
            rgb_mode='raw_plus_grad',
            adapter_type='simple',
            num_msgab_blocks=2,
            max_pyramid_levels=8,
    ):
        super().__init__()
        if rgb_mode not in ('raw', 'grad_mag', 'raw_plus_grad', 'fft_high', 'raw_plus_fft_high'):
            raise ValueError(
                "control_adapter_rgb_mode must be one of: 'raw', 'grad_mag', 'raw_plus_grad', "
                "'fft_high', 'raw_plus_fft_high'."
            )
        if adapter_type not in ('simple', 'msgab_pyramid'):
            raise ValueError("control_adapter_type must be one of: 'simple', 'msgab_pyramid'.")
        self.layer_ids = sorted([str(i) for i in layer_ids], key=lambda x: 10 ** 6 if x == 'mid' else int(x))
        self.rgb_mode = rgb_mode
        self.adapter_type = adapter_type
        self.hidden_channels = hidden_channels
        self.max_pyramid_levels = max_pyramid_levels
        in_channels = {
            'raw': 3,
            'grad_mag': 1,
            'raw_plus_grad': 4,
            'fft_high': 3,
            'raw_plus_fft_high': 6,
        }[rgb_mode]
        gray_weight = th.tensor([0.299, 0.587, 0.114], dtype=th.float32).view(1, 3, 1, 1)
        sobel_x = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32).view(1, 1, 3, 3)
        sobel_y = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32).view(1, 1, 3, 3)
        self.register_buffer('gray_weight', gray_weight)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        if adapter_type == 'simple':
            self.blocks = nn.ModuleDict({
                layer_id: self._make_simple_block(in_channels, hidden_channels)
                for layer_id in self.layer_ids
            })
        else:
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                LayerNorm2d(hidden_channels),
            )
            self.pyramid_blocks = nn.ModuleList([
                self._make_msgab_stage(hidden_channels, num_msgab_blocks)
                for _ in range(max_pyramid_levels)
            ])
            self.pyramid_downs = nn.ModuleList([
                Downsample(hidden_channels, use_conv=True, dims=2, out_channels=hidden_channels)
                for _ in range(max_pyramid_levels - 1)
            ])

    def _make_simple_block(self, in_channels, hidden_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            LayerNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            LayerNorm2d(hidden_channels),
        )

    def _make_msgab_stage(self, hidden_channels, num_msgab_blocks):
        layers = []
        for _ in range(num_msgab_blocks):
            layers.append(MSGAB(
                hidden_channels,
                dropout=0,
                out_channels=hidden_channels,
                groups_list=[1, hidden_channels],
                norm_groups=hidden_channels,
            ))
        return nn.Sequential(*layers)

    def _grad_mag(self, rgb):
        rgb = rgb.float()
        gray = F.conv2d(rgb, self.gray_weight.to(dtype=rgb.dtype))
        gx = F.conv2d(gray, self.sobel_x.to(dtype=rgb.dtype), padding=1)
        gy = F.conv2d(gray, self.sobel_y.to(dtype=rgb.dtype), padding=1)
        grad_mag = th.sqrt(gx * gx + gy * gy + 1e-8)
        mean = grad_mag.mean(dim=(2, 3), keepdim=True)
        std = grad_mag.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (grad_mag - mean) / std

    def _fft_high(self, rgb, cutoff_ratio=0.12):
        rgb = rgb.float()
        _, _, h, w = rgb.shape
        freq = th.fft.fftshift(th.fft.fft2(rgb, norm='ortho'), dim=(-2, -1))
        yy = th.arange(h, device=rgb.device, dtype=rgb.dtype).view(h, 1) - (h - 1) / 2.0
        xx = th.arange(w, device=rgb.device, dtype=rgb.dtype).view(1, w) - (w - 1) / 2.0
        radius = th.sqrt(xx * xx + yy * yy)
        cutoff = cutoff_ratio * min(h, w)
        high_mask = (radius >= cutoff).to(freq.dtype).view(1, 1, h, w)
        high = th.fft.ifft2(th.fft.ifftshift(freq * high_mask, dim=(-2, -1)), norm='ortho').real.abs()
        mean = high.mean(dim=(2, 3), keepdim=True)
        std = high.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (high - mean) / std

    def _rgb_input(self, rgb):
        if self.rgb_mode == 'raw':
            return rgb.float()
        if self.rgb_mode == 'fft_high':
            return self._fft_high(rgb)
        if self.rgb_mode == 'raw_plus_fft_high':
            return th.cat([rgb.float(), self._fft_high(rgb)], dim=1)
        grad_mag = self._grad_mag(rgb)
        if self.rgb_mode == 'grad_mag':
            return grad_mag
        return th.cat([rgb.float(), grad_mag], dim=1)

    def forward(self, rgb, target_sizes):
        rgb_in = self._rgb_input(rgb)
        feats = {}
        if self.adapter_type == 'simple':
            for layer_id, size_hw in target_sizes.items():
                cond = rgb_in
                if cond.shape[-2:] != size_hw:
                    cond = F.interpolate(cond, size=size_hw, mode='bilinear', align_corners=False)
                feats[layer_id] = self.blocks[layer_id](cond)
            return feats

        if not target_sizes:
            return feats

        base_size = max(target_sizes.values(), key=lambda hw: hw[0] * hw[1])
        cond = rgb_in
        if cond.shape[-2:] != base_size:
            cond = F.interpolate(cond, size=base_size, mode='bilinear', align_corners=False)

        pyramid = {}
        h = self.input_proj(cond)
        for level, block in enumerate(self.pyramid_blocks):
            h = block(h)
            pyramid[h.shape[-2:]] = h
            if level == len(self.pyramid_blocks) - 1:
                break
            min_target_h = min(size_hw[0] for size_hw in target_sizes.values())
            min_target_w = min(size_hw[1] for size_hw in target_sizes.values())
            if h.shape[-2] <= min_target_h and h.shape[-1] <= min_target_w:
                break
            h = self.pyramid_downs[level](h)

        for layer_id, size_hw in target_sizes.items():
            feat = pyramid.get(size_hw)
            if feat is None:
                feat = min(
                    pyramid.values(),
                    key=lambda x: abs(x.shape[-2] - size_hw[0]) + abs(x.shape[-1] - size_hw[1]),
                )
                feat = F.interpolate(feat, size=size_hw, mode='bilinear', align_corners=False)
            feats[layer_id] = feat
        return feats


class ControlResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, zero_init=True, scale_init=1.0):
        super().__init__()
        self.zero_conv = nn.Conv2d(in_channels, out_channels, 1)
        if zero_init:
            nn.init.zeros_(self.zero_conv.weight)
            nn.init.zeros_(self.zero_conv.bias)
        self.scale = nn.Parameter(th.tensor(float(scale_init), dtype=th.float32))

    def forward(self, control_feat, size_hw, dtype):
        if control_feat.shape[-2:] != size_hw:
            control_feat = F.interpolate(control_feat, size=size_hw, mode='bilinear', align_corners=False)
        residual = self.zero_conv(control_feat.to(dtype=dtype))
        return self.scale.to(dtype=dtype) * residual


class LQAwareGatedControlResidualBlock(nn.Module):
    def __init__(
            self,
            control_channels,
            lq_channels,
            out_channels,
            hidden_channels=64,
            gate_channels=1,
            gate_init_bias=-3.0,
            zero_init=True,
            scale_init=1.0,
    ):
        super().__init__()
        if gate_channels not in (1, out_channels):
            raise ValueError("control_gate_channels must be 1 or match the residual out_channels.")
        fuse_channels = control_channels + lq_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(fuse_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            LayerNorm2d(hidden_channels),
        )
        self.body = MSGAB(
            hidden_channels,
            dropout=0,
            out_channels=hidden_channels,
            groups_list=[1, hidden_channels],
            norm_groups=hidden_channels,
        )
        self.detail = nn.Conv2d(hidden_channels, out_channels, 1)
        self.gate = nn.Conv2d(hidden_channels, gate_channels, 1)
        if zero_init:
            nn.init.zeros_(self.detail.weight)
            nn.init.zeros_(self.detail.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self.scale = nn.Parameter(th.tensor(float(scale_init), dtype=th.float32))

    def forward(self, control_feat, lq_feat, size_hw, dtype):
        if control_feat.shape[-2:] != size_hw:
            control_feat = F.interpolate(control_feat, size=size_hw, mode='bilinear', align_corners=False)
        if lq_feat.shape[-2:] != size_hw:
            lq_feat = F.interpolate(lq_feat, size=size_hw, mode='bilinear', align_corners=False)
        feat = th.cat([control_feat.to(dtype=dtype), lq_feat.to(dtype=dtype)], dim=1)
        feat = self.input_proj(feat)
        feat = self.body(feat)
        detail = self.detail(feat)
        gate = th.sigmoid(self.gate(feat))
        return self.scale.to(dtype=dtype) * gate * detail



class FactorizedRGBDetailBranch(nn.Module):
    """RGB supplies spatial maps while LR-HSI supplies per-band coefficients."""

    def __init__(
            self,
            rgb_channels,
            hsi_channels,
            out_channels,
            spatial_channels=8,
            hidden_channels=32,
            coefficient_scale=4,
            local_norm_kernel=5,
            gate_init_bias=-2.0,
            residual_scale_init=0.1,
            use_highpass=False,
            highpass_kernel=5,
    ):
        super().__init__()
        for name, value in (
                ('local_norm_kernel', local_norm_kernel),
                ('highpass_kernel', highpass_kernel),
        ):
            if int(value) < 3 or int(value) % 2 == 0:
                raise ValueError(f'{name} must be an odd integer >= 3.')
        if int(coefficient_scale) < 1:
            raise ValueError('factorized_coefficient_scale must be >= 1.')

        self.rgb_channels = int(rgb_channels)
        self.hsi_channels = int(hsi_channels)
        self.out_channels = int(out_channels)
        self.spatial_channels = int(spatial_channels)
        self.coefficient_scale = int(coefficient_scale)
        self.local_norm_kernel = int(local_norm_kernel)
        self.use_highpass = bool(use_highpass)
        self.highpass_kernel = int(highpass_kernel)

        self.shared_rgb_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(1, hidden_channels),
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, self.spatial_channels, 1),
        )

        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(self.hsi_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(1, hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(1, hidden_channels),
        )
        self.coefficient_head = nn.Conv2d(
            hidden_channels, self.out_channels * self.spatial_channels, 1
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels + self.spatial_channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_init_bias))
        self.residual_scale = nn.Parameter(
            th.tensor(float(residual_scale_init), dtype=th.float32)
        )

    def _local_contrast_normalize(self, rgb):
        batch, channels, height, width = rgb.shape
        flat = rgb.float().reshape(batch * channels, 1, height, width)
        padding = self.local_norm_kernel // 2
        padded = F.pad(flat, (padding, padding, padding, padding), mode='reflect')
        mean = F.avg_pool2d(padded, self.local_norm_kernel, stride=1)
        mean_square = F.avg_pool2d(padded * padded, self.local_norm_kernel, stride=1)
        std = (mean_square - mean * mean).clamp_min(1e-6).sqrt()
        return ((flat - mean) / std).reshape(batch, channels, height, width)

    def _highpass(self, detail):
        padding = self.highpass_kernel // 2
        padded = F.pad(detail, (padding, padding, padding, padding), mode='reflect')
        low = F.avg_pool2d(padded, self.highpass_kernel, stride=1)
        return detail - low

    def forward(self, rgb, lq):
        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f'Expected {self.rgb_channels} RGB channels, got {rgb.shape[1]}.'
            )
        if lq.shape[1] != self.hsi_channels:
            raise ValueError(
                f'Expected {self.hsi_channels} LR-HSI channels, got {lq.shape[1]}.'
            )

        batch, channels, height, width = rgb.shape
        rgb_norm = self._local_contrast_normalize(rgb)
        shared = self.shared_rgb_encoder(
            rgb_norm.reshape(batch * channels, 1, height, width)
        )
        shared = shared.reshape(batch, channels, shared.shape[1], height, width)
        shared_mean = shared.mean(dim=1)
        shared_max = shared.max(dim=1).values
        spatial = self.spatial_head(th.cat([shared_mean, shared_max], dim=1))

        spectral_input = lq.float()
        if self.coefficient_scale > 1:
            spectral_input = F.avg_pool2d(
                spectral_input,
                kernel_size=self.coefficient_scale,
                stride=self.coefficient_scale,
            )
        spectral_feature = self.spectral_encoder(spectral_input)
        coefficient = self.coefficient_head(spectral_feature)
        if coefficient.shape[-2:] != (height, width):
            coefficient = F.interpolate(
                coefficient, size=(height, width), mode='bilinear', align_corners=False
            )
            spectral_feature = F.interpolate(
                spectral_feature, size=(height, width), mode='bilinear', align_corners=False
            )

        coefficient = coefficient.reshape(
            batch, self.out_channels, self.spatial_channels, height, width
        )
        coefficient = th.tanh(coefficient) / math.sqrt(self.spatial_channels)
        detail = th.einsum('bckhw,bkhw->bchw', coefficient, spatial)

        gate = th.sigmoid(self.gate_head(th.cat([spectral_feature, spatial], dim=1)))
        detail = self.residual_scale.to(dtype=detail.dtype) * gate * detail
        if self.use_highpass:
            detail = self._highpass(detail)
        return detail, spatial, gate



class CanonicalResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class NikonRGBRestorer(nn.Module):
    """Restore arbitrary HR RGB into a fixed Nikon domain using LR-HSI."""

    def __init__(
            self,
            rgb_channels=3,
            hsi_channels=31,
            scale=8,
            hidden_channels=96,
            num_blocks=4,
    ):
        super().__init__()
        if scale < 1:
            raise ValueError('nikon_restorer_scale must be >= 1.')
        if hidden_channels % 8 != 0:
            raise ValueError('nikon_restorer_hidden_channels must be divisible by 8.')
        self.rgb_channels = int(rgb_channels)
        self.hsi_channels = int(hsi_channels)
        self.scale = int(scale)
        packed_channels = self.rgb_channels * self.scale * self.scale
        self.register_buffer(
            'target_srf', th.zeros(self.rgb_channels, self.hsi_channels),
            persistent=True,
        )

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(packed_channels, hidden_channels, 3, padding=1),
            CanonicalResidualBlock(hidden_channels),
            CanonicalResidualBlock(hidden_channels),
        )
        self.spectral_encoder = nn.Sequential(
            nn.Conv2d(self.hsi_channels + self.rgb_channels, hidden_channels, 3, padding=1),
            CanonicalResidualBlock(hidden_channels),
            CanonicalResidualBlock(hidden_channels),
        )
        self.fusion_stem = nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1)
        self.fusion_blocks = nn.Sequential(*[
            MSGAB(
                hidden_channels,
                dropout=0,
                out_channels=hidden_channels,
                groups_list=[1, hidden_channels],
                norm_groups=8,
            )
            for _ in range(int(num_blocks))
        ])
        self.residual_head = nn.Conv2d(hidden_channels, packed_channels, 3, padding=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def set_target_srf(self, srf):
        srf = th.as_tensor(srf, dtype=self.target_srf.dtype)
        if tuple(srf.shape) != tuple(self.target_srf.shape):
            raise ValueError(
                f'Expected target SRF shape {tuple(self.target_srf.shape)}, got {tuple(srf.shape)}.'
            )
        if not th.isfinite(srf).all():
            raise ValueError('Target SRF contains non-finite values.')
        if (srf < 0).any():
            raise ValueError('Target SRF must be non-negative.')
        row_sums = srf.sum(dim=1, keepdim=True)
        if (row_sums <= 0).any():
            raise ValueError('Every target SRF row must have a positive sum.')
        self.target_srf.copy_(srf / row_sums)

    def project_hsi(self, hsi):
        if hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f'Expected {self.hsi_channels} HSI channels, got {hsi.shape[1]}.'
            )
        return th.einsum(
            'kc,bchw->bkhw', self.target_srf.to(dtype=hsi.dtype), hsi
        )

    def forward(self, rgb, lq_hsi):
        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f'Expected {self.rgb_channels} RGB channels, got {rgb.shape[1]}.'
            )
        expected_size = (
            lq_hsi.shape[-2] * self.scale,
            lq_hsi.shape[-1] * self.scale,
        )
        if rgb.shape[-2:] != expected_size:
            raise ValueError(
                f'RGB size {tuple(rgb.shape[-2:])} must equal LR-HSI size '
                f'{tuple(lq_hsi.shape[-2:])} multiplied by scale {self.scale}.'
            )

        nikon_lr = self.project_hsi(lq_hsi.float())
        nikon_base = F.interpolate(
            nikon_lr,
            size=rgb.shape[-2:],
            mode='bicubic',
            align_corners=False,
        )
        spatial = self.spatial_encoder(F.pixel_unshuffle(rgb.float(), self.scale))
        spectral = self.spectral_encoder(th.cat([lq_hsi.float(), nikon_lr], dim=1))
        fused = self.fusion_stem(th.cat([spatial, spectral], dim=1))
        fused = self.fusion_blocks(fused)
        residual = F.pixel_shuffle(self.residual_head(fused), self.scale)
        restored = (nikon_base + residual).clamp(0.0, 1.0)
        return restored, nikon_lr, residual


class FixedSRFRGBCanonicalizer(nn.Module):
    """Canonicalize arbitrary HR RGB with LR-HSI and a fixed target SRF."""

    def __init__(
            self,
            rgb_channels=3,
            hsi_channels=31,
            scale=8,
            mode='local_affine',
            global_ridge=1e-2,
            polynomial_ridge=1e-4,
            polynomial_degree=3,
            rbf_centers=64,
            rbf_lambda=1e-3,
            rbf_min_bandwidth=2e-2,
            local_ridge=1e-6,
            local_window=11,
            identity_threshold=0.0,
            backprojection_steps=1,
            normalize_target_srf=False,
            allow_signed_target_srf=True,
            clamp_output=False,
            degradation_sigma=2.0,
            degradation_kernel_size=7,
    ):
        super().__init__()
        if mode not in (
                'global_affine', 'global_polynomial',
                'global_forward_polynomial', 'global_full_polynomial',
                'global_forward_full_polynomial', 'global_rbf', 'local_affine',
        ):
            raise ValueError(
                "nikon_restorer_mode must be 'global_affine', "
                "'global_polynomial', 'global_forward_polynomial', "
                "'global_full_polynomial', 'global_forward_full_polynomial', "
                "'global_rbf', or 'local_affine'."
            )
        if scale < 1:
            raise ValueError('nikon_restorer_scale must be >= 1.')
        if local_window < 3 or local_window % 2 == 0:
            raise ValueError('nikon_restorer_local_window must be odd and >= 3.')
        if degradation_kernel_size < 3 or degradation_kernel_size % 2 == 0:
            raise ValueError(
                'nikon_restorer_degradation_kernel_size must be odd and >= 3.'
            )
        if global_ridge < 0 or polynomial_ridge < 0 or local_ridge < 0:
            raise ValueError('Canonicalization ridge values must be non-negative.')
        if backprojection_steps < 0:
            raise ValueError('nikon_restorer_backprojection_steps must be >= 0.')
        if identity_threshold < 0:
            raise ValueError('nikon_restorer_identity_threshold must be >= 0.')
        if int(polynomial_degree) < 1:
            raise ValueError('nikon_restorer_polynomial_degree must be >= 1.')
        if int(rbf_centers) < 1:
            raise ValueError('nikon_restorer_rbf_centers must be >= 1.')
        if rbf_lambda < 0 or rbf_min_bandwidth <= 0:
            raise ValueError('RBF regularization must be non-negative and bandwidth positive.')

        self.rgb_channels = int(rgb_channels)
        self.hsi_channels = int(hsi_channels)
        self.scale = int(scale)
        self.mode = str(mode)
        self.global_ridge = float(global_ridge)
        self.polynomial_ridge = float(polynomial_ridge)
        self.polynomial_degree = int(polynomial_degree)
        self.rbf_centers = int(rbf_centers)
        self.rbf_lambda = float(rbf_lambda)
        self.rbf_min_bandwidth = float(rbf_min_bandwidth)
        self.local_ridge = float(local_ridge)
        self.local_window = int(local_window)
        self.identity_threshold = float(identity_threshold)
        self.backprojection_steps = int(backprojection_steps)
        self.normalize_target_srf = bool(normalize_target_srf)
        self.allow_signed_target_srf = bool(allow_signed_target_srf)
        self.clamp_output = bool(clamp_output)
        self.register_buffer(
            'target_srf', th.zeros(self.rgb_channels, self.hsi_channels),
            persistent=True,
        )

        axis = th.arange(degradation_kernel_size, dtype=th.float32)
        axis = axis - degradation_kernel_size // 2
        yy, xx = th.meshgrid(axis, axis, indexing='ij')
        kernel = th.exp(-(xx.square() + yy.square()) / (2 * degradation_sigma ** 2))
        kernel = (kernel / kernel.sum()).reshape(
            1, 1, degradation_kernel_size, degradation_kernel_size
        )
        self.register_buffer('degradation_kernel', kernel, persistent=False)

    @staticmethod
    def _complete_polynomial_exponents(degree):
        exponents = []
        for total_degree in range(1, int(degree) + 1):
            for red_power in range(total_degree, -1, -1):
                remaining_degree = total_degree - red_power
                for green_power in range(remaining_degree, -1, -1):
                    blue_power = remaining_degree - green_power
                    exponents.append((red_power, green_power, blue_power))
        return tuple(exponents)

    @property
    def full_polynomial_term_count(self):
        return len(self._complete_polynomial_exponents(self.polynomial_degree)) + 1

    def set_polynomial_degree(self, degree):
        if int(degree) < 1:
            raise ValueError('Polynomial degree must be >= 1.')
        self.polynomial_degree = int(degree)

    def set_target_srf(self, srf):
        srf = th.as_tensor(srf, dtype=self.target_srf.dtype)
        if tuple(srf.shape) != tuple(self.target_srf.shape):
            raise ValueError(
                f'Expected target SRF shape {tuple(self.target_srf.shape)}, '
                f'got {tuple(srf.shape)}.'
            )
        if not th.isfinite(srf).all():
            raise ValueError('Target SRF contains non-finite values.')
        if not self.allow_signed_target_srf and (srf < 0).any():
            raise ValueError('Target SRF must be non-negative.')
        if self.normalize_target_srf:
            row_sums = srf.sum(dim=1, keepdim=True)
            if (row_sums.abs() <= 1e-8).any():
                raise ValueError('Every target SRF row must have a non-zero sum.')
            srf = srf / row_sums
        self.target_srf.copy_(srf)

    def project_hsi(self, hsi):
        if hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f'Expected {self.hsi_channels} HSI channels, got {hsi.shape[1]}.'
            )
        return th.einsum(
            'kc,bchw->bkhw', self.target_srf.to(dtype=hsi.dtype), hsi
        )

    def _degrade_rgb(self, rgb, output_size):
        channels = rgb.shape[1]
        kernel = self.degradation_kernel.to(device=rgb.device, dtype=rgb.dtype)
        padding = kernel.shape[-1] // 2
        blurred = F.conv2d(
            F.pad(rgb, (padding,) * 4, mode='reflect'),
            kernel.expand(channels, -1, -1, -1),
            groups=channels,
        )
        return F.interpolate(
            blurred, size=output_size, mode='bicubic', align_corners=False
        )

    @staticmethod
    def _ridge_eye(batch, size, device, dtype):
        eye = th.eye(size, device=device, dtype=dtype)
        eye = eye.expand(batch, -1, -1).clone()
        eye[:, -1, -1] = 0
        return eye

    def _global_affine(self, rgb, source_lr, target_lr):
        batch = rgb.shape[0]
        source = source_lr.flatten(2).transpose(1, 2)
        target = target_lr.flatten(2).transpose(1, 2)
        ones = th.ones(batch, source.shape[1], 1, device=rgb.device, dtype=rgb.dtype)
        design = th.cat([source, ones], dim=-1)
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ target
        eye = self._ridge_eye(batch, 4, rgb.device, rgb.dtype)
        coefficients = th.linalg.solve(gram + self.global_ridge * eye, rhs)

        source_hr = rgb.flatten(2).transpose(1, 2)
        hr_ones = th.ones(
            batch, source_hr.shape[1], 1, device=rgb.device, dtype=rgb.dtype
        )
        restored = th.cat([source_hr, hr_ones], dim=-1) @ coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    @staticmethod
    def _polynomial_design(source):
        red, green, blue = source.unbind(dim=-1)
        cross_terms = th.stack(
            [red * green, red * blue, green * blue], dim=-1
        )
        ones = th.ones_like(source[..., :1])
        return th.cat(
            [source, source.square(), source.pow(3), cross_terms, ones], dim=-1
        )

    def _global_polynomial(self, rgb, source_lr, target_lr):
        batch = rgb.shape[0]
        source = source_lr.flatten(2).transpose(1, 2)
        target = target_lr.flatten(2).transpose(1, 2)
        design = self._polynomial_design(source)
        gram = design.transpose(1, 2) @ design
        rhs = design.transpose(1, 2) @ target
        eye = self._ridge_eye(batch, design.shape[-1], rgb.device, rgb.dtype)
        coefficients = th.linalg.solve(
            gram + self.polynomial_ridge * eye, rhs
        )

        source_hr = rgb.flatten(2).transpose(1, 2)
        restored = self._polynomial_design(source_hr) @ coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    def _global_forward_polynomial(self, rgb, target_lr):
        batch = rgb.shape[0]
        source_hr = rgb.flatten(2).transpose(1, 2)
        design_hr = self._polynomial_design(source_hr)
        design_maps = design_hr.transpose(1, 2).reshape(
            batch, design_hr.shape[-1], *rgb.shape[-2:]
        )
        design_lr_maps = self._degrade_rgb(
            design_maps, target_lr.shape[-2:]
        )
        design_lr = design_lr_maps.flatten(2).transpose(1, 2)
        target = target_lr.flatten(2).transpose(1, 2)
        gram = design_lr.transpose(1, 2) @ design_lr
        rhs = design_lr.transpose(1, 2) @ target
        eye = self._ridge_eye(
            batch, design_lr.shape[-1], rgb.device, rgb.dtype
        )
        coefficients = th.linalg.solve(
            gram + self.polynomial_ridge * eye, rhs
        )
        restored = design_hr @ coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    def _full_polynomial_design(self, source):
        red, green, blue = source.unbind(dim=-1)
        terms = []
        for red_power, green_power, blue_power in (
                self._complete_polynomial_exponents(self.polynomial_degree)
        ):
            term = th.ones_like(red)
            if red_power:
                term = term * red.pow(red_power)
            if green_power:
                term = term * green.pow(green_power)
            if blue_power:
                term = term * blue.pow(blue_power)
            terms.append(term.unsqueeze(-1))
        terms.append(th.ones_like(source[..., :1]))
        return th.cat(terms, dim=-1)

    def _global_full_polynomial(self, rgb, source_lr, target_lr):
        batch = rgb.shape[0]
        source = source_lr.flatten(2).transpose(1, 2)
        target = target_lr.flatten(2).transpose(1, 2)
        design = self._full_polynomial_design(source)
        feature_scale = design.square().mean(dim=1, keepdim=True).sqrt()
        feature_scale = feature_scale.clamp_min(1e-6)
        feature_scale[..., -1] = 1.0
        normalized_design = design / feature_scale

        solve_design = normalized_design.double()
        solve_target = target.double()
        gram = solve_design.transpose(1, 2) @ solve_design
        rhs = solve_design.transpose(1, 2) @ solve_target
        eye = self._ridge_eye(
            batch, design.shape[-1], rgb.device, solve_design.dtype
        )
        coefficients = th.linalg.solve(
            gram + self.polynomial_ridge * eye, rhs
        ).to(dtype=rgb.dtype)

        source_hr = rgb.flatten(2).transpose(1, 2)
        design_hr = self._full_polynomial_design(source_hr)
        restored = (design_hr / feature_scale) @ coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    def _global_forward_full_polynomial(self, rgb, target_lr):
        batch = rgb.shape[0]
        source_hr = rgb.flatten(2).transpose(1, 2)
        design_hr = self._full_polynomial_design(source_hr)
        design_maps = design_hr.transpose(1, 2).reshape(
            batch, design_hr.shape[-1], *rgb.shape[-2:]
        )
        design_lr_maps = self._degrade_rgb(
            design_maps, target_lr.shape[-2:]
        )
        design_lr = design_lr_maps.flatten(2).transpose(1, 2)
        feature_scale = design_lr.square().mean(dim=1, keepdim=True).sqrt()
        feature_scale = feature_scale.clamp_min(1e-6)
        feature_scale[..., -1] = 1.0
        normalized_design = design_lr / feature_scale
        target = target_lr.flatten(2).transpose(1, 2)
        solve_design = normalized_design.double()
        solve_target = target.double()
        gram = solve_design.transpose(1, 2) @ solve_design
        rhs = solve_design.transpose(1, 2) @ solve_target
        eye = self._ridge_eye(
            batch, design_lr.shape[-1], rgb.device, solve_design.dtype
        )
        coefficients = th.linalg.solve(
            gram + self.polynomial_ridge * eye, rhs
        ).to(dtype=rgb.dtype)
        restored = (design_hr / feature_scale) @ coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    def _solve_affine_coefficients(self, source, target):
        batch = source.shape[0]
        design = th.cat([source, th.ones_like(source[..., :1])], dim=-1)
        solve_design = design.double()
        solve_target = target.double()
        gram = solve_design.transpose(1, 2) @ solve_design
        rhs = solve_design.transpose(1, 2) @ solve_target
        eye = self._ridge_eye(batch, 4, source.device, solve_design.dtype)
        return th.linalg.solve(
            gram + self.global_ridge * eye, rhs
        ).to(dtype=source.dtype)

    def _farthest_color_centers(self, source):
        batch, pixels, _ = source.shape
        center_count = min(self.rbf_centers, pixels)
        mean_color = source.mean(dim=1, keepdim=True)
        distances = (source - mean_color).square().sum(dim=-1)
        next_indices = distances.argmax(dim=1)
        minimum_distances = None
        centers = []
        batch_indices = th.arange(batch, device=source.device)
        for _ in range(center_count):
            center = source[batch_indices, next_indices]
            centers.append(center)
            center_distances = (
                source - center[:, None]
            ).square().sum(dim=-1)
            if minimum_distances is None:
                minimum_distances = center_distances
            else:
                minimum_distances = th.minimum(
                    minimum_distances, center_distances
                )
            next_indices = minimum_distances.argmax(dim=1)
        return th.stack(centers, dim=1)

    def _rbf_bandwidth(self, centers):
        pairwise = th.cdist(centers, centers)
        upper_mask = th.triu(
            th.ones(
                pairwise.shape[-2:], device=centers.device, dtype=th.bool
            ),
            diagonal=1,
        )
        bandwidths = []
        for batch_index in range(centers.shape[0]):
            distances = pairwise[batch_index][upper_mask]
            distances = distances[distances > 1e-8]
            if distances.numel() == 0:
                bandwidth = centers.new_tensor(self.rbf_min_bandwidth)
            else:
                bandwidth = distances.median().clamp_min(
                    self.rbf_min_bandwidth
                )
            bandwidths.append(bandwidth)
        return th.stack(bandwidths).view(-1, 1, 1)

    @staticmethod
    def _rbf_kernel(source, centers, bandwidth):
        squared_distance = th.cdist(source, centers).square()
        return th.exp(-squared_distance / (2.0 * bandwidth.square()))

    def _global_rbf(self, rgb, source_lr, target_lr):
        source = source_lr.flatten(2).transpose(1, 2)
        target = target_lr.flatten(2).transpose(1, 2)
        affine_coefficients = self._solve_affine_coefficients(source, target)
        affine_lr = th.cat(
            [source, th.ones_like(source[..., :1])], dim=-1
        ) @ affine_coefficients
        residual_target = target - affine_lr

        source_hr = rgb.flatten(2).transpose(1, 2)
        center_stride = max(1, source_hr.shape[1] // 4096)
        center_candidates = source_hr[:, ::center_stride]
        centers = self._farthest_color_centers(center_candidates)
        bandwidth = self._rbf_bandwidth(centers)
        kernel_hr = self._rbf_kernel(source_hr, centers, bandwidth)
        kernel_maps = kernel_hr.transpose(1, 2).reshape(
            rgb.shape[0], centers.shape[1], *rgb.shape[-2:]
        )
        kernel_lr_maps = self._degrade_rgb(
            kernel_maps, target_lr.shape[-2:]
        )
        kernel_lr = kernel_lr_maps.flatten(2).transpose(1, 2)
        kernel_centers = self._rbf_kernel(centers, centers, bandwidth)
        solve_kernel = kernel_lr.double()
        solve_residual = residual_target.double()
        pixel_count = source.shape[1]
        normal = (
            solve_kernel.transpose(1, 2) @ solve_kernel / pixel_count
            + self.rbf_lambda * kernel_centers.double()
        )
        eye = th.eye(
            centers.shape[1], device=rgb.device, dtype=normal.dtype
        ).expand(rgb.shape[0], -1, -1)
        normal = normal + 1e-6 * eye
        rhs = (
            solve_kernel.transpose(1, 2) @ solve_residual / pixel_count
        )
        residual_coefficients = th.linalg.solve(normal, rhs).to(rgb.dtype)

        affine_hr = th.cat(
            [source_hr, th.ones_like(source_hr[..., :1])], dim=-1
        ) @ affine_coefficients
        restored = affine_hr + kernel_hr @ residual_coefficients
        return restored.transpose(1, 2).reshape_as(rgb)

    def _box_mean(self, tensor):
        padding = self.local_window // 2
        return F.avg_pool2d(
            F.pad(tensor, (padding,) * 4, mode='reflect'),
            self.local_window,
            stride=1,
        )

    def _local_affine(self, rgb, source_lr, target_lr):
        design = th.cat([source_lr, th.ones_like(source_lr[:, :1])], dim=1)
        gram = self._box_mean(
            th.einsum('bihw,bjhw->bijhw', design, design).flatten(1, 2)
        )
        rhs = self._box_mean(
            th.einsum('bihw,bkhw->bikhw', design, target_lr).flatten(1, 2)
        )
        batch, _, height, width = gram.shape
        gram = gram.reshape(batch, 4, 4, height, width).permute(0, 3, 4, 1, 2)
        rhs = rhs.reshape(batch, 4, 3, height, width).permute(0, 3, 4, 1, 2)
        eye = th.eye(4, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, 4, 4)
        eye[..., -1, -1] = 0
        coefficients = th.linalg.solve(gram + self.local_ridge * eye, rhs)
        coefficients = coefficients.permute(0, 3, 4, 1, 2).reshape(
            batch, 12, height, width
        )
        coefficients = F.interpolate(
            coefficients, size=rgb.shape[-2:], mode='bilinear', align_corners=False
        ).reshape(batch, 4, 3, *rgb.shape[-2:])
        design_hr = th.cat([rgb, th.ones_like(rgb[:, :1])], dim=1)
        return th.einsum('bikhw,bihw->bkhw', coefficients, design_hr)

    def forward(self, rgb, lq_hsi):
        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f'Expected {self.rgb_channels} RGB channels, got {rgb.shape[1]}.'
            )
        expected_size = (
            lq_hsi.shape[-2] * self.scale,
            lq_hsi.shape[-1] * self.scale,
        )
        if rgb.shape[-2:] != expected_size:
            raise ValueError(
                f'RGB size {tuple(rgb.shape[-2:])} must equal LR-HSI size '
                f'{tuple(lq_hsi.shape[-2:])} multiplied by scale {self.scale}.'
            )

        rgb_float = rgb.float()
        target_lr = self.project_hsi(lq_hsi.float())
        source_lr = self._degrade_rgb(rgb_float, target_lr.shape[-2:])
        if self.mode == 'global_affine':
            restored = self._global_affine(rgb_float, source_lr, target_lr)
        elif self.mode == 'global_polynomial':
            restored = self._global_polynomial(rgb_float, source_lr, target_lr)
        elif self.mode == 'global_forward_polynomial':
            restored = self._global_forward_polynomial(rgb_float, target_lr)
        elif self.mode == 'global_full_polynomial':
            restored = self._global_full_polynomial(
                rgb_float, source_lr, target_lr
            )
        elif self.mode == 'global_forward_full_polynomial':
            restored = self._global_forward_full_polynomial(
                rgb_float, target_lr
            )
        elif self.mode == 'global_rbf':
            restored = self._global_rbf(rgb_float, source_lr, target_lr)
        else:
            restored = self._local_affine(rgb_float, source_lr, target_lr)

        for _ in range(self.backprojection_steps):
            low_residual = target_lr - self._degrade_rgb(
                restored, target_lr.shape[-2:]
            )
            restored = restored + F.interpolate(
                low_residual,
                size=rgb.shape[-2:],
                mode='bicubic',
                align_corners=False,
            )

        if self.identity_threshold > 0:
            identity_mask = (
                (source_lr - target_lr).abs().mean(dim=(1, 2, 3), keepdim=True)
                <= self.identity_threshold
            )
            restored = th.where(identity_mask, rgb_float, restored)
        if self.clamp_output:
            restored = restored.clamp(0.0, 1.0)
        return restored, target_lr, restored - rgb_float


class ConditionalRGBCanonicalizer(nn.Module):
    """Map unknown-camera RGB into the fixed RGB domain conditioned on LR-HSI."""

    def __init__(
            self,
            rgb_channels=3,
            hsi_channels=31,
            base_channels=32,
            hsi_feature_channels=64,
            lq_downsample_factor=8,
            local_norm_kernel=5,
            highpass_kernel=5,
            use_gate=False,
            gate_init_bias=-4.0,
    ):
        super().__init__()
        for name, value in (
                ('local_norm_kernel', local_norm_kernel),
                ('highpass_kernel', highpass_kernel),
        ):
            if int(value) < 3 or int(value) % 2 == 0:
                raise ValueError(f'{name} must be an odd integer >= 3.')
        if int(lq_downsample_factor) < 1:
            raise ValueError('canonicalizer_lq_downsample_factor must be >= 1.')

        self.rgb_channels = int(rgb_channels)
        self.hsi_channels = int(hsi_channels)
        self.base_channels = int(base_channels)
        self.lq_downsample_factor = int(lq_downsample_factor)
        self.local_norm_kernel = int(local_norm_kernel)
        self.highpass_kernel = int(highpass_kernel)
        self.use_gate = bool(use_gate)

        channels0 = self.base_channels
        channels1 = self.base_channels + 16
        channels2 = self.base_channels * 2
        if any(channels % 8 != 0 for channels in (channels0, channels1, channels2)):
            raise ValueError(
                'canonicalizer channel widths must be divisible by 8; '
                'use a base_channels value compatible with base+16 and base*2.'
            )

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(self.rgb_channels * 2, channels0, 3, padding=1),
            CanonicalResidualBlock(channels0),
        )
        self.rgb_down1 = nn.Sequential(
            nn.Conv2d(channels0, channels1, 3, stride=2, padding=1),
            CanonicalResidualBlock(channels1),
        )
        self.rgb_down2 = nn.Sequential(
            nn.Conv2d(channels1, channels2, 3, stride=2, padding=1),
            CanonicalResidualBlock(channels2),
        )

        self.hsi_encoder = nn.Sequential(
            nn.Conv2d(self.hsi_channels, hsi_feature_channels, 3, padding=1),
            nn.GroupNorm(8, hsi_feature_channels),
            nn.LeakyReLU(0.2),
            CanonicalResidualBlock(hsi_feature_channels),
            CanonicalResidualBlock(hsi_feature_channels),
        )
        self.film2 = nn.Conv2d(hsi_feature_channels, channels2 * 2, 1)
        self.film1 = nn.Conv2d(hsi_feature_channels, channels1 * 2, 1)
        self.film0 = nn.Conv2d(hsi_feature_channels, channels0 * 2, 1)
        for film in (self.film2, self.film1, self.film0):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        self.decode1 = nn.Sequential(
            nn.Conv2d(channels2 + channels1, channels1, 3, padding=1),
            CanonicalResidualBlock(channels1),
        )
        self.decode0 = nn.Sequential(
            nn.Conv2d(channels1 + channels0, channels0, 3, padding=1),
            CanonicalResidualBlock(channels0),
        )

        self.low_head = nn.Conv2d(channels2, self.rgb_channels, 1)
        self.high_head = nn.Conv2d(channels0, self.rgb_channels, 1)
        nn.init.zeros_(self.low_head.weight)
        nn.init.zeros_(self.low_head.bias)
        nn.init.zeros_(self.high_head.weight)
        nn.init.zeros_(self.high_head.bias)

        self.gate_head = None
        if self.use_gate:
            gate_in_channels = channels2 + hsi_feature_channels + self.rgb_channels * 2
            self.gate_head = nn.Sequential(
                nn.Linear(gate_in_channels, base_channels),
                nn.LeakyReLU(0.2),
                nn.Linear(base_channels, 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, float(gate_init_bias))

    @staticmethod
    def apply_gate_mode(gate, mode='soft', threshold=0.5):
        mode = str(mode).lower()
        if mode == 'soft':
            return gate
        if mode == 'binary':
            return (gate >= float(threshold)).to(dtype=gate.dtype)
        if mode == 'force_zero':
            return th.zeros_like(gate)
        if mode == 'force_one':
            return th.ones_like(gate)
        raise ValueError(
            "canonicalizer gate mode must be 'soft', 'binary', "
            "'force_zero', or 'force_one'."
        )

    def _local_contrast_normalize(self, rgb):
        padding = self.local_norm_kernel // 2
        padded = F.pad(rgb.float(), (padding, padding, padding, padding), mode='reflect')
        mean = F.avg_pool2d(padded, self.local_norm_kernel, stride=1)
        mean_square = F.avg_pool2d(
            padded * padded, self.local_norm_kernel, stride=1
        )
        std = (mean_square - mean * mean).clamp_min(1e-6).sqrt()
        return (rgb.float() - mean) / std

    def _film(self, rgb_feature, hsi_feature, layer):
        condition = F.interpolate(
            hsi_feature,
            size=rgb_feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        gamma, beta = layer(condition).chunk(2, dim=1)
        return rgb_feature * (1.0 + gamma) + beta

    def _highpass(self, feature):
        padding = self.highpass_kernel // 2
        padded = F.pad(feature, (padding, padding, padding, padding), mode='reflect')
        low = F.avg_pool2d(padded, self.highpass_kernel, stride=1)
        return feature - low

    def forward(
            self,
            rgb,
            lq,
            gate_mode='soft',
            gate_threshold=0.5,
            return_gate=False,
    ):
        if rgb.shape[1] != self.rgb_channels:
            raise ValueError(
                f'Expected {self.rgb_channels} RGB channels, got {rgb.shape[1]}.'
            )
        if lq.shape[1] != self.hsi_channels:
            raise ValueError(
                f'Expected {self.hsi_channels} LR-HSI channels, got {lq.shape[1]}.'
            )

        rgb_float = rgb.float()
        rgb_input = th.cat(
            [rgb_float, self._local_contrast_normalize(rgb_float)], dim=1
        )
        rgb0 = self.rgb_stem(rgb_input)
        rgb1 = self.rgb_down1(rgb0)
        rgb2 = self.rgb_down2(rgb1)

        target_h = max(1, lq.shape[-2] // self.lq_downsample_factor)
        target_w = max(1, lq.shape[-1] // self.lq_downsample_factor)
        lq_low = F.interpolate(
            lq.float(), size=(target_h, target_w), mode='area'
        )
        hsi_feature = self.hsi_encoder(lq_low)

        fused2 = self._film(rgb2, hsi_feature, self.film2)
        decoded1 = F.interpolate(
            fused2, size=rgb1.shape[-2:], mode='bilinear', align_corners=False
        )
        decoded1 = self.decode1(th.cat([decoded1, rgb1], dim=1))
        decoded1 = self._film(decoded1, hsi_feature, self.film1)

        decoded0 = F.interpolate(
            decoded1, size=rgb0.shape[-2:], mode='bilinear', align_corners=False
        )
        decoded0 = self.decode0(th.cat([decoded0, rgb0], dim=1))
        decoded0 = self._film(decoded0, hsi_feature, self.film0)

        delta_low = self.low_head(fused2)
        delta_low = F.interpolate(
            delta_low, size=rgb.shape[-2:], mode='bilinear', align_corners=False
        )
        delta_high = self._highpass(self.high_head(decoded0))
        correction = delta_low + delta_high

        if self.gate_head is None:
            learned_gate = th.ones(
                (rgb.shape[0], 1, 1, 1),
                device=rgb.device,
                dtype=correction.dtype,
            )
        else:
            rgb_mean = rgb_float.mean(dim=(2, 3))
            rgb_std = rgb_float.std(dim=(2, 3), unbiased=False)
            gate_feature = th.cat([
                F.adaptive_avg_pool2d(rgb2, 1).flatten(1),
                F.adaptive_avg_pool2d(hsi_feature, 1).flatten(1),
                rgb_mean,
                rgb_std,
            ], dim=1)
            learned_gate = th.sigmoid(self.gate_head(gate_feature))
            learned_gate = learned_gate.view(-1, 1, 1, 1)

        applied_gate = self.apply_gate_mode(
            learned_gate, mode=gate_mode, threshold=gate_threshold
        )
        canonical_rgb = (rgb_float + applied_gate * correction).clamp(0.0, 1.0)
        if return_gate:
            return canonical_rgb, delta_low, delta_high, learned_gate
        return canonical_rgb, delta_low, delta_high


class BAFUNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            lqrgb_channels=34,
            rgb_channels=3,
            norm_groups=34,
            condition_use_rgb=True,
            condition_rgb_mode='raw',
            condition_rgb_instance_norm=False,
            use_control_adapter_fusion=False,
            control_adapter_layers=None,
            control_adapter_rgb_mode='raw_plus_grad',
            control_adapter_channels=64,
            control_adapter_type='simple',
            control_adapter_msgab_blocks=2,
            control_adapter_injection='residual',
            control_adapter_mid=False,
            control_residual_zero_init=True,
            control_residual_scale_init=1.0,
            control_gate_channels=1,
            control_gate_init_bias=-3.0,
            use_edge_controlnet=False,
            edge_control_layers=None,
            edge_control_channels=64,
            edge_control_zero_init=True,
            edge_control_scale_init=1.0,
            use_factorized_rgb_detail=False,
            factorized_spatial_channels=8,
            factorized_hidden_channels=32,
            factorized_coefficient_scale=4,
            factorized_local_norm_kernel=5,
            factorized_gate_init_bias=-2.0,
            factorized_residual_scale_init=0.1,
            factorized_use_highpass=False,
            factorized_highpass_kernel=5,
            use_rgb_canonicalizer=False,
            canonicalizer_base_channels=32,
            canonicalizer_hsi_feature_channels=64,
            canonicalizer_lq_downsample_factor=8,
            canonicalizer_local_norm_kernel=5,
            canonicalizer_highpass_kernel=5,
            canonicalizer_use_gate=False,
            canonicalizer_gate_init_bias=-4.0,
            use_nikon_rgb_restorer=False,
            nikon_restorer_hsi_channels=None,
            nikon_restorer_scale=8,
            nikon_restorer_hidden_channels=96,
            nikon_restorer_blocks=4,
            nikon_restorer_mode='learned_bicubic',
            nikon_restorer_global_ridge=1e-2,
            nikon_restorer_polynomial_ridge=1e-4,
            nikon_restorer_polynomial_degree=3,
            nikon_restorer_rbf_centers=64,
            nikon_restorer_rbf_lambda=1e-3,
            nikon_restorer_rbf_min_bandwidth=2e-2,
            nikon_restorer_local_ridge=1e-6,
            nikon_restorer_local_window=11,
            nikon_restorer_identity_threshold=0.0,
            nikon_restorer_backprojection_steps=1,
            nikon_restorer_normalize_target_srf=False,
            nikon_restorer_allow_signed_target_srf=True,
            nikon_restorer_clamp_output=False,
    ):
        global band
        band = norm_groups
        self._groups_list = [1, band]
        super().__init__()

        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            assert len(num_res_blocks) == len(channel_mult)
        self.num_res_blocks = num_res_blocks

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.lqrgb_channels = lqrgb_channels
        self.rgb_channels = rgb_channels
        self.condition_use_rgb = condition_use_rgb
        self.condition_rgb_mode = condition_rgb_mode
        self.condition_rgb_instance_norm = condition_rgb_instance_norm
        if self.condition_rgb_mode not in ('raw', 'grad_mag'):
            raise ValueError("condition_rgb_mode must be one of: 'raw', 'grad_mag'.")
        self.use_control_adapter_fusion = use_control_adapter_fusion
        self.control_adapter_layers = set(control_adapter_layers or [])
        self.control_adapter_rgb_mode = control_adapter_rgb_mode
        self.control_adapter_channels = control_adapter_channels
        self.control_adapter_type = control_adapter_type
        self.control_adapter_msgab_blocks = control_adapter_msgab_blocks
        self.control_adapter_injection = control_adapter_injection
        self.control_adapter_mid = control_adapter_mid
        if self.control_adapter_injection not in ('residual', 'encoder_residual', 'gated_encoder_residual'):
            raise ValueError(
                "control_adapter_injection must be one of: 'residual', 'encoder_residual', 'gated_encoder_residual'."
            )
        self.control_residual_zero_init = control_residual_zero_init
        self.control_residual_scale_init = control_residual_scale_init
        self.control_gate_channels = control_gate_channels
        self.control_gate_init_bias = control_gate_init_bias
        control_adapter_layer_ids = set(self.control_adapter_layers)
        if self.control_adapter_mid:
            control_adapter_layer_ids.add('mid')
        self.control_adapter = RGBMultiScaleControlAdapter(
            layer_ids=control_adapter_layer_ids,
            hidden_channels=self.control_adapter_channels,
            rgb_mode=self.control_adapter_rgb_mode,
            adapter_type=self.control_adapter_type,
            num_msgab_blocks=self.control_adapter_msgab_blocks,
        ) if use_control_adapter_fusion else None
        self.control_residual_blocks = nn.ModuleDict()
        self.control_mid_residual_block = None
        self.use_edge_controlnet = use_edge_controlnet
        self.edge_control_layers = set(edge_control_layers or [])
        self.edge_control_channels = edge_control_channels
        self.edge_control_zero_init = edge_control_zero_init
        self.edge_control_scale_init = edge_control_scale_init
        self.edge_control_blocks = nn.ModuleDict()
        self.condition_channels = lqrgb_channels
        if self.condition_channels <= 0:
            raise ValueError(
                f"Invalid condition_channels={self.condition_channels}. "
                "Check lqrgb_channels and condition_use_rgb."
            )
        self.use_factorized_rgb_detail = bool(use_factorized_rgb_detail)
        if self.use_factorized_rgb_detail and self.condition_use_rgb:
            raise ValueError(
                'use_factorized_rgb_detail=True requires condition_use_rgb=False '
                'so raw RGB cannot bypass the factorized branch.'
            )
        self.factorized_rgb_detail = None
        if self.use_factorized_rgb_detail:
            self.factorized_rgb_detail = FactorizedRGBDetailBranch(
                rgb_channels=rgb_channels,
                hsi_channels=self.condition_channels,
                out_channels=out_channels,
                spatial_channels=factorized_spatial_channels,
                hidden_channels=factorized_hidden_channels,
                coefficient_scale=factorized_coefficient_scale,
                local_norm_kernel=factorized_local_norm_kernel,
                gate_init_bias=factorized_gate_init_bias,
                residual_scale_init=factorized_residual_scale_init,
                use_highpass=factorized_use_highpass,
                highpass_kernel=factorized_highpass_kernel,
            )
        self.use_rgb_canonicalizer = bool(use_rgb_canonicalizer)
        if self.use_rgb_canonicalizer and not self.condition_use_rgb:
            raise ValueError(
                'use_rgb_canonicalizer=True requires condition_use_rgb=True.'
            )
        if self.use_rgb_canonicalizer and self.use_factorized_rgb_detail:
            raise ValueError(
                'RGB canonicalizer and factorized RGB detail cannot be enabled together.'
            )
        self.rgb_canonicalizer = None
        if self.use_rgb_canonicalizer:
            self.rgb_canonicalizer = ConditionalRGBCanonicalizer(
                rgb_channels=rgb_channels,
                hsi_channels=in_channels,
                base_channels=canonicalizer_base_channels,
                hsi_feature_channels=canonicalizer_hsi_feature_channels,
                lq_downsample_factor=canonicalizer_lq_downsample_factor,
                local_norm_kernel=canonicalizer_local_norm_kernel,
                highpass_kernel=canonicalizer_highpass_kernel,
                use_gate=canonicalizer_use_gate,
                gate_init_bias=canonicalizer_gate_init_bias,
            )
        self.use_nikon_rgb_restorer = bool(use_nikon_rgb_restorer)
        if self.use_nikon_rgb_restorer and not self.condition_use_rgb:
            raise ValueError(
                'use_nikon_rgb_restorer=True requires condition_use_rgb=True.'
            )
        if self.use_nikon_rgb_restorer and (
                self.use_rgb_canonicalizer or self.use_factorized_rgb_detail
        ):
            raise ValueError(
                'Nikon RGB restorer, legacy RGB canonicalizer, and factorized detail '
                'are mutually exclusive.'
            )
        self.nikon_rgb_restorer = None
        if self.use_nikon_rgb_restorer:
            # HSI anchor for the restorer is the true LR-HSI (band_dim), which
            # may differ from in_channels when fake-MSI pads the input.
            restorer_hsi_channels = (
                int(nikon_restorer_hsi_channels)
                if nikon_restorer_hsi_channels is not None
                else in_channels
            )
            if nikon_restorer_mode == 'learned_bicubic':
                self.nikon_rgb_restorer = NikonRGBRestorer(
                    rgb_channels=rgb_channels,
                    hsi_channels=restorer_hsi_channels,
                    scale=nikon_restorer_scale,
                    hidden_channels=nikon_restorer_hidden_channels,
                    num_blocks=nikon_restorer_blocks,
                )
            else:
                self.nikon_rgb_restorer = FixedSRFRGBCanonicalizer(
                    rgb_channels=rgb_channels,
                    hsi_channels=restorer_hsi_channels,
                    scale=nikon_restorer_scale,
                    mode=nikon_restorer_mode,
                    global_ridge=nikon_restorer_global_ridge,
                    polynomial_ridge=nikon_restorer_polynomial_ridge,
                    polynomial_degree=nikon_restorer_polynomial_degree,
                    rbf_centers=nikon_restorer_rbf_centers,
                    rbf_lambda=nikon_restorer_rbf_lambda,
                    rbf_min_bandwidth=nikon_restorer_rbf_min_bandwidth,
                    local_ridge=nikon_restorer_local_ridge,
                    local_window=nikon_restorer_local_window,
                    identity_threshold=nikon_restorer_identity_threshold,
                    backprojection_steps=nikon_restorer_backprojection_steps,
                    normalize_target_srf=nikon_restorer_normalize_target_srf,
                    allow_signed_target_srf=nikon_restorer_allow_signed_target_srf,
                    clamp_output=nikon_restorer_clamp_output,
                )

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.LeakyReLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.init_channels = model_channels
        ch = input_ch = int(channel_mult[0] * self.init_channels)

        self.input_blocks_denoise = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, size, padding=padding))]
        )


        self.input_blocks_lqrgb = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, self.condition_channels, ch, size, padding=padding))]
        )

        input_block_chans = [ch]
        ds = image_size

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                current_out_ch = int(mult * self.init_channels)

                denoise_layers = [
                    MSGABT(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=current_out_ch,
                        dims=dims,
                        groups_list=self._groups_list,
                        norm_groups=norm_groups,
                    )
                ]

                lq_layers = [
                    MSGAB(
                        ch,
                        dropout,
                        out_channels=current_out_ch,
                        dims=dims,
                        groups_list=self._groups_list,
                        norm_groups=norm_groups,
                    )
                ]

                ch = current_out_ch
                self.input_blocks_denoise.append(TimestepEmbedSequential(*denoise_layers))
                self.input_blocks_lqrgb.append(nn.Sequential(*lq_layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                denoise_down = TimestepEmbedSequential(
                    MSGABT(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        down=True,
                        groups_list=self._groups_list,
                        norm_groups=norm_groups,
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )

                lq_down = nn.Sequential(
                    MSGAB(
                        ch,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        down=True,
                        groups_list=self._groups_list,
                        norm_groups=norm_groups,
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )

                self.input_blocks_denoise.append(denoise_down)
                self.input_blocks_lqrgb.append(lq_down)
                ch = out_ch
                input_block_chans.append(ch)
                ds //= 2

        self.middle_block = nn.Sequential(
            MSGAB(
                ch * 2,
                dropout,
                dims=dims,
                groups_list=self._groups_list,
                norm_groups=norm_groups,
            ),
        )
        ch = ch * 2
        if self.use_control_adapter_fusion and self.control_adapter_mid:
            if self.control_adapter_injection == 'encoder_residual':
                self.control_mid_residual_block = ControlResidualBlock(
                    in_channels=self.control_adapter_channels,
                    out_channels=ch,
                    zero_init=self.control_residual_zero_init,
                    scale_init=self.control_residual_scale_init,
                )
            elif self.control_adapter_injection == 'gated_encoder_residual':
                self.control_mid_residual_block = LQAwareGatedControlResidualBlock(
                    control_channels=self.control_adapter_channels,
                    lq_channels=ch // 2,
                    out_channels=ch,
                    hidden_channels=self.control_adapter_channels,
                    gate_channels=self.control_gate_channels if self.control_gate_channels != 1 else 1,
                    gate_init_bias=self.control_gate_init_bias,
                    zero_init=self.control_residual_zero_init,
                    scale_init=self.control_residual_scale_init,
                )
        self.output_blocks = nn.ModuleList([])
        output_block_idx = 0
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    MSGAB(
                        ch + ich * 2,
                        dropout,
                        out_channels=int(self.init_channels * mult),
                        dims=dims,
                        groups_list=self._groups_list,
                        norm_groups=norm_groups,
                    )
                ]
                ch = int(self.init_channels * mult)
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(
                            MSGAB(
                                ch,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                groups_list=self._groups_list,
                                norm_groups=norm_groups,
                            )
                        )
                    else:
                        layers.append(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                    ds *= 2
                self.output_blocks.append(nn.Sequential(*layers))
                if self.use_control_adapter_fusion and output_block_idx in self.control_adapter_layers:
                    if self.control_adapter_injection == 'gated_encoder_residual':
                        self.control_residual_blocks[str(output_block_idx)] = LQAwareGatedControlResidualBlock(
                            control_channels=self.control_adapter_channels,
                            lq_channels=ch,
                            out_channels=ch,
                            hidden_channels=self.control_adapter_channels,
                            gate_channels=self.control_gate_channels if self.control_gate_channels != 1 else 1,
                            gate_init_bias=self.control_gate_init_bias,
                            zero_init=self.control_residual_zero_init,
                            scale_init=self.control_residual_scale_init,
                        )
                    else:
                        self.control_residual_blocks[str(output_block_idx)] = ControlResidualBlock(
                            in_channels=self.control_adapter_channels,
                            out_channels=ch,
                            zero_init=self.control_residual_zero_init,
                            scale_init=self.control_residual_scale_init,
                        )
                if self.use_edge_controlnet and output_block_idx in self.edge_control_layers:
                    self.edge_control_blocks[str(output_block_idx)] = EdgeControlBlock(
                        out_channels=ch,
                        hidden_channels=self.edge_control_channels,
                        zero_init=self.edge_control_zero_init,
                        scale_init=self.edge_control_scale_init,
                    )
                output_block_idx += 1

    def _rgb_grad_mag(self, rgb):
        rgb = rgb.float()
        gray_weight = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        sobel_x = rgb.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = rgb.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        gray = F.conv2d(rgb, gray_weight)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = th.sqrt(gx * gx + gy * gy + 1e-8)
        mean = grad_mag.mean(dim=(2, 3), keepdim=True)
        std = grad_mag.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (grad_mag - mean) / std

    def _condition_aux(self, rgb):
        if self.condition_rgb_mode == 'raw':
            condition = rgb
            if self.condition_rgb_instance_norm:
                condition_float = condition.float()
                mean = condition_float.mean(dim=(2, 3), keepdim=True)
                var = condition_float.var(dim=(2, 3), keepdim=True, unbiased=False)
                condition = (condition_float - mean) * th.rsqrt(var + 1e-6)
            return condition
        if self.condition_rgb_mode == 'grad_mag':
            return self._rgb_grad_mag(rgb)
        raise ValueError(f'Unsupported condition_rgb_mode: {self.condition_rgb_mode}')

    def forward(
            self,
            x,
            rgb,
            lq,
            timesteps,
            control_rgb=None,
            return_condition_features=False,
            return_rgb_spatial_feature=False,
            return_canonical_rgb=False,
            return_canonical_gate=False,
            canonicalizer_gate_mode='soft',
            canonicalizer_gate_threshold=0.5,
            rgb_restorer_lq=None,
            apply_nikon_rgb_restorer=True,
    ):

        canonical_rgb = rgb
        canonical_gate = None
        if self.nikon_rgb_restorer is not None and apply_nikon_rgb_restorer:
            if rgb_restorer_lq is None:
                raise ValueError(
                    'rgb_restorer_lq is required when the Nikon RGB restorer is enabled.'
                )
            canonical_rgb, _, _ = self.nikon_rgb_restorer(rgb, rgb_restorer_lq)
        elif self.rgb_canonicalizer is not None:
            canonical_rgb, _, _, canonical_gate = self.rgb_canonicalizer(
                rgb,
                lq,
                gate_mode=canonicalizer_gate_mode,
                gate_threshold=canonicalizer_gate_threshold,
                return_gate=True,
            )
        elif return_canonical_rgb and (
                self.nikon_rgb_restorer is None or apply_nikon_rgb_restorer
        ):
            raise ValueError(
                'return_canonical_rgb=True requires an RGB restoration module.'
            )
        model_rgb = canonical_rgb

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)).type(self.dtype)
        hs_denoise = []
        h_denoise = x.type(self.dtype)
        for module in self.input_blocks_denoise:
            h_denoise = module(h_denoise, emb)
            hs_denoise.append(h_denoise)
        hs_lqrgb = []
        h_lq = th.cat([lq, self._condition_aux(model_rgb)], dim=1) if self.condition_use_rgb else lq
        h_lqrgb = h_lq.type(self.dtype)
        for module in self.input_blocks_lqrgb:
            h_lqrgb = module(h_lqrgb)
            hs_lqrgb.append(h_lqrgb)

        control_feats = None
        if self.control_adapter is not None:
            adapter_rgb = control_rgb if control_rgb is not None else model_rgb
            target_sizes = {}
            for idx in self.control_adapter.layer_ids:
                if idx == 'mid':
                    target_sizes[idx] = hs_lqrgb[-1].shape[-2:]
                    continue
                idx_int = int(idx)
                if 0 <= idx_int < len(hs_lqrgb):
                    target_sizes[idx] = hs_lqrgb[-(idx_int + 1)].shape[-2:]
            control_feats = self.control_adapter(adapter_rgb, target_sizes)
            if self.control_adapter_injection in ('encoder_residual', 'gated_encoder_residual'):
                for idx, control_feat in control_feats.items():
                    if idx == 'mid' or idx not in self.control_residual_blocks:
                        continue
                    idx_int = int(idx)
                    hs_idx = -(idx_int + 1)
                    if self.control_adapter_injection == 'gated_encoder_residual':
                        residual = self.control_residual_blocks[idx](
                            control_feat,
                            hs_lqrgb[hs_idx],
                            hs_lqrgb[hs_idx].shape[-2:],
                            hs_lqrgb[hs_idx].dtype,
                        )
                    else:
                        residual = self.control_residual_blocks[idx](
                            control_feat, hs_lqrgb[hs_idx].shape[-2:], hs_lqrgb[hs_idx].dtype
                        )
                    hs_lqrgb[hs_idx] = hs_lqrgb[hs_idx] + residual
                h_lqrgb = hs_lqrgb[-1]

        h = th.cat([h_denoise, h_lqrgb], dim=1)
        h = self.middle_block(h)
        if control_feats is not None and 'mid' in control_feats and self.control_mid_residual_block is not None:
            if self.control_adapter_injection == 'gated_encoder_residual':
                h = h + self.control_mid_residual_block(control_feats['mid'], h_lqrgb, h.shape[-2:], h.dtype)
            else:
                h = h + self.control_mid_residual_block(control_feats['mid'], h.shape[-2:], h.dtype)
        if self.control_adapter_injection in ('encoder_residual', 'gated_encoder_residual'):
            control_feats = None
        edge_control = self._rgb_grad_mag(model_rgb) if self.use_edge_controlnet else None
        condition_features_for_loss = tuple(hs_lqrgb) if return_condition_features else None
        upsample_outputs = []
        for i, module in enumerate(self.output_blocks):
            denoise_skip = hs_denoise.pop()
            lqrgb_skip = hs_lqrgb.pop()
            skip = th.cat([denoise_skip, lqrgb_skip], dim=1)
            h = th.cat([h, skip], dim=1)
            h = module(h)
            fusion_key = str(i)
            if control_feats is not None and fusion_key in self.control_residual_blocks:
                h = h + self.control_residual_blocks[fusion_key](control_feats[fusion_key], h.shape[-2:], h.dtype)
            if edge_control is not None and fusion_key in self.edge_control_blocks:
                h = h + self.edge_control_blocks[fusion_key](edge_control, h.shape[-2:], h.dtype)
            upsample_outputs.append(h)

        rgb_spatial_feature = None
        if self.factorized_rgb_detail is not None:
            detail, rgb_spatial_feature, _ = self.factorized_rgb_detail(model_rgb, lq)
            if detail.shape != h.shape:
                raise ValueError(
                    f'Factorized RGB detail shape {tuple(detail.shape)} does not match '
                    f'network output shape {tuple(h.shape)}.'
                )
            h = h + detail
        elif return_rgb_spatial_feature:
            raise ValueError(
                'return_rgb_spatial_feature=True requires use_factorized_rgb_detail=True.'
            )

        extras = []
        if return_condition_features:
            extras.append(condition_features_for_loss)
        if return_rgb_spatial_feature:
            extras.append(rgb_spatial_feature)
        if return_canonical_rgb:
            extras.append(canonical_rgb)
        if return_canonical_gate:
            if canonical_gate is None:
                raise ValueError(
                    'return_canonical_gate=True requires use_rgb_canonicalizer=True.'
                )
            extras.append(canonical_gate)
        if extras:
            return (h, upsample_outputs, *extras)
        return h, upsample_outputs

    def convert_to_fp16(self):
        self.input_blocks_denoise.apply(convert_module_to_f16)
        self.input_blocks_lqrgb.apply(convert_module_to_f16)

        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        if self.control_adapter is not None:
            self.control_adapter.apply(convert_module_to_f16)
        if self.factorized_rgb_detail is not None:
            self.factorized_rgb_detail.apply(convert_module_to_f16)
        if self.rgb_canonicalizer is not None:
            self.rgb_canonicalizer.apply(convert_module_to_f16)
        if self.nikon_rgb_restorer is not None:
            self.nikon_rgb_restorer.apply(convert_module_to_f16)
        self.control_residual_blocks.apply(convert_module_to_f16)
        if self.control_mid_residual_block is not None:
            self.control_mid_residual_block.apply(convert_module_to_f16)
        self.edge_control_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks_denoise.apply(convert_module_to_f32)
        self.input_blocks_lqrgb.apply(convert_module_to_f32)

        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        if self.control_adapter is not None:
            self.control_adapter.apply(convert_module_to_f32)
        if self.factorized_rgb_detail is not None:
            self.factorized_rgb_detail.apply(convert_module_to_f32)
        if self.rgb_canonicalizer is not None:
            self.rgb_canonicalizer.apply(convert_module_to_f32)
        if self.nikon_rgb_restorer is not None:
            self.nikon_rgb_restorer.apply(convert_module_to_f32)
        self.control_residual_blocks.apply(convert_module_to_f32)
        if self.control_mid_residual_block is not None:
            self.control_mid_residual_block.apply(convert_module_to_f32)
        self.edge_control_blocks.apply(convert_module_to_f32)

class MSGAB(nn.Module):

    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            kernel_size=7,
            up=False,
            down=False,
            mlp_ratio=4.0,
            groups_list=[1, band],
            norm_groups=34,

    ):
        super().__init__()
        self.channels = channels
        self.groups_list = groups_list
        self.group_convs = nn.ModuleList()
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.sptt = SpatialAttention(kernel_size)
        self.in_layers = nn.Sequential(
            normalization(channels, norm_groups),
            nn.LeakyReLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=kernel_size // 2),
        )

        self.updown = up or down
        self.LayerNorm2d = LayerNorm2d(self.channels)
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=kernel_size // 2
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        mlp_hidden_dim = int(self.out_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.out_channels * 2,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_channels,
            act_layer=nn.LeakyReLU,
            drop=dropout
        )

        for g in groups_list:
            self.group_convs.append(nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels * 2),
                nn.Conv2d(self.out_channels * 2, self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list) * 2, len(groups_list) * 2, 1),
            nn.Softmax(dim=1)
        )

        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):

        group_bothoutputs = []
        h1 = self.scale1(x)
        h = self.sptt(h1) * h1
        for conv in self.group_convs:
            group_bothoutputs.append(conv(h))
        attn_weights = self.attention(torch.cat([
            F.adaptive_avg_pool2d(out, 1) for out in group_bothoutputs
        ], dim=1))
        h = sum(attn_weights[:, i:i + 1] * group_bothoutputs[i]
                for i in range(len(self.groups_list)))
        h = self.mlp(h)
        return h + self.skip_connection(x)

class MSGABT(TimestepBlock):

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=7,
        mlp_ratio=4.0,
        groups_list=[1, band],
        norm_groups=34,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.groups_list = groups_list
        self.group_convs = nn.ModuleList()

        self.in_layers = nn.Sequential(
            normalization(channels, norm_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, size, padding=padding),
        )
        self.LayerNorm2d = LayerNorm2d(self.channels)
        self.updown = up or down
        self.sptt = SpatialAttention(kernel_size)
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        mlp_hidden_dim = int(self.out_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.out_channels*2,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_channels,
            act_layer=nn.LeakyReLU,
            drop=dropout
        )
        for g in groups_list:
            self.group_convs.append(nn.Sequential(
                nn.Conv2d(self.out_channels , self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
                LayerNorm2d(self.out_channels * 2),
                nn.Conv2d(self.out_channels * 2, self.out_channels * 2, 3, padding=1, groups=g, bias=False),
                nn.LeakyReLU(0.2),
            ))

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * len(groups_list) * 2, len(groups_list) * 2, 1),
            nn.Softmax(dim=1)
        )
        self.emb_layers = nn.Sequential(
            nn.LeakyReLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.scale1 = nn.Sequential(
            nn.Conv2d(channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(self.out_channels),
            nn.LeakyReLU(0.2)
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, size, padding=padding
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
            group_bothoutputs = []
            h1 = self.scale1(x)
            h1 = self.sptt(h1) * h1
            emb_out = self.emb_layers(emb).type(h1.dtype)
            while len(emb_out.shape) < len(h1.shape):
                emb_out = emb_out[..., None]
            h = h1 + emb_out
            for conv in self.group_convs:
                group_bothoutputs.append(conv(h))
            attn_weights = self.attention(torch.cat([
                F.adaptive_avg_pool2d(out, 1) for out in group_bothoutputs
            ], dim=1))
            h11 = sum(attn_weights[:, i:i + 1] * group_bothoutputs[i]
                               for i in range(len(self.groups_list)))
            h = self.mlp(h11)
            return h + self.skip_connection(x)
class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x):
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
            return x
