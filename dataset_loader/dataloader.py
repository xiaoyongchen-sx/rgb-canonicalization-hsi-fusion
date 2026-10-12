
import glob
import random
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import torch.nn.functional as F


def _random_row_normalized_srf(bands, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    # Internal shape is [3, bands] because rgb = SRF @ HSI.
    # Each RGB row sums to 1, so RGB stays in the HSI data range.
    srf = torch.rand((3, int(bands)), generator=generator).float()
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _dynamic_random_row_normalized_srf(bands):
    """Sample a new SRF from the DataLoader worker RNG on every fetch."""
    srf = torch.rand((3, int(bands))).float()
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)




def _sample_log_uniform(min_value, max_value):
    min_value = float(min_value)
    max_value = float(max_value)
    if min_value <= 0 or max_value < min_value:
        raise ValueError('Log-uniform bounds must satisfy 0 < min <= max.')
    low = torch.log(torch.tensor(min_value))
    high = torch.log(torch.tensor(max_value))
    return torch.exp(low + torch.rand(()) * (high - low))


def _augment_rgb_for_canonicalization(gt, clean_rgb, opt, return_flags=False):
    """Apply independent photometric/SRF shifts while keeping clean RGB as target."""
    probability_names = (
        'exposure_probability',
        'white_balance_probability',
        'color_matrix_probability',
        'random_srf_probability',
    )
    probabilities = {name: float(opt.get(name, 0.0)) for name in probability_names}
    for name, probability in probabilities.items():
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f'{name} must be in [0, 1], got {probability}.')

    flags = {
        'random_srf': torch.rand(()).item() < probabilities['random_srf_probability'],
        'exposure': torch.rand(()).item() < probabilities['exposure_probability'],
        'white_balance': torch.rand(()).item() < probabilities['white_balance_probability'],
        'color_matrix': torch.rand(()).item() < probabilities['color_matrix_probability'],
    }

    if flags['random_srf']:
        bands = int(opt.get('random_srf_bands', gt.shape[0]))
        rgb = _make_rgb_from_srf(gt, _dynamic_random_row_normalized_srf(bands))
    else:
        rgb = clean_rgb.clone()

    if flags['exposure']:
        factor = _sample_log_uniform(
            opt.get('exposure_min_factor', 0.5),
            opt.get('exposure_max_factor', 2.0),
        )
        rgb = rgb * factor.to(dtype=rgb.dtype)

    if flags['white_balance']:
        min_gain = float(opt.get('white_balance_min_gain', 0.7))
        max_gain = float(opt.get('white_balance_max_gain', 1.4))
        gains = torch.stack([
            _sample_log_uniform(min_gain, max_gain) for _ in range(rgb.shape[0])
        ]).to(dtype=rgb.dtype)
        if bool(opt.get('white_balance_normalize_geometric_mean', True)):
            gains = gains / gains.log().mean().exp().clamp_min(1e-8)
        rgb = rgb * gains.view(-1, 1, 1)

    if flags['color_matrix']:
        max_strength = float(opt.get('color_matrix_max_strength', 0.5))
        if not 0.0 <= max_strength <= 1.0:
            raise ValueError('color_matrix_max_strength must be in [0, 1].')
        random_matrix = torch.rand((3, 3), dtype=rgb.dtype)
        random_matrix = random_matrix / random_matrix.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        strength = torch.rand((), dtype=rgb.dtype) * max_strength
        matrix = (
            (1.0 - strength) * torch.eye(3, dtype=rgb.dtype)
            + strength * random_matrix
        )
        rgb = torch.matmul(matrix, rgb.reshape(3, -1)).reshape_as(rgb)

    rgb = rgb.clamp(0.0, 1.0)
    if return_flags:
        return rgb, flags
    return rgb


def _build_srf(opt):
    srf_mode = opt.get('srf_mode', 'file')
    if srf_mode in ('file', 'fixed', 'mixed_fixed_random_per_sample_row_normalized'):
        srf_path = opt.get('srf_path', 'SRF/P_N_V2.mat')
        srf_data = sio.loadmat(srf_path)
        return torch.from_numpy(srf_data['P_20N']).float()

    if srf_mode == 'random_row_normalized':
        bands = int(opt.get('random_srf_bands', opt.get('band_dim', 31)))
        seed = int(opt.get('random_srf_seed', 0))
        return _random_row_normalized_srf(bands, seed)

    if srf_mode in ('random_per_sample_row_normalized', 'random_per_fetch_row_normalized'):
        return None

    raise ValueError(
        f"Unsupported srf_mode: {srf_mode}. "
        "Use 'file', 'random_row_normalized', 'random_per_sample_row_normalized', "
        "'random_per_fetch_row_normalized', or "
        "'mixed_fixed_random_per_sample_row_normalized'."
    )


def _make_rgb_from_srf(gt, srf):
    gt1 = gt.reshape(gt.size()[0], -1)
    return torch.matmul(srf, gt1).reshape(srf.size()[0], gt.size()[1], gt.size()[2])

def _sample_srf(srf, srf_mode, bands, seed, index, random_prob=1.0):
    if srf_mode == 'random_per_sample_row_normalized':
        return _random_row_normalized_srf(bands, seed + int(index))
    if srf_mode == 'random_per_fetch_row_normalized':
        return _dynamic_random_row_normalized_srf(bands)
    if srf_mode == 'mixed_fixed_random_per_sample_row_normalized':
        if random.random() < float(random_prob):
            return _random_row_normalized_srf(bands, seed + int(index))
        return srf
    return srf


class DataloaderSimpleTrain(Dataset):

    def __init__(self, opt):
        self.paths = opt['paths']
        self.data = []
        self.sf = opt['sf']
        self.gt_size = opt.get('gt_size', 256)
        self.srf_mode = opt.get('srf_mode', 'file')
        self.random_srf_seed = int(opt.get('random_srf_seed', 0))
        self.random_srf_bands = int(opt.get('random_srf_bands', opt.get('band_dim', 31)))
        self.random_srf_prob = float(opt.get('random_srf_prob', 1.0))
        self.return_control_rgb = bool(opt.get('return_control_rgb', False))
        self.control_srf_mode = opt.get('control_srf_mode', self.srf_mode)
        self.control_random_srf_seed = int(opt.get('control_random_srf_seed', self.random_srf_seed))
        self.control_random_srf_bands = int(opt.get('control_random_srf_bands', self.random_srf_bands))
        self.control_random_srf_prob = float(opt.get('control_random_srf_prob', self.random_srf_prob))
        self.return_clean_rgb_target = bool(opt.get('return_clean_rgb_target', False))
        self.return_augmentation_label = bool(
            opt.get('return_augmentation_label', False)
        )
        self.rgb_canonical_augmentation = dict(
            opt.get('rgb_canonical_augmentation', {})
        )
        if self.return_augmentation_label and not self.return_clean_rgb_target:
            raise ValueError(
                'return_augmentation_label requires return_clean_rgb_target=True.'
            )
        if self.return_clean_rgb_target and self.return_control_rgb:
            raise ValueError(
                'return_clean_rgb_target and return_control_rgb cannot both be enabled.'
            )

        # 固定 SRF 缓存；per-sample random SRF 在 __getitem__ 按 index 生成
        self.srf = _build_srf(opt)
        control_opt = dict(opt)
        control_opt['srf_mode'] = self.control_srf_mode
        control_opt['srf_path'] = opt.get('control_srf_path', opt.get('srf_path', 'SRF/P_N_V2.mat'))
        control_opt['random_srf_seed'] = self.control_random_srf_seed
        control_opt['random_srf_bands'] = self.control_random_srf_bands
        control_opt['random_srf_prob'] = self.control_random_srf_prob
        self.control_srf = _build_srf(control_opt) if self.return_control_rgb else None

        # 预加载所有数据到内存，消除磁盘 I/O 瓶颈
        data_paths = []
        for path in self.paths:
            data_paths.extend(glob.glob(path + '/*.mat'))
        for p in data_paths:
            mat = sio.loadmat(p)
            self.data.append(torch.from_numpy(mat['ref']).float().permute(2, 0, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]  # (C, H, W), 已经是 float32
        _, H, W = img.shape

        # 随机裁剪：在图像内随机选取一个 gt_size × gt_size 的区域
        max_i = H - self.gt_size
        max_j = W - self.gt_size
        i = random.randint(0, max_i) if max_i > 0 else 0
        j = random.randint(0, max_j) if max_j > 0 else 0
        gt = img[:, i:i+self.gt_size, j:j+self.gt_size].clone()

        gt_min = torch.min(gt)
        gt_max = torch.max(gt)
        gt = (gt - gt_min) / (gt_max - gt_min + 1e-8)

        # SRF: (3, C) @ (C, H*W) -> (3, H, W)
        srf = _sample_srf(self.srf, self.srf_mode, self.random_srf_bands, self.random_srf_seed, index, self.random_srf_prob)
        rgb = _make_rgb_from_srf(gt, srf)
        clean_rgb_target = None
        augmentation_label = None
        if self.return_clean_rgb_target:
            clean_rgb_target = rgb.clone()
            if self.return_augmentation_label:
                rgb, augmentation_flags = _augment_rgb_for_canonicalization(
                    gt, clean_rgb_target, self.rgb_canonical_augmentation,
                    return_flags=True,
                )
                augmentation_label = torch.tensor(
                    float(any(augmentation_flags.values())), dtype=torch.float32
                )
            else:
                rgb = _augment_rgb_for_canonicalization(
                    gt, clean_rgb_target, self.rgb_canonical_augmentation
                )
        control_rgb = None
        if self.return_control_rgb:
            control_srf = _sample_srf(
                self.control_srf, self.control_srf_mode,
                self.control_random_srf_bands, self.control_random_srf_seed, index, self.control_random_srf_prob,
            )
            control_rgb = _make_rgb_from_srf(gt, control_srf)

        # gaussian_blur + 8x 下采样（用 F.interpolate 替代 core.imresize，~220x 加速）
        lq = _gaussian_blur_torch(gt, sigma=2)
        lq = F.interpolate(lq.unsqueeze(0), scale_factor=1.0/self.sf,
                           mode='bicubic', align_corners=False).squeeze(0)

        if self.return_clean_rgb_target:
            if self.return_augmentation_label:
                return gt, lq, rgb, clean_rgb_target, augmentation_label
            return gt, lq, rgb, clean_rgb_target
        if self.return_control_rgb:
            return gt, lq, rgb, control_rgb
        return gt, lq, rgb


class DataloaderSimpleTest(Dataset):
    def __init__(self, opt):
        self.paths = opt['paths']
        self.data = []
        self.sf = opt['sf']
        self.gt_size = opt.get('gt_size', 256)
        self.crop_pos = opt.get('crop_pos', [0, 0])
        self.use_full_img = opt.get('use_full_img', False)
        self.srf_mode = opt.get('srf_mode', 'file')
        self.random_srf_seed = int(opt.get('random_srf_seed', 0))
        self.random_srf_bands = int(opt.get('random_srf_bands', opt.get('band_dim', 31)))
        self.random_srf_prob = float(opt.get('random_srf_prob', 1.0))
        self.return_control_rgb = bool(opt.get('return_control_rgb', False))
        self.control_srf_mode = opt.get('control_srf_mode', self.srf_mode)
        self.control_random_srf_seed = int(opt.get('control_random_srf_seed', self.random_srf_seed))
        self.control_random_srf_bands = int(opt.get('control_random_srf_bands', self.random_srf_bands))
        self.control_random_srf_prob = float(opt.get('control_random_srf_prob', self.random_srf_prob))

        # 固定 SRF 缓存；per-sample random SRF 在 __getitem__ 按 index 生成
        self.srf = _build_srf(opt)
        control_opt = dict(opt)
        control_opt['srf_mode'] = self.control_srf_mode
        control_opt['srf_path'] = opt.get('control_srf_path', opt.get('srf_path', 'SRF/P_N_V2.mat'))
        control_opt['random_srf_seed'] = self.control_random_srf_seed
        control_opt['random_srf_bands'] = self.control_random_srf_bands
        control_opt['random_srf_prob'] = self.control_random_srf_prob
        self.control_srf = _build_srf(control_opt) if self.return_control_rgb else None

        # 预加载所有数据到内存
        data_paths = []
        for path in self.paths:
            data_paths.extend(sorted(glob.glob(path + '/*.mat')))
        for p in data_paths:
            mat = sio.loadmat(p)
            self.data.append(torch.from_numpy(mat['ref']).float().permute(2, 0, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        if self.use_full_img:
            # 整张图片测试，不做裁剪
            gt = img.clone()
        else:
            # 中心裁剪（验证时使用，保持训练效率）
            i, j = self.crop_pos[0], self.crop_pos[1]
            gt = img[:, i:i+self.gt_size, j:j+self.gt_size].clone()

        gt_min = torch.min(gt)
        gt_max = torch.max(gt)
        gt = (gt - gt_min) / (gt_max - gt_min + 1e-8)

        srf = _sample_srf(self.srf, self.srf_mode, self.random_srf_bands, self.random_srf_seed, index, self.random_srf_prob)
        rgb = _make_rgb_from_srf(gt, srf)
        control_rgb = None
        if self.return_control_rgb:
            control_srf = _sample_srf(
                self.control_srf, self.control_srf_mode,
                self.control_random_srf_bands, self.control_random_srf_seed, index, self.control_random_srf_prob,
            )
            control_rgb = _make_rgb_from_srf(gt, control_srf)

        lq = _gaussian_blur_torch(gt, sigma=2)
        lq = F.interpolate(lq.unsqueeze(0), scale_factor=1.0/self.sf,
                           mode='bicubic', align_corners=False).squeeze(0)

        if self.return_control_rgb:
            return gt, lq, rgb, control_rgb
        return gt, lq, rgb


# ---- 预计算高斯核缓存，避免每次重建 ----
_gaussian_kernel_cache = {}

def _gaussian_blur_torch(tensor, sigma, kernel_size=7):
    """用 PyTorch conv2d 批量高斯模糊，替代逐 band 的 numpy cv2 调用"""
    key = (sigma, kernel_size, tensor.shape[0])
    if key not in _gaussian_kernel_cache:
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).reshape(1, 1, kernel_size, kernel_size)
        _gaussian_kernel_cache[key] = kernel
    kernel = _gaussian_kernel_cache[key].to(
        device=tensor.device, dtype=tensor.dtype
    )

    C = tensor.shape[0]
    pad = kernel_size // 2
    x = tensor.unsqueeze(0)  # (1, C, H, W)
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x, kernel.expand(C, -1, -1, -1), groups=C).squeeze(0)
