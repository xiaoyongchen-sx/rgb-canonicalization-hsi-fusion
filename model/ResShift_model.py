import os
import logging
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


def safe_sam(pred, gt, metric):
    """计算 SAM，排除全零像素（黑边）"""
    # pred/gt shape: [B, C, H, W]
    # 找出 GT 和 pred 都非全零的有效像素
    gt_norm = torch.norm(gt, dim=1, keepdim=True)    # [B, 1, H, W]
    pred_norm = torch.norm(pred, dim=1, keepdim=True)
    mask = (gt_norm.squeeze(1) > 1e-8) & (pred_norm.squeeze(1) > 1e-8)  # [B, H, W]

    if mask.sum() == 0:
        return 0.0

    # 只在有效像素上计算 SAM
    B, C, H, W = pred.shape
    pred_flat = pred.permute(0, 2, 3, 1)[mask]  # [N, C]
    gt_flat = gt.permute(0, 2, 3, 1)[mask]      # [N, C]

    dot = (pred_flat * gt_flat).sum(dim=1)
    norm_p = torch.norm(pred_flat, dim=1)
    norm_g = torch.norm(gt_flat, dim=1)
    cos_angle = torch.clamp(dot / (norm_p * norm_g + 1e-10), -1.0, 1.0)
    sam_per_pixel = torch.acos(cos_angle)  # [N]

    return sam_per_pixel.mean().item()
import math
import cv2
import torch.nn.functional as F
from arch.BAFUnet import  BAFUNet
from dataset_loader.dataloader import (
    DataloaderSimpleTest, DataloaderSimpleTrain, _gaussian_blur_torch,
)
from torch.utils.data import DataLoader
from torchmetrics.image import *
import scipy.io as sio
from tqdm import tqdm
from EMRDiff import *
import time
from thop import profile

def sparse_checkerboard(img,size):
    b,c,h,w = img.shape
    img_array = np.array(img.detach().cpu().numpy())
    result_array = np.zeros_like(img_array)
    height, width = h,w
    for i in range(0, height, size):
        for j in range(0, width, size):
                result_array[:, :,i, j] = img_array[:, :,i, j]
    return torch.from_numpy(result_array)

def pdown(img,size):
    b,c,h,w = img.shape
    img_array = np.array(img.detach().cpu().numpy())
    height, width = h,w
    new_height = height // size
    new_width = width // size
    compact_array = np.zeros((b,c,new_height, new_width), dtype=img_array.dtype)
    for i in range(0, height, size):
        for j in range(0, width, size):
            if i // size < new_height and j // size < new_width:
                compact_array[:, :, i // size, j // size] = img_array[:,:,i, j]
    return torch.from_numpy(compact_array)

def gradient_loss(pred, target):
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def save_checkpoint(model, optimizer, scheduler, epoch, output_dir, best_metric=None, ema_model=None):
    """保存 checkpoint，只保留 best 和 latest"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()
    # 始终覆盖 latest
    torch.save(state, os.path.join(checkpoint_dir, "latest.pth.tar"))
    return checkpoint_dir

def save_best_checkpoint(model, optimizer, scheduler, epoch, output_dir, best_metric, ema_model=None):
    """保存最佳 checkpoint"""
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()
    torch.save(state, os.path.join(checkpoint_dir, "best.pth.tar"))
        #print("Checkpoints saved to {}".format(model_out_path))
class ResShiftTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device("cuda:0")
        self.epochs = self.configs.train['epochs']
        self.num_timesteps = self.configs.diffusion.params.get("steps")
        self.diffusion_sf = self.configs.diffusion.params.get("sf")
        self.diffusion_scale_factor = self.configs.diffusion.params.get("scale_factor")
        # 是否使用 pseudo-MSI 作为额外输出通道（默认 True 保持向后兼容）
        self.use_pseudo_msi = self.configs.model.params.get('use_pseudo_msi', True)
        # 是否在扩散噪声中使用 RGB Sobel edge（默认 True 保持向后兼容）
        self.use_rgb_edge = self.configs.model.params.get('use_rgb_edge', True)
        # 可选 HR 梯度/边缘损失；默认 0 保持旧实验完全不变
        self.gradient_loss_weight = float(self.configs.train.get('gradient_loss_weight', 0.0))
        self.use_rgb_canonicalizer = bool(
            self.configs.model.params.get('use_rgb_canonicalizer', False)
        )
        self.use_nikon_rgb_restorer = bool(
            self.configs.model.params.get('use_nikon_rgb_restorer', False)
        )
        self.use_rgb_restoration = (
            self.use_rgb_canonicalizer or self.use_nikon_rgb_restorer
        )
        self.nikon_restorer_hr_hsi_only = bool(
            self.configs.train.get('nikon_restorer_hr_hsi_only', True)
        )
        self.nikon_restorer_target_srf_path = self.configs.train.get(
            'nikon_restorer_target_srf_path',
            self.configs.data.get('srf_path', None),
        )
        self.rgb_canonical_loss_weight = float(
            self.configs.train.get('rgb_canonical_loss_weight', 0.0)
        )
        self.rgb_identity_loss_weight = float(
            self.configs.train.get('rgb_identity_loss_weight', 0.0)
        )
        self.canonical_gate_loss_weight = float(
            self.configs.train.get('canonical_gate_loss_weight', 0.0)
        )
        self.hsi_reconstruction_loss_weight = float(
            self.configs.train.get('hsi_reconstruction_loss_weight', 1.0)
        )
        self.validation_seed = self.configs.train.get('validation_seed', None)
        self.return_clean_rgb_target = bool(
            self.configs.data.train.params.get('return_clean_rgb_target', False)
        )
        self.return_augmentation_label = bool(
            self.configs.data.train.params.get('return_augmentation_label', False)
        )
        if self.canonical_gate_loss_weight > 0 and not self.return_augmentation_label:
            raise ValueError(
                'canonical_gate_loss_weight > 0 requires return_augmentation_label=True.'
            )
        if (
                self.rgb_canonical_loss_weight > 0
                or self.rgb_identity_loss_weight > 0
        ) and not self.use_rgb_restoration:
            raise ValueError(
                'RGB restoration losses require an RGB restoration module.'
            )
        if self.hsi_reconstruction_loss_weight < 0:
            raise ValueError('hsi_reconstruction_loss_weight must be non-negative.')
        if self.use_rgb_restoration and not self.return_clean_rgb_target:
            raise ValueError(
                'RGB restoration requires train.return_clean_rgb_target=True.'
            )
        if self.use_nikon_rgb_restorer and not self.nikon_restorer_target_srf_path:
            raise ValueError(
                'Nikon RGB restorer requires train.nikon_restorer_target_srf_path.'
            )
        self.random_rgb_consistency = bool(self.configs.train.get('random_rgb_consistency', False))
        self.random_rgb_weight = float(self.configs.train.get('random_rgb_weight', 1.0))
        self.random_rgb_multiscale = bool(
            self.configs.train.get('random_rgb_multiscale', True)
        )
        self.consistency_weight = float(self.configs.train.get('consistency_weight', 0.0))
        self.factorized_spatial_consistency_weight = float(
            self.configs.train.get('factorized_spatial_consistency_weight', 0.0)
        )
        self.rgb_feature_consistency = bool(
            self.configs.train.get('rgb_feature_consistency', False)
        )
        self.rgb_feature_consistency_weight = float(
            self.configs.train.get('rgb_feature_consistency_weight', 0.0)
        )
        self.rgb_feature_consistency_layers = list(
            self.configs.train.get('rgb_feature_consistency_layers', [])
        )
        self.random_rgb_curriculum = bool(
            self.configs.train.get('random_rgb_curriculum', False)
        )
        self.random_rgb_weight_initial = float(
            self.configs.train.get('random_rgb_weight_initial', self.random_rgb_weight)
        )
        self.random_rgb_weight_final = float(
            self.configs.train.get('random_rgb_weight_final', self.random_rgb_weight)
        )
        self.random_rgb_ramp_start_epoch = int(
            self.configs.train.get('random_rgb_ramp_start_epoch', 0)
        )
        self.random_rgb_ramp_end_epoch = int(
            self.configs.train.get('random_rgb_ramp_end_epoch', 0)
        )
        self.use_ema_detail_teacher = bool(
            self.configs.train.get('use_ema_detail_teacher', False)
        )
        self.ema_decay = float(self.configs.train.get('ema_decay', 0.999))
        self.ema_start_epoch = int(self.configs.train.get('ema_start_epoch', 0))
        self.detail_consistency_weight_initial = float(
            self.configs.train.get('detail_consistency_weight_initial', 0.0)
        )
        self.detail_consistency_weight_final = float(
            self.configs.train.get('detail_consistency_weight_final', 0.0)
        )
        self.detail_ramp_start_epoch = int(
            self.configs.train.get('detail_ramp_start_epoch', 0)
        )
        self.detail_ramp_end_epoch = int(
            self.configs.train.get('detail_ramp_end_epoch', 0)
        )
        self.detail_highpass_kernel = int(
            self.configs.train.get('detail_highpass_kernel', 5)
        )
        if self.detail_highpass_kernel < 3 or self.detail_highpass_kernel % 2 == 0:
            raise ValueError('detail_highpass_kernel must be an odd integer >= 3.')
        self.ema_teacher = None
        self._pending_ema_state = None
        # 默认不保存逐样本 .mat，避免训练/测试输出目录快速膨胀
        self.save_val_mat = bool(self.configs.train.get('save_val_mat', False))
        self.save_test_mat = bool(self.configs.get('test', {}).get('save_mat', False))
        self.test_use_amp = bool(self.configs.get('test', {}).get('use_amp', False))
        self.test_final_state = str(
            self.configs.get('test', {}).get('final_state', 'x_pred')
        )
        if self.test_final_state not in ('x_pred', 'posterior'):
            raise ValueError(
                "test.final_state must be either 'x_pred' or 'posterior'."
            )
        self.test_hsi_backprojection_steps = int(
            self.configs.get('test', {}).get('hsi_backprojection_steps', 0)
        )
        self.test_hsi_backprojection_strength = float(
            self.configs.get('test', {}).get(
                'hsi_backprojection_strength', 1.0
            )
        )
        if self.test_hsi_backprojection_steps < 0:
            raise ValueError('test.hsi_backprojection_steps must be non-negative.')
        if self.test_hsi_backprojection_strength < 0:
            raise ValueError(
                'test.hsi_backprojection_strength must be non-negative.'
            )
        self.canonicalizer_test_gate_mode = str(
            self.configs.get('test', {}).get('canonicalizer_gate_mode', 'soft')
        )
        self.canonicalizer_gate_threshold = float(
            self.configs.get('test', {}).get('canonicalizer_gate_threshold', 0.5)
        )
        self.base_model_lr_scale = float(
            self.configs.train.get('base_model_lr_scale', 1.0)
        )
        # ControlNet-style 训练：加载无 control 主干后，只训练 control 分支
        self.freeze_base_model = bool(self.configs.train.get('freeze_base_model', False))

        # 创建带时间戳的输出目录
        output_base = self.configs.data.get('output_dir', 'output')
        dataset_name = self.configs.data.get('dataset_name', '')
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.output_dir = os.path.join(output_base, dataset_name, timestamp) if dataset_name else os.path.join(output_base, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化 logger
        self.logger = logging.getLogger('EMRDiff')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(self.output_dir, 'train.log'))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 记录配置信息
        self.logger.info(f'Output dir: {self.output_dir}')
        self.logger.info(f'Config: {OmegaConf.to_yaml(self.configs)}')

        # 初始化 TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))

        self.test_only = bool(
            self.configs.get('test', {}).get('test_only', False)
        )
        if self.test_only:
            self.train_dataloader = None
            self.val_dataloader = None
            self.logger.info('Test-only initialization: skipped training/validation dataloaders')
        else:
            self.train_dataloader = self.build_training_dataloader()
            self.val_dataloader = self.build_val_dataloader()
        self.build_model()
        self.build_diffusion_model()

        # 加载预训练权重（只加载模型参数，不加载优化器和 epoch）
        pretrained_path = self.configs.train.get('pretrained', None)
        if pretrained_path:
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f'Pretrained checkpoint not found: {pretrained_path}')
            self.load_pretrained(pretrained_path)

        if self.freeze_base_model:
            self.freeze_base_model_params()

        if self.test_only:
            self.optimizer = None
            self.scheduler = None
        else:
            self.setup_optimization()

        # 最佳指标跟踪 (PSNR 越大越好)
        self.best_psnr = 0.0
        self.start_epoch = 0

        # resume
        resume_path = self.configs.train.get('resume', None)
        if resume_path:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
            self.resume(resume_path)

        if self.use_ema_detail_teacher:
            self.ema_teacher = deepcopy(self.Net).to(self.device).eval()
            for parameter in self.ema_teacher.parameters():
                parameter.requires_grad_(False)
            if self._pending_ema_state is not None:
                self.ema_teacher.load_state_dict(self._pending_ema_state)
                self.logger.info('Restored EMA teacher from checkpoint')
            else:
                self.logger.info('Initialized EMA teacher from student weights')

    def setup_optimization(self):
        lr = self.configs.train.get('lr')
        optimizer_type = self.configs.train.get('optimizer', 'adam').lower()
        weight_decay = self.configs.train.get('weight_decay', 0)

        trainable_params = [p for p in self.Net.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError('No trainable parameters found. Check freeze_base_model/control modules.')

        optimizer_params = trainable_params
        if (
                self.use_rgb_canonicalizer
                and not self.freeze_base_model
                and self.base_model_lr_scale != 1.0
        ):
            canonicalizer_params = []
            base_params = []
            for name, parameter in self.Net.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith('rgb_canonicalizer'):
                    canonicalizer_params.append(parameter)
                else:
                    base_params.append(parameter)
            optimizer_params = [
                {'params': base_params, 'lr': lr * self.base_model_lr_scale},
                {'params': canonicalizer_params, 'lr': lr},
            ]
            self.logger.info(
                f'Differential LR: base={lr * self.base_model_lr_scale:.2e}, '
                f'canonicalizer={lr:.2e}'
            )

        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(optimizer_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Choose 'adam' or 'adamw'.")

        total_params = sum(p.numel() for p in self.Net.parameters())
        trainable_count = sum(p.numel() for p in self.Net.parameters() if p.requires_grad)
        self.logger.info(
            f'Trainable params: {trainable_count}/{total_params} '
            f'({trainable_count / max(total_params, 1) * 100:.2f}%)'
        )

        # ---- LR Scheduler（统一通过 lr_scheduler 配置项控制） ----
        self.warmup_epochs = self.configs.train.get('warmup_epochs', 0)
        warmup_start_factor = self.configs.train.get('warmup_start_factor', 0.1)
        total_epochs = self.configs.train.get('epochs')
        scheduler_name = self.configs.train.get('lr_scheduler', 'none').lower()
        scheduler_params = dict(self.configs.train.get('lr_scheduler_params', {}))

        # 1) 创建主 LR scheduler
        if scheduler_name == 'none':
            main_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda epoch: 1.0
            )
        elif scheduler_name == 'cosine':
            eta_min = scheduler_params.get('eta_min', lr * 0.01)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - max(0, self.warmup_epochs),
                eta_min=eta_min,
            )
        else:
            raise ValueError(
                f"Unsupported lr_scheduler: {scheduler_name}. "
                f"Choose 'none' or 'cosine'."
            )

        # 2) 叠加 warmup（plateau 不支持 SequentialLR 包装，单独处理）
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
            self.logger.info(
                f'LR scheduler: {scheduler_name} with warmup '
                f'(warmup_epochs={self.warmup_epochs}, '
                f'start_factor={warmup_start_factor})'
            )
        else:
            self.scheduler = main_scheduler
            self.logger.info(f'LR scheduler: {scheduler_name}')

    def resume(self, checkpoint_path):
        """从 checkpoint 恢复训练"""
        self.logger.info(f'Resuming from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.Net.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint.get('best_metric', 0.0) or 0.0
        self._pending_ema_state = checkpoint.get('ema_model')
        self.logger.info(f'Resumed at epoch {self.start_epoch}, best_psnr={self.best_psnr:.4f}')

    def load_pretrained(self, checkpoint_path):
        """加载预训练权重（仅模型参数）"""
        self.logger.info(f'Loading pretrained weights from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        pretrained_state = dict(checkpoint['model'])
        model_state = self.Net.state_dict()
        adapted_keys = []
        skipped_keys = []
        condition_input_key = 'input_blocks_lqrgb.0.0.weight'
        for key, value in list(pretrained_state.items()):
            if key not in model_state or value.shape == model_state[key].shape:
                continue
            target = model_state[key]
            if (
                    key == condition_input_key
                    and value.ndim == 4
                    and target.ndim == 4
                    and value.shape[0] == target.shape[0]
                    and value.shape[2:] == target.shape[2:]
                    and value.shape[1] >= target.shape[1]
            ):
                # Baseline condition input is [LR-HSI, RGB]. E19 keeps only
                # the leading LR-HSI channels in the main condition encoder.
                pretrained_state[key] = value[:, :target.shape[1]].clone()
                adapted_keys.append(key)
            else:
                pretrained_state.pop(key)
                skipped_keys.append(key)
        result = self.Net.load_state_dict(pretrained_state, strict=False)
        self.logger.info(
            f'Pretrained weights loaded | missing={len(result.missing_keys)}, '
            f'unexpected={len(result.unexpected_keys)}, adapted={len(adapted_keys)}, '
            f'shape_skipped={len(skipped_keys)}'
        )
        if adapted_keys:
            self.logger.info(f'Adapted pretrained tensors: {adapted_keys}')
        if skipped_keys:
            self.logger.info(f'Skipped shape-mismatched tensors: {skipped_keys}')

    def freeze_base_model_params(self):
        trainable_prefixes = (
            'control_adapter',
            'control_residual_blocks',
            'control_mid_residual_block',
            'edge_control_blocks',
            'rgb_canonicalizer',
            'nikon_rgb_restorer',
        )
        for _, param in self.Net.named_parameters():
            param.requires_grad = False
        trainable_names = []
        for name, param in self.Net.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad = True
                trainable_names.append(name)
        if not trainable_names:
            raise ValueError(
                'freeze_base_model=True but no auxiliary parameters were found. '
                'Check control, edge, or RGB canonicalizer modules.'
            )
        self.logger.info(
            'freeze_base_model=True: frozen base model; training auxiliary modules only '
            f'({len(trainable_names)} tensors).'
        )

    def build_model(self):
        params = dict(self.configs.model.get('params', dict))
        # 以下是训练/扩散控制超参，不传给模型架构
        params.pop('use_pseudo_msi', None)
        params.pop('use_rgb_edge', None)
        self.Net = BAFUNet(**params)
        self.Net = self.Net.to(self.device)
        if self.use_nikon_rgb_restorer:
            srf_data = sio.loadmat(self.nikon_restorer_target_srf_path)
            if 'P_20N' not in srf_data:
                raise KeyError(
                    f'{self.nikon_restorer_target_srf_path} does not contain P_20N.'
                )
            self.Net.nikon_rgb_restorer.set_target_srf(
                torch.from_numpy(srf_data['P_20N']).float().to(self.device)
            )
            self.logger.info(
                f'Loaded Nikon restorer target SRF from '
                f'{self.nikon_restorer_target_srf_path}'
            )
    def build_diffusion_model(self):
        diffusion_opt = self.configs.get('diffusion', dict)
        self.EMRDIFF = EMRDIFF(diffusion_opt)
    def build_training_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.train.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.train.params.get('gt_size')
        opt['srf_path'] = self.configs.data.train.params.get('srf_path', self.configs.data.get('srf_path', 'SRF/P_N_V2.mat'))
        opt['srf_mode'] = self.configs.data.train.params.get('srf_mode', self.configs.data.get('srf_mode', 'file'))
        opt['random_srf_seed'] = self.configs.data.train.params.get('random_srf_seed', self.configs.data.get('random_srf_seed', 0))
        opt['random_srf_bands'] = self.configs.data.train.params.get(
            'random_srf_bands', self.configs.data.get('random_srf_bands', self.configs.diffusion.params.get('band_dim', 31))
        )
        opt['random_srf_prob'] = self.configs.data.train.params.get('random_srf_prob', self.configs.data.get('random_srf_prob', 1.0))
        opt['return_control_rgb'] = self.configs.data.train.params.get('return_control_rgb', self.configs.data.get('return_control_rgb', False))
        opt['control_srf_mode'] = self.configs.data.train.params.get('control_srf_mode', self.configs.data.get('control_srf_mode', opt['srf_mode']))
        opt['control_srf_path'] = self.configs.data.train.params.get('control_srf_path', self.configs.data.get('control_srf_path', opt['srf_path']))
        opt['control_random_srf_seed'] = self.configs.data.train.params.get('control_random_srf_seed', self.configs.data.get('control_random_srf_seed', opt['random_srf_seed']))
        opt['control_random_srf_bands'] = self.configs.data.train.params.get(
            'control_random_srf_bands', self.configs.data.get('control_random_srf_bands', opt['random_srf_bands'])
        )
        opt['control_random_srf_prob'] = self.configs.data.train.params.get('control_random_srf_prob', self.configs.data.get('control_random_srf_prob', opt['random_srf_prob']))
        opt['return_clean_rgb_target'] = self.configs.data.train.params.get(
            'return_clean_rgb_target', False
        )
        opt['return_augmentation_label'] = self.configs.data.train.params.get(
            'return_augmentation_label', False
        )
        opt['rgb_canonical_augmentation'] = self.configs.data.train.params.get(
            'rgb_canonical_augmentation', {}
        )
        opt['crop_pos'] = self.configs.data.train.params.get('crop_pos', [500, 500])
        batch_size = self.configs.train.get('batch')[0]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTrain(opt), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)
    def build_val_dataloader(self):
        opt = {}
        opt['paths'] = self.configs.data.val.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.val.params.get('gt_size')
        opt['srf_path'] = self.configs.data.val.params.get('srf_path', self.configs.data.get('srf_path', 'SRF/P_N_V2.mat'))
        opt['srf_mode'] = self.configs.data.val.params.get('srf_mode', self.configs.data.get('srf_mode', 'file'))
        opt['random_srf_seed'] = self.configs.data.val.params.get('random_srf_seed', self.configs.data.get('random_srf_seed', 0))
        opt['random_srf_bands'] = self.configs.data.val.params.get(
            'random_srf_bands', self.configs.data.get('random_srf_bands', self.configs.diffusion.params.get('band_dim', 31))
        )
        opt['random_srf_prob'] = self.configs.data.val.params.get('random_srf_prob', self.configs.data.get('random_srf_prob', 1.0))
        opt['return_control_rgb'] = self.configs.data.val.params.get('return_control_rgb', self.configs.data.get('return_control_rgb', False))
        opt['control_srf_mode'] = self.configs.data.val.params.get('control_srf_mode', self.configs.data.get('control_srf_mode', opt['srf_mode']))
        opt['control_srf_path'] = self.configs.data.val.params.get('control_srf_path', self.configs.data.get('control_srf_path', opt['srf_path']))
        opt['control_random_srf_seed'] = self.configs.data.val.params.get('control_random_srf_seed', self.configs.data.get('control_random_srf_seed', opt['random_srf_seed']))
        opt['control_random_srf_bands'] = self.configs.data.val.params.get(
            'control_random_srf_bands', self.configs.data.get('control_random_srf_bands', opt['random_srf_bands'])
        )
        opt['control_random_srf_prob'] = self.configs.data.val.params.get('control_random_srf_prob', self.configs.data.get('control_random_srf_prob', opt['random_srf_prob']))
        opt['use_full_img'] = True  # 全图验证
        batch_size = self.configs.train.get('batch')[1]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTest(opt), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)


    def unpack_rgb_batch(self, batch):
        control_rgb = None
        canonical_rgb_target = None
        augmentation_label = None
        if len(batch) == 5:
            canonical_rgb_target = batch[3]
            augmentation_label = batch[4]
        elif len(batch) == 4:
            if self.return_clean_rgb_target:
                canonical_rgb_target = batch[3]
            else:
                control_rgb = batch[3]
        return (
            batch[0], batch[1], batch[2], control_rgb,
            canonical_rgb_target, augmentation_label,
        )

    @staticmethod
    def _linear_ramp(epoch, start_epoch, end_epoch, initial, final):
        if end_epoch <= start_epoch:
            return float(final)
        progress = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)
        return float(initial + progress * (final - initial))

    def _current_random_rgb_weight(self, epoch):
        if not self.random_rgb_curriculum:
            return self.random_rgb_weight
        return self._linear_ramp(
            epoch,
            self.random_rgb_ramp_start_epoch,
            self.random_rgb_ramp_end_epoch,
            self.random_rgb_weight_initial,
            self.random_rgb_weight_final,
        )

    def _current_detail_weight(self, epoch):
        if not self.use_ema_detail_teacher:
            return 0.0
        return self._linear_ramp(
            epoch,
            self.detail_ramp_start_epoch,
            self.detail_ramp_end_epoch,
            self.detail_consistency_weight_initial,
            self.detail_consistency_weight_final,
        )

    def _high_pass(self, image):
        padding = self.detail_highpass_kernel // 2
        padded = F.pad(image, (padding, padding, padding, padding), mode='reflect')
        low = F.avg_pool2d(padded, kernel_size=self.detail_highpass_kernel, stride=1)
        return image - low

    @torch.no_grad()
    def _update_ema_teacher(self):
        if self.ema_teacher is None:
            return
        for teacher_parameter, student_parameter in zip(
                self.ema_teacher.parameters(), self.Net.parameters()
        ):
            teacher_parameter.mul_(self.ema_decay).add_(
                student_parameter.detach(), alpha=1.0 - self.ema_decay
            )
        for teacher_buffer, student_buffer in zip(
                self.ema_teacher.buffers(), self.Net.buffers()
        ):
            teacher_buffer.copy_(student_buffer)

    def train(self, epoch, verbose):
        for i in range(self.start_epoch, epoch):
            current_random_rgb_weight = self._current_random_rgb_weight(i)
            current_detail_weight = self._current_detail_weight(i)
            self.logger.info(f'Epoch {i}/{epoch}')

            for step, batch in enumerate(tqdm(self.train_dataloader)):
                gt, lq, rgb, control_rgb, canonical_rgb_target, augmentation_label = self.unpack_rgb_batch(batch)

                lq_up8 = nn.functional.interpolate(lq, scale_factor=8, mode='bicubic', align_corners=False)
                lq_up4 = nn.functional.interpolate(lq, scale_factor=4, mode='bicubic', align_corners=False)
                lq_up2 = nn.functional.interpolate(lq, scale_factor=2, mode='bicubic', align_corners=False)
                lq_up8 = lq_up8.to(self.device).type(torch.float32)
                lq_up4 = lq_up4.to(self.device).type(torch.float32)
                lq_up2 = lq_up2.to(self.device).type(torch.float32)
                lq = lq.to(self.device).type(torch.float32)
                rgb = rgb.to(self.device).type(torch.float32)
                control_rgb = control_rgb.to(self.device).type(torch.float32) if control_rgb is not None else None
                canonical_rgb_target = (
                    canonical_rgb_target.to(self.device).type(torch.float32)
                    if canonical_rgb_target is not None else None
                )
                augmentation_label = (
                    augmentation_label.to(self.device).type(torch.float32)
                    if augmentation_label is not None else None
                )
                gt = gt.to(self.device).type(torch.float32)
                self.optimizer.zero_grad()
                tt = torch.randint(
                    0, self.num_timesteps,
                    size=(lq.shape[0],),
                    device=lq.device,
                )
                loss_func = nn.L1Loss()
                loss_random = torch.tensor(0.0, device=self.device)
                loss_cons = torch.tensor(0.0, device=self.device)
                loss_feature_cons = torch.tensor(0.0, device=self.device)
                loss_spatial_cons = torch.tensor(0.0, device=self.device)
                loss_canonical_rgb = torch.tensor(0.0, device=self.device)
                loss_identity_rgb = torch.tensor(0.0, device=self.device)
                loss_canonical_gate = torch.tensor(0.0, device=self.device)
                augmented_gate_mean = torch.tensor(0.0, device=self.device)
                clean_gate_mean = torch.tensor(0.0, device=self.device)
                loss_detail = torch.tensor(0.0, device=self.device)

                if self.use_pseudo_msi:
                    # ---- 原始模式：HSI + pseudo-MSI 作为输出 ----
                    lqrgb = torch.cat((lq_up8, rgb), dim=1)
                    noise = torch.randn(size=lqrgb.shape, device=lq.device)
                    band = [0, 1, 2]
                    p_MSI = gt[:, band, :, :]
                    x_start = torch.cat((gt, p_MSI), dim=1)
                    x_t = self.EMRDIFF.forward_addnoise(x_start=x_start, y=lqrgb, t=tt, noise=noise,
                                                        rgb_hr=rgb if self.use_rgb_edge else None)
                    network_output, up_out = self.Net(x_t, rgb, lq_up8, tt, control_rgb=control_rgb)

                    rgb_down8 = pdown(rgb, 8).to(self.device)
                    rgb_down4 = pdown(rgb, 4).to(self.device)
                    rgb_down2 = pdown(rgb, 2).to(self.device)
                    x_down8 = pdown(x_start, 8).to(self.device)
                    x_down4 = pdown(x_start, 4).to(self.device)
                    x_down2 = pdown(x_start, 2).to(self.device)
                    pred_hr = network_output + lqrgb
                    loss1 = loss_func(pred_hr, x_start)
                    loss2 = loss_func(up_out[2] + torch.cat((lq, rgb_down8), dim=1), x_down8)
                    loss3 = loss_func(up_out[4] + torch.cat((lq_up2, rgb_down4), dim=1), x_down4)
                    loss4 = loss_func(up_out[6] + torch.cat((lq_up4, rgb_down2), dim=1), x_down2)
                else:
                    # ---- HSI-only 模式：仅输出 HSI，RGB 仅作辅助引导 ----
                    noise = torch.randn(size=gt.shape, device=lq.device)
                    x_start = gt
                    model_rgb = rgb
                    apply_nikon_rgb_restorer = True
                    if self.use_nikon_rgb_restorer:
                        model_rgb, _, _ = self.Net.nikon_rgb_restorer(rgb, lq)
                        apply_nikon_rgb_restorer = False
                    x_t = self.EMRDIFF.forward_addnoise(
                        x_start=x_start,
                        y=lq_up8,
                        t=tt,
                        noise=noise,
                        rgb_hr=model_rgb if self.use_rgb_edge else None,
                    )
                    return_condition_features = (
                        self.rgb_feature_consistency and control_rgb is not None
                    )
                    return_rgb_spatial_feature = (
                        self.factorized_spatial_consistency_weight > 0
                        and control_rgb is not None
                    )
                    return_canonical_rgb = canonical_rgb_target is not None
                    return_canonical_gate = (
                        self.canonical_gate_loss_weight > 0
                        and augmentation_label is not None
                    )
                    network_result = self.Net(
                        x_t,
                        model_rgb,
                        lq_up8,
                        tt,
                        control_rgb=control_rgb,
                        return_condition_features=return_condition_features,
                        return_rgb_spatial_feature=return_rgb_spatial_feature,
                        return_canonical_rgb=return_canonical_rgb,
                        return_canonical_gate=return_canonical_gate,
                        rgb_restorer_lq=lq,
                        apply_nikon_rgb_restorer=apply_nikon_rgb_restorer,
                    )
                    network_output, up_out = network_result[:2]
                    result_index = 2
                    condition_features = None
                    fixed_spatial_feature = None
                    if return_condition_features:
                        condition_features = network_result[result_index]
                        result_index += 1
                    if return_rgb_spatial_feature:
                        fixed_spatial_feature = network_result[result_index]
                        result_index += 1
                    if return_canonical_rgb:
                        canonical_rgb_prediction = network_result[result_index]
                        result_index += 1
                        loss_canonical_rgb = loss_func(
                            canonical_rgb_prediction, canonical_rgb_target
                        )
                    augmented_gate = None
                    if return_canonical_gate:
                        augmented_gate = network_result[result_index]
                        result_index += 1
                        augmented_gate_mean = augmented_gate.mean()
                    if (
                            self.rgb_identity_loss_weight > 0
                            or return_canonical_gate
                    ):
                        (
                            identity_rgb_prediction,
                            _,
                            _,
                            clean_gate,
                        ) = self.Net.rgb_canonicalizer(
                            canonical_rgb_target,
                            lq_up8,
                            return_gate=True,
                        )
                        clean_gate_mean = clean_gate.mean()
                        if self.rgb_identity_loss_weight > 0:
                            loss_identity_rgb = loss_func(
                                identity_rgb_prediction, canonical_rgb_target
                            )
                        if return_canonical_gate:
                            gate_target = augmentation_label.view(-1, 1, 1, 1)
                            loss_canonical_gate = (
                                F.binary_cross_entropy(
                                    augmented_gate, gate_target
                                )
                                + F.binary_cross_entropy(
                                    clean_gate, torch.zeros_like(clean_gate)
                                )
                            )

                    x_down8 = pdown(x_start, 8).to(self.device)
                    x_down4 = pdown(x_start, 4).to(self.device)
                    x_down2 = pdown(x_start, 2).to(self.device)
                    pred_hr = network_output + lq_up8
                    loss1 = loss_func(pred_hr, x_start)
                    loss2 = loss_func(up_out[2] + lq, x_down8)
                    loss3 = loss_func(up_out[4] + lq_up2, x_down4)
                    loss4 = loss_func(up_out[6] + lq_up4, x_down2)

                    if self.random_rgb_consistency and control_rgb is not None:
                        random_result = self.Net(
                            x_t,
                            control_rgb,
                            lq_up8,
                            tt,
                            return_condition_features=return_condition_features,
                            return_rgb_spatial_feature=return_rgb_spatial_feature,
                        )
                        random_condition_features = None
                        random_spatial_feature = None
                        if return_condition_features and return_rgb_spatial_feature:
                            (
                                network_output_random,
                                up_out_random,
                                random_condition_features,
                                random_spatial_feature,
                            ) = random_result
                        elif return_condition_features:
                            network_output_random, up_out_random, random_condition_features = random_result
                        elif return_rgb_spatial_feature:
                            network_output_random, up_out_random, random_spatial_feature = random_result
                        else:
                            network_output_random, up_out_random = random_result
                        pred_hr_random = network_output_random + lq_up8
                        loss_random1 = loss_func(pred_hr_random, x_start)
                        if self.random_rgb_multiscale:
                            loss_random2 = loss_func(up_out_random[2] + lq, x_down8)
                            loss_random3 = loss_func(up_out_random[4] + lq_up2, x_down4)
                            loss_random4 = loss_func(up_out_random[6] + lq_up4, x_down2)
                            loss_random = (
                                loss_random1 + loss_random2 + loss_random3 + loss_random4
                            )
                        else:
                            loss_random = loss_random1
                        if return_rgb_spatial_feature:
                            loss_spatial_cons = loss_func(
                                random_spatial_feature,
                                fixed_spatial_feature.detach(),
                            )
                        if self.consistency_weight > 0:
                            loss_cons = loss_func(pred_hr_random, pred_hr.detach())
                        if return_condition_features:
                            layer_ids = self.rgb_feature_consistency_layers or list(
                                range(len(condition_features))
                            )
                            feature_losses = []
                            for layer_id in layer_ids:
                                if layer_id < 0 or layer_id >= len(condition_features):
                                    raise ValueError(
                                        f'rgb_feature_consistency_layers contains invalid index ' 
                                        f'{layer_id}; available range is ' 
                                        f'0..{len(condition_features) - 1}'
                                    )
                                anchor_feature = F.instance_norm(
                                    condition_features[layer_id].detach().float()
                                )
                                random_feature = F.instance_norm(
                                    random_condition_features[layer_id].float()
                                )
                                feature_losses.append(
                                    loss_func(random_feature, anchor_feature)
                                )
                            loss_feature_cons = torch.stack(feature_losses).mean()

                        if self.use_ema_detail_teacher and current_detail_weight > 0:
                            with torch.no_grad():
                                teacher_output, _ = self.ema_teacher(
                                    x_t, rgb, lq_up8, tt
                                )
                                teacher_prediction = teacher_output + lq_up8
                            loss_detail = loss_func(
                                self._high_pass(pred_hr_random.float()),
                                self._high_pass(teacher_prediction.float()),
                            )

                loss_grad = pred_hr.new_tensor(0.0)
                if self.gradient_loss_weight > 0:
                    loss_grad = gradient_loss(pred_hr, x_start)
                if self.use_nikon_rgb_restorer and self.nikon_restorer_hr_hsi_only:
                    hsi_loss = loss1 + self.gradient_loss_weight * loss_grad
                else:
                    hsi_loss = (
                        loss1 + loss3 + loss4 + loss2
                        + self.gradient_loss_weight * loss_grad
                    )
                loss = self.hsi_reconstruction_loss_weight * hsi_loss
                loss = loss + self.rgb_canonical_loss_weight * loss_canonical_rgb
                loss = loss + self.rgb_identity_loss_weight * loss_identity_rgb
                loss = loss + self.canonical_gate_loss_weight * loss_canonical_gate
                if (not self.use_pseudo_msi) and self.random_rgb_consistency and control_rgb is not None:
                    loss = loss + current_random_rgb_weight * loss_random + self.consistency_weight * loss_cons
                    loss = loss + self.rgb_feature_consistency_weight * loss_feature_cons
                    loss = loss + (
                        self.factorized_spatial_consistency_weight * loss_spatial_cons
                    )
                    loss = loss + current_detail_weight * loss_detail
                loss.backward()
                self.optimizer.step()
                if self.use_ema_detail_teacher and i >= self.ema_start_epoch:
                    self._update_ema_teacher()

            # 记录训练 loss 和学习率
            self.writer.add_scalar('Train/Loss_total', loss.item(), i)
            self.writer.add_scalar('Train/Loss_L1', loss1.item(), i)
            self.writer.add_scalar('Train/Loss_L2', loss2.item(), i)
            self.writer.add_scalar('Train/Loss_L3', loss3.item(), i)
            self.writer.add_scalar('Train/Loss_L4', loss4.item(), i)
            self.writer.add_scalar('Train/Loss_grad', loss_grad.item(), i)
            self.writer.add_scalar('Train/Loss_random_rgb', loss_random.item(), i)
            self.writer.add_scalar('Train/Loss_rgb_consistency', loss_cons.item(), i)
            self.writer.add_scalar(
                'Train/Loss_rgb_feature_consistency', loss_feature_cons.item(), i
            )
            self.writer.add_scalar(
                'Train/Loss_factorized_spatial_consistency', loss_spatial_cons.item(), i
            )
            self.writer.add_scalar(
                'Train/Loss_rgb_canonical', loss_canonical_rgb.item(), i
            )
            self.writer.add_scalar(
                'Train/Loss_rgb_identity', loss_identity_rgb.item(), i
            )
            self.writer.add_scalar(
                'Train/Loss_canonical_gate', loss_canonical_gate.item(), i
            )
            self.writer.add_scalar(
                'Train/Gate_augmented_mean', augmented_gate_mean.item(), i
            )
            self.writer.add_scalar(
                'Train/Gate_clean_mean', clean_gate_mean.item(), i
            )
            self.writer.add_scalar('Train/Loss_detail_consistency', loss_detail.item(), i)
            self.writer.add_scalar('Train/Weight_random_rgb', current_random_rgb_weight, i)
            self.writer.add_scalar('Train/Weight_detail', current_detail_weight, i)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, i)
            self.logger.info(
                f'Epoch {i} | Train Loss: {loss.item():.6f} | '
                f'L1: {loss1.item():.6f} L2: {loss2.item():.6f} '
                f'L3: {loss3.item():.6f} L4: {loss4.item():.6f} '
                f'Grad: {loss_grad.item():.6f} | '
                f'Rand: {loss_random.item():.6f} '
                f'Cons: {loss_cons.item():.6f} | '
                f'FeatCons: {loss_feature_cons.item():.6f} | '
                f'SpatialCons: {loss_spatial_cons.item():.6f} | '
                f'CanonicalRGB: {loss_canonical_rgb.item():.6f} '
                f'IdentityRGB: {loss_identity_rgb.item():.6f} | '
                f'GateLoss: {loss_canonical_gate.item():.6f} '
                f'GateAug: {augmented_gate_mean.item():.4f} '
                f'GateClean: {clean_gate_mean.item():.4f} | '
                f'Detail: {loss_detail.item():.6f} | '
                f'Wrand: {current_random_rgb_weight:.4f} '
                f'Wdetail: {current_detail_weight:.4f} | '
                f'LR: {current_lr:.2e}'
            )

            # 每个 epoch 结束后更新学习率
            self.scheduler.step()

            if (i + 1) % verbose == 0:
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = None
                if torch.cuda.is_available():
                    cuda_rng_state = torch.cuda.get_rng_state(self.device)
                if self.validation_seed is not None:
                    torch.manual_seed(int(self.validation_seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(self.validation_seed))
                img_index = 0
                psnr_sum = 0
                ssim_sum = 0
                sam_sum = 0
                ergas_sum = 0
                identity_rgb_sum = 0.0
                rgb_psnr_sum = 0.0
                for step, batch in enumerate(tqdm(self.val_dataloader)):
                    gt, lq, rgb, control_rgb, canonical_rgb_target, augmentation_label = self.unpack_rgb_batch(batch)
                    lq_up8 = nn.functional.interpolate(lq, scale_factor=8, mode='bicubic', align_corners=False)
                    lq_lr = lq.to(self.device).type(torch.float32)
                    lq_up8 = lq_up8.to(self.device).type(torch.float32)
                    rgb = rgb.to(self.device).type(torch.float32)
                    control_rgb = control_rgb.to(self.device).type(torch.float32) if control_rgb is not None else None
                    gt = gt.to(self.device).type(torch.float32)
                    model_rgb = rgb
                    apply_nikon_rgb_restorer = True
                    if self.use_nikon_rgb_restorer:
                        with torch.no_grad():
                            model_rgb, _, _ = self.Net.nikon_rgb_restorer(rgb, lq_lr)
                            rgb_target = self.Net.nikon_rgb_restorer.project_hsi(gt)
                        identity_rgb_sum += (
                            model_rgb - rgb_target
                        ).abs().mean().item()
                        rgb_mse = (model_rgb - rgb_target).square().mean().clamp_min(1e-12)
                        rgb_psnr_sum += (-10.0 * torch.log10(rgb_mse)).item()
                        apply_nikon_rgb_restorer = False
                    elif self.use_rgb_canonicalizer:
                        with torch.no_grad():
                            canonical_clean, _, _ = self.Net.rgb_canonicalizer(
                                rgb,
                                lq_up8,
                                gate_mode=self.canonicalizer_test_gate_mode,
                                gate_threshold=self.canonicalizer_gate_threshold,
                            )
                        identity_rgb_sum += (
                            canonical_clean - rgb
                        ).abs().mean().item()
                    if self.use_rgb_edge:
                        edgeget = Edge()
                        rgb_edge = edgeget(model_rgb)
                    else:
                        rgb_edge = None
                    indices = list(range(self.num_timesteps))[::-1]

                    if self.use_pseudo_msi:
                        # 原始模式：噪声/先验基于 cat(HSI, RGB)
                        lqrgb = torch.cat((lq_up8, rgb), dim=1)
                        noise = torch.randn_like(lqrgb)
                        x_t = self.EMRDIFF.prior_sample(lqrgb, noise, edge_map=rgb_edge)
                        residual_base = lqrgb
                    else:
                        # HSI-only 模式：噪声/先验仅基于 HSI
                        noise = torch.randn_like(lq_up8)
                        x_t = self.EMRDIFF.prior_sample(lq_up8, noise, edge_map=rgb_edge)
                        residual_base = lq_up8

                    psnr = PeakSignalNoiseRatio().to(self.device)
                    sam = SpectralAngleMapper().to(self.device)
                    ssim = MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
                    ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(self.device)
                    results = []
                    for t in indices:
                        tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                        with torch.no_grad():
                            x_pred, _ = self.Net(
                                x_t,
                                model_rgb,
                                lq_up8,
                                tt,
                                control_rgb=control_rgb,
                                canonicalizer_gate_mode=self.canonicalizer_test_gate_mode,
                                canonicalizer_gate_threshold=self.canonicalizer_gate_threshold,
                                rgb_restorer_lq=lq_lr,
                                apply_nikon_rgb_restorer=apply_nikon_rgb_restorer,
                            )
                            x_pred = x_pred + residual_base
                            noise = torch.randn_like(x_pred)
                            x_t = self.EMRDIFF.inverse_denoise(x_start=x_pred, x_t=x_t, t=tt, noise=noise, edge_map=rgb_edge)
                            results.append(deepcopy(x_t))

                    for j in range(len(results)):
                        img = results[j]
                        results[j] = img.clamp(min=-1.0, max=1.0)
                    results.append(lq)
                    final_prediction = results[-2]
                    if self.use_pseudo_msi:
                        # 原始模式：切除末尾3通道的 pseudo-MSI
                        band_p = self.configs.model.params.get('lqrgb_channels', dict)
                        final_prediction = final_prediction[:, 0:band_p - 3, :, :]
                    # HSI-only 模式：无需切片，输出即 HSI
                    psnr_v = psnr(final_prediction, gt).item()
                    ssim_v = ssim(final_prediction, gt).item()
                    sam_v = safe_sam(final_prediction, gt, sam)
                    ergas_v = ergas(final_prediction, gt).item()
                    print('PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(psnr_v, ssim_v, sam_v, ergas_v))
                    self.logger.info(
                        f'Val [{img_index}] PSNR: {psnr_v:.4f}, SSIM: {ssim_v:.4f}, '
                        f'SAM: {sam_v:.4f}, ERGAS: {ergas_v:.4f}'
                    )
                    if self.save_val_mat:
                        mat_data = {'data': final_prediction.squeeze(0).detach().cpu().numpy()}
                        sio.savemat(os.path.join(self.output_dir, f'{img_index}.mat'), mat_data)
                    img_index += 1
                    psnr_sum += psnr_v
                    ssim_sum += ssim_v
                    sam_sum += sam_v
                    ergas_sum += ergas_v
                avg_psnr = psnr_sum / img_index
                avg_ssim = ssim_sum / img_index
                avg_sam = sam_sum / img_index
                avg_ergas = ergas_sum / img_index
                avg_identity_rgb = identity_rgb_sum / img_index
                avg_rgb_psnr = rgb_psnr_sum / img_index if self.use_nikon_rgb_restorer else 0.0
                print('Average PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(avg_psnr, avg_ssim, avg_sam, avg_ergas))
                rgb_metric_label = (
                    'RGB Restoration L1'
                    if self.use_nikon_rgb_restorer else 'RGB Identity L1'
                )
                self.logger.info(
                    f'Epoch {i} | Val Avg | PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, '
                    f'SAM: {avg_sam:.4f}, ERGAS: {avg_ergas:.4f}, '
                    f'{rgb_metric_label}: {avg_identity_rgb:.6f}, '
                    f'RGB Restoration PSNR: {avg_rgb_psnr:.4f}'
                )
                self.writer.add_scalar('Val/PSNR', avg_psnr, i)
                self.writer.add_scalar('Val/SSIM', avg_ssim, i)
                self.writer.add_scalar('Val/SAM', avg_sam, i)
                self.writer.add_scalar('Val/ERGAS', avg_ergas, i)
                rgb_metric_key = (
                    'Val/RGB_restoration_L1'
                    if self.use_nikon_rgb_restorer else 'Val/RGB_identity_L1'
                )
                self.writer.add_scalar(rgb_metric_key, avg_identity_rgb, i)
                if self.use_nikon_rgb_restorer:
                    self.writer.add_scalar('Val/RGB_restoration_PSNR', avg_rgb_psnr, i)

                # 保存 latest checkpoint
                save_checkpoint(
                    self.Net, self.optimizer, self.scheduler, i + 1, self.output_dir,
                    best_metric=self.best_psnr, ema_model=self.ema_teacher,
                )
                self.logger.info(f'Saved latest checkpoint at epoch {i + 1}')

                # 保存 best checkpoint (PSNR 越大越好)
                if avg_psnr > self.best_psnr:
                    self.best_psnr = avg_psnr
                    save_best_checkpoint(
                        self.Net, self.optimizer, self.scheduler, i + 1, self.output_dir,
                        self.best_psnr, ema_model=self.ema_teacher,
                    )
                    self.logger.info(f'New best PSNR: {self.best_psnr:.4f}, saved best checkpoint')

                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, self.device)

        self.writer.close()

    def build_test_dataloader(self):
        """为最终测试构建整张图片的 dataloader（不做裁剪）"""
        opt = {}
        opt['paths'] = self.configs.data.val.params['dir_paths']
        opt['sf'] = self.configs.diffusion.params.get("sf")
        opt['gt_size'] = self.configs.data.val.params.get('gt_size')
        opt['srf_path'] = self.configs.data.val.params.get('srf_path', self.configs.data.get('srf_path', 'SRF/P_N_V2.mat'))
        opt['srf_mode'] = self.configs.data.val.params.get('srf_mode', self.configs.data.get('srf_mode', 'file'))
        opt['random_srf_seed'] = self.configs.data.val.params.get('random_srf_seed', self.configs.data.get('random_srf_seed', 0))
        opt['random_srf_bands'] = self.configs.data.val.params.get(
            'random_srf_bands', self.configs.data.get('random_srf_bands', self.configs.diffusion.params.get('band_dim', 31))
        )
        opt['random_srf_prob'] = self.configs.data.val.params.get('random_srf_prob', self.configs.data.get('random_srf_prob', 1.0))
        opt['return_control_rgb'] = self.configs.data.val.params.get('return_control_rgb', self.configs.data.get('return_control_rgb', False))
        opt['control_srf_mode'] = self.configs.data.val.params.get('control_srf_mode', self.configs.data.get('control_srf_mode', opt['srf_mode']))
        opt['control_srf_path'] = self.configs.data.val.params.get('control_srf_path', self.configs.data.get('control_srf_path', opt['srf_path']))
        opt['control_random_srf_seed'] = self.configs.data.val.params.get('control_random_srf_seed', self.configs.data.get('control_random_srf_seed', opt['random_srf_seed']))
        opt['control_random_srf_bands'] = self.configs.data.val.params.get(
            'control_random_srf_bands', self.configs.data.get('control_random_srf_bands', opt['random_srf_bands'])
        )
        opt['control_random_srf_prob'] = self.configs.data.val.params.get('control_random_srf_prob', self.configs.data.get('control_random_srf_prob', opt['random_srf_prob']))
        opt['use_full_img'] = True  # 整张图片测试
        batch_size = self.configs.train.get('batch')[1]
        num_workers = self.configs.train.get('num_workers')
        return DataLoader(DataloaderSimpleTest(opt), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    def _apply_hsi_backprojection(self, prediction, lq_hsi):
        for _ in range(self.test_hsi_backprojection_steps):
            batch, channels, height, width = prediction.shape
            flattened = prediction.reshape(batch * channels, height, width)
            degraded = _gaussian_blur_torch(flattened, sigma=2.0)
            degraded = F.interpolate(
                degraded.unsqueeze(0),
                size=lq_hsi.shape[-2:],
                mode='bicubic',
                align_corners=False,
            ).reshape_as(lq_hsi)
            residual = lq_hsi - degraded
            prediction = prediction + self.test_hsi_backprojection_strength * F.interpolate(
                residual, size=(height, width), mode='bicubic', align_corners=False
            )
        return prediction

    def test(self):
        # 构建整张图片测试的 dataloader
        test_dataloader = self.build_test_dataloader()
        self.logger.info(f'Test with full images, {len(test_dataloader)} samples')

        img_index = 0
        psnr1 = 0
        ssim1 = 0
        sam1 = 0
        ergas1 = 0
        for step, batch in enumerate(tqdm(test_dataloader)):
            gt, lq1, rgb, control_rgb, canonical_rgb_target, augmentation_label = self.unpack_rgb_batch(batch)
            lq = nn.functional.interpolate(lq1, scale_factor=8, mode='bicubic', align_corners=False)
            lq_lr = lq1.to(self.device).type(torch.float32)
            lq = lq.to(self.device).type(torch.float32)
            rgb = rgb.to(self.device).type(torch.float32)
            control_rgb = control_rgb.to(self.device).type(torch.float32) if control_rgb is not None else None
            model_rgb = rgb
            apply_nikon_rgb_restorer = True
            if self.use_nikon_rgb_restorer:
                with torch.no_grad():
                    model_rgb, _, _ = self.Net.nikon_rgb_restorer(rgb, lq_lr)
                apply_nikon_rgb_restorer = False
            if self.use_rgb_edge:
                edgeget = Edge()
                rgb_edge = edgeget(model_rgb)
                rgb_edge = rgb_edge.to(self.device).type(torch.float32)
            else:
                rgb_edge = None
            gt = gt.to(self.device).type(torch.float32)
            model = self.Net

            if self.use_pseudo_msi:
                # 原始模式：用于 FLOPs 计算和 prior_sample 的输入含 RGB
                lqrgb = torch.cat((lq, rgb), dim=1)
                # FLOPs 计算
                profile_checkpoint = self.configs.get('test', {}).get('profile_checkpoint', 'output/Harvard/HSI-only/checkpoints/best.pth.tar')
                if profile_checkpoint:
                    checkpoint = torch.load(profile_checkpoint, weights_only=False)
                    model.load_state_dict(checkpoint['model'])
                t = torch.tensor([1]).to(self.device)
                model.eval()
                # FLOPs profiling calls forward with positional args only and
                # cannot supply rgb_restorer_lq; skip it when the restorer is on
                # (backbone FLOPs are unchanged; the canonicalizer is closed-form).
                skip_flops = self.configs.get('test', {}).get(
                    'skip_flops_profile', False
                ) or self.use_nikon_rgb_restorer
                if not skip_flops:
                    model_copy = deepcopy(model)
                    model_copy.eval()
                    flops, params = profile(model_copy, inputs=(lqrgb, lq, rgb, t), verbose=False)
                    print(f'FLOPs: {flops}')
                    print(f'Params: {params}')
                    del model_copy

                lqrgb = torch.cat((lq, rgb), dim=1)
                band_p = self.configs.model.params.get('lqrgb_channels', dict)
                noise = torch.randn_like(lqrgb)
                x_t = self.EMRDIFF.prior_sample(lqrgb, noise, edge_map=rgb_edge)
                residual_base = lqrgb
            else:
                # HSI-only 模式
                # FLOPs 计算
                profile_checkpoint = self.configs.get('test', {}).get('profile_checkpoint', 'output/Harvard/HSI-only/checkpoints/best.pth.tar')
                if profile_checkpoint:
                    checkpoint = torch.load(profile_checkpoint, weights_only=False)
                    model.load_state_dict(checkpoint['model'])
                t = torch.tensor([1]).to(self.device)
                model.eval()
                model_copy = deepcopy(model)
                model_copy.eval()
                # flops, params = profile(model_copy, inputs=(lq, lq, rgb, t), verbose=False)
                # print(f'FLOPs: {flops}')
                # print(f'Params: {params}')
                del model_copy

                noise = torch.randn_like(lq)
                x_t = self.EMRDIFF.prior_sample(lq, noise, edge_map=rgb_edge)
                residual_base = lq

            indices = list(range(self.num_timesteps))[::-1]
            psnr = PeakSignalNoiseRatio().to(self.device)
            sam = SpectralAngleMapper().to(self.device)
            ssim = MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
            ergas = ErrorRelativeGlobalDimensionlessSynthesis().to(self.device)
            #start_time = time.time()
            final_prediction = None
            use_amp = self.test_use_amp and self.device.type == 'cuda'
            for t in indices:
                tt = torch.tensor([t] * x_t.shape[0], device=x_t.device)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        x_pred, _ = model(
                            x_t,
                            model_rgb,
                            lq,
                            tt,
                            control_rgb=control_rgb,
                            canonicalizer_gate_mode=self.canonicalizer_test_gate_mode,
                            canonicalizer_gate_threshold=self.canonicalizer_gate_threshold,
                            rgb_restorer_lq=lq_lr,
                            apply_nikon_rgb_restorer=apply_nikon_rgb_restorer,
                        )
                        x_pred = x_pred + residual_base
                        noise = torch.randn_like(x_pred)
                        x_t = self.EMRDIFF.inverse_denoise(x_start=x_pred, x_t=x_t, t=tt,
                                                                           noise=noise, edge_map=rgb_edge)
                    if self.test_final_state == 'posterior':
                        final_state = x_t
                    else:
                        final_state = x_pred
                    final_prediction = final_state.float()
            #Ours_time = time.time() - start_time
            #print(Ours_time)
            if self.use_pseudo_msi:
                final_prediction = final_prediction[:, 0:band_p - 3, :, :]
            # HSI-only 模式：无需切片
            final_prediction = self._apply_hsi_backprojection(
                final_prediction, lq_lr
            ).clamp(min=-1.0, max=1.0)
            psnr_v = psnr(final_prediction, gt).item()
            ssim_v  = ssim(final_prediction, gt).item()
            sam_v = safe_sam(final_prediction, gt, sam)
            ergas_v  = ergas(final_prediction, gt).item()
            print('PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(psnr_v , ssim_v , sam_v , ergas_v ))
            self.logger.info(
                f'Test [{img_index}] PSNR: {psnr_v:.4f}, SSIM: {ssim_v:.4f}, '
                f'SAM: {sam_v:.4f}, ERGAS: {ergas_v:.4f}'
            )
            psnr1 += psnr_v
            ssim1 += ssim_v
            sam1 += sam_v
            ergas1 += ergas_v
            if self.save_test_mat:
                mat_data = {'data': final_prediction.squeeze(0).detach().cpu().numpy()}
                sio.savemat(os.path.join(self.output_dir, f'{img_index}.mat'), mat_data)
            img_index += 1
        avg_psnr = psnr1/img_index
        avg_ssim = ssim1/img_index
        avg_sam = sam1/img_index
        avg_ergas = ergas1/img_index
        print('Average PSNR: {:.4f}, ssim: {:.4f}, sam: {:.4f}, ergas:{:.4f}.'.format(avg_psnr, avg_ssim, avg_sam, avg_ergas))
        self.logger.info(
            f'Test Avg | PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, '
            f'SAM: {avg_sam:.4f}, ERGAS: {avg_ergas:.4f}'
        )
        self.writer.add_scalar('Test/PSNR', avg_psnr, 0)
        self.writer.add_scalar('Test/SSIM', avg_ssim, 0)
        self.writer.add_scalar('Test/SAM', avg_sam, 0)
        self.writer.add_scalar('Test/ERGAS', avg_ergas, 0)
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'sam': avg_sam,
            'ergas': avg_ergas,
        }
        self.writer.close()

