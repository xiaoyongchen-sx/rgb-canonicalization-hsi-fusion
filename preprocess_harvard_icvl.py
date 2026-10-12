"""
Harvard & ICVL 数据集预处理脚本

根据作者在 GitHub issue 中的说明：
- Harvard: 室内(CZ_hsdbi) + 室外(CZ_hsdb), 随机采样 40 个样本 (30 train + 10 test)
- ICVL: ~200 张图, 随机采样 70 个样本 (50 train + 20 test)
- 每个样本是原图的 center crop (256×256)

输出格式与 dataloader 一致：
- .mat 文件, key='ref', shape=(H, W, C)=(256, 256, bands)
"""

import os
import random
import glob
import numpy as np
import scipy.io as sio
import h5py

PATCH_SIZE = 256
random.seed(42)  # 固定随机种子保证可复现


def center_crop(img, crop_size):
    """从图像中心裁剪 crop_size×crop_size 的 patch
    img: (H, W, C)
    """
    h, w = img.shape[:2]
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    i = (h - ch) // 2
    j = (w - cw) // 2
    return img[i:i+ch, j:j+cw, :]


def preprocess_harvard():
    """处理 Harvard 数据集: 室内+室外 → 随机采样40个 (30 train + 10 test)"""
    print("=" * 50)
    print("处理 Harvard 数据集")

    # 收集所有 .mat 文件
    outdoor_dir = 'dataset/harvard/CZ_hsdb'
    indoor_dir = 'dataset/harvard/CZ_hsdbi'

    outdoor_files = sorted(glob.glob(os.path.join(outdoor_dir, '*.mat')))
    indoor_files = sorted(glob.glob(os.path.join(indoor_dir, '*.mat')))

    print(f"  室外 (CZ_hsdb): {len(outdoor_files)} 张")
    print(f"  室内 (CZ_hsdbi): {len(indoor_files)} 张")

    all_files = outdoor_files + indoor_files
    print(f"  总计: {len(all_files)} 张")

    # 随机采样 40 个
    n_sample = 40
    n_train = 30
    n_test = 10

    sampled = random.sample(all_files, n_sample)
    train_files = sampled[:n_train]
    test_files = sampled[n_train:]

    # 创建输出目录
    train_dir = 'dataset/Harvard/train'
    test_dir = 'dataset/Harvard/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 处理并保存
    for split, files, out_dir in [('train', train_files, train_dir), ('test', test_files, test_dir)]:
        for idx, fpath in enumerate(files):
            d = sio.loadmat(fpath)
            ref = d['ref']  # (1040, 1392, 31)
            patch = center_crop(ref, PATCH_SIZE)  # (256, 256, 31)
            save_path = os.path.join(out_dir, f'{idx:04d}.mat')
            sio.savemat(save_path, {'ref': patch})
        print(f"  {split}: {len(files)} 个样本保存到 {out_dir}/")

    print(f"Harvard 完成! train={n_train}, test={n_test}")


def preprocess_icvl():
    """处理 ICVL 数据集: ~200张 → 随机采样70个 (50 train + 20 test)"""
    print("=" * 50)
    print("处理 ICVL 数据集")

    mat_dir = 'dataset/icvl/mat'
    all_files = sorted(glob.glob(os.path.join(mat_dir, '*.mat')))
    print(f"  总计: {len(all_files)} 张")

    # 随机采样 70 个
    n_sample = 70
    n_train = 50
    n_test = 20

    sampled = random.sample(all_files, n_sample)
    train_files = sampled[:n_train]
    test_files = sampled[n_train:]

    # 创建输出目录
    train_dir = 'dataset/ICVL/train'
    test_dir = 'dataset/ICVL/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 处理并保存
    for split, files, out_dir in [('train', train_files, train_dir), ('test', test_files, test_dir)]:
        for idx, fpath in enumerate(files):
            with h5py.File(fpath, 'r') as hf:
                rad = np.array(hf['rad'])  # (31, H, W)
            # 转为 (H, W, C)
            rad = rad.transpose(1, 2, 0)
            # 中心裁剪
            patch = center_crop(rad, PATCH_SIZE)  # (256, 256, 31)
            save_path = os.path.join(out_dir, f'{idx:04d}.mat')
            sio.savemat(save_path, {'ref': patch})
        print(f"  {split}: {len(files)} 个样本保存到 {out_dir}/")

    print(f"ICVL 完成! train={n_train}, test={n_test}")


if __name__ == '__main__':
    preprocess_harvard()
    print()
    preprocess_icvl()
    print("\n全部预处理完成!")
