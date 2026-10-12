# RGB Color-Domain Canonicalization for Hyperspectral Image Fusion

Test-time RGB color-domain canonicalization for frozen hyperspectral super-resolution models.

This project builds on [EMR-Diff](https://github.com/luocz55/EMR-Diff), retains [UAFL](https://github.com/yingkai-zhang/UAFL) code, and uses the real-data conventions and utilities distributed with [BUSIFusion](https://github.com/CPREgroup/Real-Spec-RGB-Fusion).

## Core implementation

- `scripts/cviu_canonicalizer_degree.py`: `canonicalize()` implements LR-HSI anchoring, shared polynomial fitting, scale-normalized ridge regression and residual back-projection. The default is `c5_forward_bp1` with `lam=1e-6`.
- `scripts/cviu_observations.py`, `composite_pipeline.py`, `rgb_filters_citable.py`: camera, illumination and RGB color transformations.
- `arch/BAFUnet.py` and `arch/uafl_model.py`: reconstruction backbones.

## Environment

Python 3.12, PyTorch 2.5.1 with CUDA, NumPy, SciPy, torchvision, OmegaConf, torchmetrics, einops, timm, thop, tqdm, colour-science, OpenCV, scikit-image, h5py, Matplotlib and TensorBoard.

## Data and checkpoints

Download [Harvard](https://vision.seas.harvard.edu/hyperspec/index.html), [ICVL](https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html), and [BUSIFusion](https://github.com/CPREgroup/Real-Spec-RGB-Fusion) from their original providers. Data and checkpoints are not included.

Synthetic datasets use `dataset/{Harvard,ICVL}/{train,test}/*.mat`, with an HWC `ref` array. Use the same data split, preprocessing and matching checkpoints as the experiments. Expected checkpoint paths:

```text
output/Harvard/h_change_dataset_split_and_dataprocess_basline/checkpoints/best.pth.tar
output/ICVL/baseline/checkpoints/best.pth.tar
output/Harvard/uafl_baseline/best.pth.tar
```

## Main experiments

Run commands from the repository root. Outputs are written to `output/CVIU/<name>/`; use a new name when changing an experiment.

```bash
python scripts/cviu_run_cross.py --backbone emr --dataset Harvard --split test --max-scenes 20 --crop 0 --methods raw c5_forward_bp1 --lambdas 1e-6 --save-scenes 0 --name harvard_main
```

For ICVL, use `--dataset ICVL`. For the Harvard UAFL experiment, use `--backbone uafl --crop 512`. `--conditions` accepts `clean`, `gamma_encode`, `white_balance_fluorescent`, `phase_one`, `A_indoor`, `B_outdoor`, `C_stress`, and additional conditions defined in `scripts/cviu_observations.py`.

## Ablations

```bash
python scripts/cviu_run_degree.py --split test --max-scenes 20 --crop 0 --methods raw anchor bp c5_lr_bp0 c5_lr_bp1 c5_forward_bp0 c5_forward_bp1 --lambdas 1e-6 --save-scenes 0 --name mechanism_ablation
python scripts/cviu_run_degree.py --split test --max-scenes 20 --crop 0 --methods affine_forward_bp1 q2_forward_bp1 c3_forward_bp1 c4_forward_bp1 c5_forward_bp1 root2_forward_bp1 rbf_forward_bp1 --lambdas 1e-6 --save-scenes 0 --name basis_ablation
```

Use `--split train` for development-set parameter selection and `--lambdas` for a regularization sweep. `scripts/cviu_run.py` retains the earlier basis implementation used by the corresponding experiments. `--seed` controls paired inference randomness. Evaluation CSVs include reconstruction metrics and runtime measurements.

## Real RGB-HSI experiments

Place the provider's `rgbrybspec/` folder and `wavelength.mat` under `dataset/CVIU_real/inspection/`, then run:

```bash
python scripts/cviu_prepare_real.py
python scripts/cviu_refine_real.py
python scripts/cviu_real_eval.py
```

The first two steps generate registered samples and preparation records under `prepared_v1/` and `prepared_v2/`. Scene IDs and the canonicalization settings are in `config/real_scenes.json` and `config/canonicalization.json`.

Backbone training entry points are `Train.py config/harvard_baseline.yaml`, `Train.py config/icvl_baseline.yaml`, and `scripts/train_test_uafl.py --mode train`.
