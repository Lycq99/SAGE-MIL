# SAGE-MIL

**Pathology-Related Semantic Guidance with Normal-Tissue References for Weakly Supervised Whole-Slide Image Classification**

This repository provides the SAGE-MIL implementation and the accompanying training, evaluation, and analysis utilities. The model combines pathology-related semantic guidance, a normal-tissue reference dictionary, sparse semantic routing, spatial MIL, and a Sinkhorn-based feature-replacement branch for prediction-sensitivity analysis.

## Overview

```text
Whole-slide image
  ↓
20× tissue patch extraction
  ↓
Frozen CONCH / PLIP patch features
  ↓
Morphology / Texture / Microenvironment anchors
  ↓
Normal-tissue reference adjustment
  ↓
Sparse semantic routing
  ↓
Nyström Transformer + SCConv
  ↓
Bag prediction + auxiliary instance prediction
```

During training, high-response patches are matched to normal-tissue reference targets with Sinkhorn OT and continuously replaced in feature space. The same replacement mechanism is used after training for OT-Key and matched control analyses. Routine inference uses only the classification path.

## Repository layout

```text
configs/              experiment settings and semantic prompts
data/                 manifest and split utilities
feature_extraction/   WSI preprocessing and PLIP/CONCH feature extraction
semantic_anchors/     text-anchor construction
proxy_pool/           normal-tissue reference dictionary construction
src/sage_mil/         model, losses, data loading, and core modules
train/                training entry points
evaluation/           classification and calibration metrics
audit/                controlled feature-replacement analyses
robustness/           synthetic feature perturbations
figures/              plotting utilities for quantitative analyses
docs/                 method and reproduction notes
tests/                unit tests
```

## Environment

The experiments were run with:

- Python 3.9.23
- PyTorch 2.5.1
- CUDA 12.1
- 2 × NVIDIA GeForce RTX 4090
- NVIDIA driver 535.161.08

Install a CUDA-enabled PyTorch 2.5.1 build first, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

CONCH is installed from its official repository. PLIP is loaded through the Hugging Face CLIP interface.

## Data and patch features

Raw CAMELYON16 and TCGA slides are not redistributed. Classification manifests use one row per WSI:

```csv
patient_id,slide_id,feature_path,label,split
TCGA-XX-0001,slide_001,/path/to/slide_001.pt,0,train
```

WSIs are processed at 20× with non-overlapping 256×256 model inputs. Tissue is identified from the HSV saturation channel with Otsu thresholding, using a 10% tissue threshold. No additional pathology-specific stain normalization is applied.

TCGA splits are patient-level. `Solid Tissue Normal` slides are kept separate from the subtype-classification cohorts and are used only to construct training-side normal-tissue reference dictionaries.

## Semantic anchors

Three fixed pathology concepts are used:

1. Morphology
2. Texture
3. Microenvironment

The prompts are stored in `configs/semantic_prompts.yaml`. Each concept prompt is encoded with the frozen text encoder associated with the feature space and stored as a fixed semantic anchor.

CONCH example:

```bash
python semantic_anchors/encode_conch_prompts.py \
  --prompts configs/semantic_prompts.yaml \
  --weight /path/to/conch_checkpoint.bin \
  --output assets/conch_semantic_anchors.pt
```

PLIP example:

```bash
python semantic_anchors/encode_plip_prompts.py \
  --prompts configs/semantic_prompts.yaml \
  --output assets/plip_semantic_anchors.pt
```

## Normal-tissue reference dictionary

Reference prototypes are constructed from training-side normal-tissue features only. The reported configuration uses `M=100` prototypes. For CAMELYON16-CONCH, the stored reference tensor has shape `100 × 512`.

```bash
python proxy_pool/build_proxy_pool.py \
  --manifest data/proxy/train_normal_manifest.csv \
  --output assets/camelyon16_proxy_pool.pt \
  --num-prototypes 100 \
  --seed 2022
```

## Training

Experiment settings are stored under `configs/experiments/`.

```bash
PYTHONPATH=src python train/train.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --seed 2022
```

The four training seeds are `2021`, `2022`, `2023`, and `2024`; the data split seed is `2022`.

Main settings used in the study:

| Setting | Value |
|---|---:|
| Feature dimension | 512 |
| Semantic concepts | 3 |
| Reference prototypes | 100 |
| Semantic top-k ratio | 0.30 |
| Reference temperature | 0.10 |
| Correction mask | [1, 1, 0] |
| Feature noise during training | σ = 0.005 |
| Instance top-k | 10 → 1 |
| Instance top-k at inference | 6 |
| Training replacement patches | 5 |
| Sinkhorn epsilon | 0.05 |
| Sinkhorn iterations | 30 |
| Norm-match clip | [0.5, 2.0] |
| Margin target | 0.25 |
| Replacement-loss weight | 0.10 |
| Replacement warm-up | 20 epochs |
| Orthogonality weight | 0.10 |
| Evaluation probability | 0.5 bag + 0.5 instance |

Semantic response routing uses projected anchors, patch features, and reference prototypes. Cosine responses use a learned scale capped at 100, reference allocation uses temperature `T_e`, and the correction mask is applied to Morphology and Texture but not Microenvironment. Sparse responses are normalized within each head and averaged before semantic pooling.

The checkpoint with the lowest validation loss is retained. For binary tasks, the operating threshold is selected on the validation set by maximizing F1 over 0.10–0.90 in steps of 0.05. RCC uses the maximum predicted class probability.

## Calibration

Calibration uses the fused bag/instance probabilities and 15 equal-width confidence bins.

Binary task:

```bash
python evaluation/calibration_metrics.py predictions.csv --task binary
```

Multiclass task:

```bash
python evaluation/calibration_metrics.py predictions.csv --task multiclass
```

For multiclass predictions, the input CSV contains `prob_0`, `prob_1`, ... columns. Brier score is computed from the one-hot target vector; ECE uses maximum predicted probability and correctness.

## Feature-replacement analysis

```bash
PYTHONPATH=src:. python audit/replacement_protocol.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --checkpoint /path/to/best.ckpt \
  --budget 200 \
  --output outputs/audit/replacement_curves.csv
```

The retrospective protocol includes OT-Key, OT-Random, OT-Low, Mean-Key, and Zero-Key controls. Summary utilities report AURD, AUPC, AOPC, AOPCR, paired bootstrap confidence intervals, and paired sign-randomization tests.


## Robustness analysis

```bash
PYTHONPATH=src:. python robustness/run_robustness.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --checkpoint /path/to/best.ckpt \
  --output outputs/robustness.csv \
  --device cuda
```

The robustness analysis uses four controlled perturbations. Gaussian noise is scaled by the global standard deviation of all patch-feature elements within each WSI. Dictionary noise is scaled by the global standard deviation of the current prototype bank. Background mixing is applied only to the lowest clean Morphology/Texture-attention 50% of patches, with one proxy target sampled independently for each selected patch. Channel-affine shift uses clipped Gaussian scale and shift directions for every WSI and feature channel, with the shift term scaled by the within-WSI channel standard deviation. Exact definitions are given in `robustness/README.md`.

## Implementation notes

- A zero-mean Gaussian perturbation with standard deviation `0.005` is applied to input features during training only.
- OT candidate features are detached while replacement targets are constructed.
- Bag and fused bag/instance outputs are re-evaluated after replacement for the retrospective sensitivity analysis.
- Patch coordinates are carried by the data loader but are not consumed by the current forward path.
- Raster-order square grids use zero padding without an explicit padding mask.
- Binary margins use the signed scalar logit, equivalent to the two-class margin under a symmetric logit representation; RCC uses the explicit true-class versus strongest-other-class margin.

## Released material

The repository contains source code, experiment configurations, data-manifest utilities, evaluation code, analysis code, and plotting utilities. Raw WSIs, third-party pretrained weights, experiment checkpoints, predictions, and local logs are not included.
