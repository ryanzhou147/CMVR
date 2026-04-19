# SCARCE-CXR

**Self-supervised Cross-domain Adaptation for Rare Clinical Entities in Chest X-Rays**

[![Blog](https://img.shields.io/badge/Blog-rzhou.me-informational)](https://rzhou.me/thoughts/scarce-cxr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

When a novel disease appears, you might have days to build a classifier with 23 labeled examples. Standard supervised training needs thousands. This project tests whether SSL pretraining on 112k unlabeled NIH chest X-rays closes that gap, evaluated on 10 genuinely rare diseases from PadChest (Spanish hospital, zero label overlap) at 1, 5, 10, 20, and 50 shots.

**Short answer: yes, for MoCo.** Domain-specific SSL beats ImageNet initialization at every shot count. The gap is largest at 1–5 shots, which is where it matters most clinically.

---

## Results

Two evaluation protocols per backbone:
- **Probe:** frozen backbone + logistic regression (isolates representation quality)
- **Finetune:** gradient descent through layer2+, prototype head init (deployment-realistic)

**50-shot probe AUC, averaged across 10 rare PadChest diseases:**

| Method | Backbone | SSL pretrained | ImageNet | Random init |
|--------|----------|:--------------:|:--------:|:-----------:|
| MoCo v2 + VICReg | ResNet50, ep774 | **0.742** | 0.694 | 0.623 |
| BarlowTwins | ResNet18, ep191 | 0.725 | 0.694 | 0.623 |
| SparK | ResNet50, ep199 | 0.641 | 0.694 | 0.623 |
| DINO | ResNet50, ep15 | 0.487 (failed) | N/A | N/A |

MoCo SSL > ImageNet at all shots (p ≤ 0.01 at 1-shot). BarlowTwins matches within 1.7pp using a 4x smaller backbone. SparK beats ImageNet only at 1-shot, as generative pretraining pushes reconstruction, not discrimination. DINO collapsed from epoch 0: loss pinned, representations worse than random.

---

## Pipeline

```
NIH ChestX-ray14          SSL Pretraining            PadChest Eval
112k unlabeled images  →  MoCo / BarlowTwins  →  Linear Probe   →  Rare Disease AUC
(cross-domain source)     SparK / DINO            Gradient Finetune   (1–50 shots)
```

Cross-domain transfer: US hospital dataset (NIH) to Spanish hospital dataset (PadChest). Domain gap measured at 0.010 mean pixel shift: small enough to be meaningful, large enough to matter.

---

## Setup

Requires Python 3.14 and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/ryanzhou147/SCARCE-CXR
cd SCARCE-CXR
uv sync
```

**Data:**

```bash
# NIH ChestX-ray14 (112k images, ~3.1GB after pre-resize)
uv run python -m data.download --dataset nih

# PadChest (requires manual access approval; see results/reference.md)
uv run python -m data.download --dataset padchest
```

---

## Pretraining

```bash
# MoCo v2 + VICReg variance term (ResNet50, ~63h on L4 GPU)
uv run python main.py pretrain-moco --config configs/moco.yaml

# BarlowTwins (ResNet18, ~15h on L4 GPU)
uv run python main.py pretrain-barlow --config configs/barlow.yaml

# SparK masked autoencoder (ResNet50, ~20h on L4 GPU)
uv run python main.py pretrain-spark --config configs/spark.yaml

# Override any config field
uv run python main.py pretrain-moco --training.lr=1e-3 --training.epochs=50
```

> DINO is not recommended on this dataset. See `results/eval_summary.md` and [Part 4 of the blog](https://rzhou.me/thoughts/scarce-cxr4) for the failure analysis.

---

## Evaluation

```bash
# Frozen probe (primary metric)
uv run python -m finetune.probe --checkpoint outputs_cloud/moco-v2/best.pt

# Gradient finetune
uv run python -m finetune.finetune --checkpoint outputs_cloud/moco-v2/best.pt

# Check disease counts before running (threshold-based selection, not hand-picked)
uv run python -m finetune.count_labels

# Collapse monitor (run during pretraining to catch dimensional collapse early)
uv run python -m data.eval.collapse_monitor --outputs-dir outputs/moco-v2

# Run all probes + finetunes
bash run_eval.sh
```

---

## Checkpoints

| Path | Method | Backbone | Epoch |
|------|--------|----------|-------|
| `outputs_cloud/moco-v2/best.pt` | MoCo v2 + VICReg | ResNet50 | 774 |
| `outputs_cloud/barlow/best.pt` | BarlowTwins | ResNet18 | 191 |
| `outputs_cloud/spark/best.pt` | SparK | ResNet50 | 199 |

Load any checkpoint with `data/load_backbone.py`: auto-detects method from checkpoint dict and returns a frozen feature extractor.

---

## Key Implementation Notes

**RAM caching:** images loaded into shared memory at startup. Without it: ~60% GPU utilization, I/O-bound. With it: ~95%, compute-bound. Single highest-impact optimization on the L4 instance.

**Collapse monitoring:** `mean_cos` and `eff_rank` tracked across checkpoints. MoCo v1 collapsed silently at epoch 100; the VICReg variance term fixed it. Run this early, because catching collapse at epoch 50 is much cheaper than epoch 200.

**Surgical unfreezing:** MoCo and BarlowTwins unfreeze layer2+. SparK unfreezes layer3+ only (layer2 adapted during U-Net decoder pretraining). BatchNorm always frozen, because statistics from 112k NIH images are worth preserving.

**Patient-level splits:** train/val split by PatientID, not by image. A patient with scans in both splits leaks anatomy into evaluation. Patient-level isolation prevents this.

**Disease selection:** `EXCLUDE_LABELS` in `finetune/_data.py` removes diseases with large public labeled datasets (pneumonia, pleural effusion, cardiomegaly, etc.). The entire point is evaluating on findings with no existing public dataset.

**SparK adaptation:** the original SparK uses sparse convolution to skip masked regions entirely, which is critical. Standard dense convolutions applied to zeroed patches corrupt neighboring unmasked features through the receptive field. Sparse convolution libraries (MinkowskiEngine) were unavailable on the L4 cloud instance, so this implementation uses dense convolutions with zeroed patches, a known approximation that likely explains SparK's weaker probe performance relative to the original paper's results.

---

## Cloud Budget

All pretraining ran within the $300 Google Cloud free trial on a single L4 GPU (22GB VRAM).

| Method | Batch size | Time | Est. cost |
|--------|-----------|------|-----------|
| MoCo v2 | 384 | ~63h | ~$44 |
| BarlowTwins | 512 | ~15h | ~$10 |
| SparK | 256 | ~20h | ~$14 |
| DINO | 64 | ~1h (killed) | ~$1 |

---

## Repository Structure

```
ssl_methods/
├── moco/          # MoCo v2 + VICReg variance term
├── barlow/        # BarlowTwins cross-correlation loss
├── spark/         # Masked autoencoder (dense conv approximation)
└── dino/          # DINO self-distillation (failed on this domain)

finetune/
├── probe.py       # Frozen backbone + logistic regression
├── finetune.py    # Gradient finetune, layer2+ unfrozen
├── _data.py       # PadChest loading, EXCLUDE_LABELS, n-shot sampling
└── _plots.py      # AUC curves per disease and averaged

data/
├── dataloader.py      # RAM-cached dataset loader
├── load_backbone.py   # Shared backbone loader, auto-detects method
├── download.py        # NIH + PadChest download
├── eval/
│   └── collapse_monitor.py   # mean_cos + eff_rank across checkpoints
└── viz/               # GradCAM, augmentation viewer, embeddings

configs/               # Per-method YAML hyperparameters
results/               # Eval summaries, probe tables, design notes
```

---

## Blog

Full write-up in 6 parts at [rzhou.me/thoughts/scarce-cxr](https://rzhou.me/thoughts/scarce-cxr):

1. [Motivation](https://rzhou.me/thoughts/scarce-cxr1): why SSL for rare disease, the labeled data problem
2. [Data + Setup](https://rzhou.me/thoughts/scarce-cxr2): RAM caching, backbone choice, X-ray augmentation
3. [MoCo + BarlowTwins](https://rzhou.me/thoughts/scarce-cxr3): InfoNCE, dimensional collapse, VICReg fix
4. [DINO + SparK](https://rzhou.me/thoughts/scarce-cxr4): DINO's failure, generative pretraining, training costs
5. [Finetuning](https://rzhou.me/thoughts/scarce-cxr5): domain gap, probe vs finetune, Grad-CAM analysis
6. [Analysis](https://rzhou.me/thoughts/scarce-cxr6): results breakdown, limitations, next steps
