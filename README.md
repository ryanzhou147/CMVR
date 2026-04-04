# CLEAR-CXR
*Contrastive Learning for Emerging Anomalies and Rarities in Chest X-Rays. For the long tail of medicine.*

SSL pretraining on 112k unlabeled NIH chest X-rays, then few-shot fine-tuning on 10 rare diseases from PadChest. Built to answer a specific question: when a novel disease appears and your hospital has 23 labeled examples, does domain-specific SSL actually help?

The short version: **yes, for MoCo.** SSL pretrained features beat ImageNet initialization at every shot count from 1 to 50, with the gap largest at 1-5 shots where it matters most. DINO failed completely — loss flatlined from epoch 0, final representations worse than random init. All of this ran on a single L4 GPU for $0 on Google Cloud free credits.


---

## Results

Probe = frozen backbone + sklearn LogisticRegressionCV (clean signal, isolates representation quality).
Fine-tune = gradient descent through layer2+, AdamW, prototype head init.

**50-shot probe AUC, averaged across 10 rare diseases:**

| Method | Backbone | Epochs | SSL pretrained | ImageNet | Random init |
|--------|----------|--------|----------------|----------|-------------|
| MoCo + VICReg | ResNet50 | 774 | **0.742** | 0.694 | 0.623 |
| BarlowTwins | ResNet18 | 191 | 0.725 | 0.694 | 0.623 |
| SparK | ResNet50 | 199 | 0.641 | 0.694 | 0.623 |
| DINO | ResNet50 | 15 (failed) | 0.487 | — | — |

MoCo SSL > ImageNet at all shots (p ≤ 0.01 at 1-shot, p ≤ 0.026 at worst point).
BarlowTwins 1.7pp behind MoCo at 50-shot with a 4× smaller backbone.
SparK only beats ImageNet at 1-shot; generative pretraining doesn't push embeddings apart.
DINO: mean_cos increasing and eff_rank decreasing throughout — actively collapsing during training.

---

## Repo Structure

```
ssl_methods/
├── moco/          # MoCo v2 + VICReg variance term (ResNet50, queue + InfoNCE)
├── barlow/        # BarlowTwins (ResNet18, cross-correlation decorrelation loss)
├── spark/         # SparK masked autoencoder (ResNet50 encoder + U-Net decoder)
└── dino/          # DINO self-distillation (ResNet50, failed on this domain)

finetune/
├── probe.py       # Frozen backbone + logistic regression — primary eval metric
├── finetune.py    # Gradient fine-tuning through layer2+ (layer3+ for SparK)
├── _data.py       # PadChest data loading, EXCLUDE_LABELS, n-shot sampling
├── _plots.py      # AUC plotting (mean + per-disease)
└── count_labels.py

data/
├── dataloader.py      # RAM-cached dataset loader (~2GB in-memory, eliminates disk I/O bottleneck)
├── load_backbone.py   # Shared backbone loader — auto-detects MoCo/DINO/Barlow/SparK from checkpoint
├── download.py        # NIH + PadChest download scripts
├── eval/
│   ├── collapse_monitor.py   # mean_cos + eff_rank across checkpoints (tracks dimensional collapse)
│   └── few_shot_probe.py     # NIH 15-class few-shot probe (not the primary metric — see CLAUDE.md)
└── viz/               # GradCAM, augmentation viewer, embedding visualizations

configs/
├── moco.yaml      # ResNet50, queue=65536, temp=0.07, VICReg var_weight=1.0, bs=384, ep=800
├── barlow.yaml    # ResNet18, proj_dim=2048, lambda=0.005, bs=512, ep=200
├── spark.yaml     # ResNet50, mask_ratio=0.60, bs=256, ep=200
└── dino.yaml      # ResNet50, out_dim=256, ep=15 (killed — loss pinned from epoch 0)

results/
├── eval_summary.md        # Full collapse monitor tables + method selection rationale
├── reference.md           # Design decisions for PadChest fine-tuning eval
└── probe_padchest_*.md    # Per-disease AUC tables
```

---

## Setup

```bash
# Python 3.14, uv package manager
uv sync

# Download NIH ChestX-ray14 (112k images, ~3.1GB after pre-resize)
uv run python -m data.download --dataset nih

# Download PadChest (requires manual approval — see results/reference.md)
uv run python -m data.download --dataset padchest
```

---

## Pretraining

```bash
# MoCo + VICReg (ResNet50, 800 epochs, ~63h on L4)
uv run python main.py pretrain-moco --config configs/moco.yaml

# BarlowTwins (ResNet18, 200 epochs, ~15h on L4)
uv run python main.py pretrain-barlow --config configs/barlow.yaml

# SparK masked autoencoder (ResNet50, 200 epochs, ~20h on L4)
uv run python main.py pretrain-spark --config configs/spark.yaml

# DINO — not recommended on this dataset (see results/eval_summary.md)
uv run python main.py pretrain-dino --config configs/dino.yaml

# Override any config field on the CLI
uv run python main.py pretrain-moco --training.lr=1e-3 --training.epochs=50
```

---

## Evaluation

```bash
# PadChest few-shot probe (primary metric — frozen backbone + logistic regression)
uv run python -m finetune.probe --checkpoint outputs_cloud/moco-v3/best.pt

# PadChest fine-tuning (gradient descent, layer2+ unfrozen)
uv run python -m finetune.finetune --checkpoint outputs_cloud/moco-v3/best.pt

# Check class counts before running (disease selection is threshold-based, not hand-picked)
uv run python -m finetune.count_labels

# Collapse monitor — run during or after pretraining to catch dimensional collapse early
uv run python -m data.eval.collapse_monitor --outputs-dir outputs/moco-v3

# Run all probes + finetunes (bash script, failures don't abort siblings)
bash run_eval.sh
```

---

## Key Implementation Notes

**RAM caching** (`data/dataloader.py`): all images loaded into shared memory at startup. Without it: ~60% GPU utilization, dataloader-bound. With it: ~95%, compute-bound. On the L4 instance this was the single most impactful optimization.

**Collapse monitoring** (`data/eval/collapse_monitor.py`): tracks `mean_cos` (pairwise cosine similarity, low = diverse embeddings) and `eff_rank` (effective rank of embedding covariance, high = many directions used). MoCo v1 collapsed silently at epoch 100 — the VICReg variance term fixed it. Run this early; catching collapse at epoch 50 is much cheaper than catching it at epoch 200.

**Surgical unfreezing** (`data/load_backbone.py`): MoCo/Barlow/ImageNet unfreeze layer2+. SparK unfreezes layer3+ only — layer2 was already adapted during U-Net decoder pretraining. BatchNorm always frozen (stats from 112k NIH images are worth keeping).

**Patient-level split** (`finetune/_data.py`): train/val split by PatientID, not by image. A patient with multiple X-rays in both splits leaks anatomy — the backbone would have seen that patient's chest during pretraining. Patient-level isolation prevents this.

**EXCLUDE_LABELS** (`finetune/_data.py`): diseases with large public labeled datasets (pneumonia, pleural effusion, cardiomegaly, etc.) are excluded. The point is evaluating label efficiency on findings with no existing public dataset. See CLAUDE.md for the full list and rationale.

---

## Checkpoints

Pretrained backbones in `outputs_cloud/`:

| Path | Method | Backbone | Epoch |
|------|--------|----------|-------|
| `moco-v3/best.pt` | MoCo + VICReg | ResNet50 | 774 |
| `barlow/best.pt` | BarlowTwins | ResNet18 | 191 |
| `spark/best.pt` | SparK | ResNet50 | 199 |

Load any checkpoint with `data/load_backbone.py` — auto-detects the method from the checkpoint dict and returns a frozen feature extractor ready for downstream evaluation.

---

## Cloud Setup

L4 GPU (22GB VRAM), 16 vCPU, 64GB RAM — ~$0.70/hr on-demand.
All pretraining ran within the $300 Google Cloud free trial.

| Method | Batch size | Time | Cost |
|--------|-----------|------|------|
| MoCo | 384 | ~63h | ~$44 |
| BarlowTwins | 512 | ~15h | ~$10 |
| SparK | 256 | ~20h | ~$14 |
| DINO | 64 | ~1h (killed) | ~$1 |
