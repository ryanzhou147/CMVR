# Evaluation Summary

All evaluations on NIH Chest X-rays (15 classes, 73k train / 9k val).
Primary metric: few-shot linear probe (logistic regression, 5 trials averaged).
KNN eval is not a reliable metric on this dataset — see CLAUDE.md.

**Narrative:** moco-v1 showed dimensional collapse during training → added VICReg variance term (v2) to prevent it.

## Run Directory Mapping

| eval name    | output dir             | backbone  | dataset | augs   | notes                              |
|--------------|------------------------|-----------|---------|--------|------------------------------------|
| moco-v1      | outputs/moco_cloud/    | ResNet50  | 112k    | weak   | baseline, ep264 final              |
| moco-v2      | outputs/moco-v2/       | ResNet50  | 112k    | strong | VICReg var_weight=1.0, ep304 final |
| barlow-cloud | outputs/barlow_cloud/  | ResNet18  | 112k    | strong | ep200 final, converged at ~ep125   |
| spark        | outputs/spark_cloud/spark/ | ResNet50  | 112k    | —      | masked autoencoder, ep161, plateauing |
| dino         | outputs/dino/          | ResNet50  | 112k    | strong | **FAILED** — loss flatlined at ep0, ran only 15 epochs |

---

## Few-Shot Probe (NIH, 15 classes)

| shots | moco-v1 ep264 | moco-v2 ep304 (VICReg) | barlow ep191 (best) | spark ep102   | dino ep15 ❌  | random        |
|------:|---------------|------------------------|---------------------|---------------|---------------|---------------|
| 1     | 9.4 ± 4.1%    | 9.3 ± 3.8%             | 9.1 ± 4.7%          | 8.5 ± 4.3%   | 3.0 ± 0.4%   | 7.8 ± 2.8%   |
| 5     | 11.2 ± 3.4%   | 11.8 ± 4.5%            | 9.2 ± 3.6%          | 8.8 ± 6.4%   | 5.7 ± 5.5%   | 10.0 ± 8.4%  |
| 10    | 9.6 ± 2.1%    | 9.1 ± 2.4%             | 7.8 ± 1.9%          | 3.5 ± 1.9%   | 7.3 ± 8.2%   | 5.8 ± 3.9%   |
| 20    | 14.6 ± 3.0%   | **16.4 ± 3.0%**        | 14.8 ± 4.0%         | 12.9 ± 6.2%  | 7.1 ± 6.0%   | 4.8 ± 1.7%   |
| 50    | 17.5 ± 1.4%   | **18.5 ± 1.4%**        | 16.8 ± 2.1%         | 10.5 ± 1.4%  | 2.9 ± 1.6%   | 4.4 ± 0.0%   |

Chance = 6.7% (15 classes).

**Notes:**
- moco-v1: ResNet50, weak augmentations, no VICReg, ep264 final
- moco-v2: ResNet50, strong X-ray augmentations, VICReg var_weight=1.0, ep304 final
- barlow: ResNet18 (4x smaller), strong augmentations, ep191 best (converged ~ep125) — only 1.7pp behind moco-v2 despite smaller backbone
- spark: ResNet50, masked autoencoder, ep161 — slowly improving but plateauing; 50-shot gap vs random grew from +6.0pp (ep102) to +6.1pp (ep161). mean_cos flatlined at 0.953 since ep80. 10-shot still below random.
- dino ❌: **FAILED** — loss flatlined at 5.42 from epoch 0, never learned. Pretrained is worse than random init at 1, 5, and 50-shot; scores below chance (6.7%) at 1-shot (3.0%) and 50-shot (2.9%). mean_cos increasing (0.974→0.978) and eff_rank decreasing — actively collapsing during training.

---

## Collapse Monitor

### moco-v1 (outputs/moco_cloud/, 112k images, weak augmentations, no VICReg)
| epoch | std   | mean_cos | eff_rank |
|------:|-------|----------|----------|
| 50    | 0.363 | 0.675    | 175.6    |
| 100   | 0.162 | 0.533    | 219.3    |
| 150   | 0.074 | 0.458    | 249.7    |
| 200   | 0.061 | 0.446    | 251.7    |
| 250   | 0.059 | 0.411    | 257.9    |
| 264   | 0.060 | 0.393    | 261.8    |

eff_rank still rising at ep264 but std is very low (0.06) — embeddings are nearly unit-norm with low spread across dimensions. Motivation for adding VICReg.

### moco-v2 (outputs/moco-v2/, 112k images, strong augmentations + VICReg var_weight=1.0)
| epoch | std   | mean_cos | eff_rank |
|------:|-------|----------|----------|
| 50    | 0.429 | 0.672    | 153.4    |
| 100   | 0.174 | 0.578    | 211.4    |
| 139   | 0.143 | 0.539    | 223.5    |

Still early (139/800 epochs). Higher std at all epochs vs v1 — VICReg is keeping embedding magnitudes more spread. eff_rank comparison will be meaningful at ep264+.

### spark (outputs/spark/, ResNet50, 112k images, masked autoencoder, 60% mask ratio)
| epoch | std   | mean_cos | eff_rank |
|------:|-------|----------|----------|
| 20    | 0.193 | 0.967    | 259.0    |
| 40    | 0.199 | 0.961    | 244.9    |
| 60    | 0.187 | 0.959    | 244.5    |
| 80    | 0.197 | 0.954    | 242.8    |
| 100   | 0.194 | 0.953    | 242.9    |
| 120   | 0.183 | 0.953    | 247.7    |
| 124   | 0.191 | 0.953    | 247.5    |

**Note: collapse monitor is not meaningful for SparK.** mean_cos is high (0.953) because SparK has no contrastive or variance loss to push embeddings apart — features cluster in a narrow cone in embedding space, which is expected for generative pretraining. eff_rank is stable around 245-260. The few-shot probe is the only honest quality signal for SparK.

### barlow-cloud (outputs/barlow_cloud/, ResNet18, 112k images, strong augmentations)
| epoch | std   | mean_cos | eff_rank |
|------:|-------|----------|----------|
| 25    | 0.228 | 0.581    | 161.9    |
| 50    | 0.187 | 0.527    | 179.9    |
| 75    | 0.174 | 0.497    | 190.2    |
| 100   | 0.167 | 0.474    | 196.6    |
| 125   | 0.162 | 0.475    | 200.8    |
| 150   | 0.163 | 0.462    | 201.3    |
| 175   | 0.161 | 0.463    | 202.3    |
| 200   | 0.161 | 0.461    | 202.4    |

eff_rank out of 512 (ResNet18) = 39% capacity at ep200. Converged at ~ep125 — no meaningful improvement after. No collapse. mean_cos steady decrease shows embeddings becoming more diverse throughout training.

---

## Method Selection Rationale

### SparK over MAE
MAE (Masked Autoencoders, He et al. 2022) is the standard masked image modeling method but requires a ViT backbone. ViTs have no inductive biases — they learn spatial relationships entirely from data via attention, which requires ImageNet-scale datasets (1M+ images) to work. On 112k chest X-rays a ViT would underfit badly and produce worse features than a CNN trained on the same data.

SparK (Tian et al. 2023) adapts masked modeling to work with ResNet50. This matters for three reasons:
1. **Same backbone as every other method** — ResNet50 for all four (MoCo, BarlowTwins, SparK, DINO) means all produce identical 2048-d GAP features and the few-shot probe comparison is fair.
2. **CNN inductive biases suit X-rays** — translation equivariance and local connectivity match chest X-ray structure, where anatomy is in consistent positions across images. The model doesn't need to learn "ribs are in a fixed location" from scratch.
3. **Works at smaller scale** — the reconstruction task (predict masked patches from context) is well-defined even with 11k images, unlike ViT attention which needs many images to learn meaningful global relationships.

Mask ratio is 60% rather than MAE's 75% because standard convolutions leak some context across masked patches — the task is slightly easier, so less masking is needed to maintain difficulty.

### BarlowTwins
BarlowTwins (Zbontar et al. 2021) is the simplest method in the set: no teacher network, no momentum queue, no centering buffer. The loss operates directly on the cross-correlation matrix between two augmented views' projections:
- **Diagonal → 1**: same image should produce the same embedding (invariance)
- **Off-diagonal → 0**: different embedding dimensions should be independent (decorrelation)

This directly prevents both collapse modes in one loss. It's the most explicit form of the decorrelation idea — comparable to what VICReg's covariance term does in MoCo-v2, but without any contrastive component at all. Including BarlowTwins lets us test whether a pure decorrelation objective matches or beats contrastive + VICReg on this dataset.

---

## DINO Training Notes (local run, plateauing issues)

DINO was never successfully trained to a useful checkpoint. Multiple hyperparameter changes were needed just to stop it plateauing:

1. **out_dim=2048 → 256** — large prototype dimension made the teacher distribution too diffuse. The centering buffer couldn't track it meaningfully, so the cross-entropy loss provided near-zero gradient signal. Dropping to 256 made teacher probabilities sharp enough to learn from.

2. **center_momentum=0.9 → 0.99** — 0.9 is a 10-step EMA. The center was adapting so fast it immediately cancelled out any teacher signal, leaving a near-uniform distribution. Slowing it to 0.99 (~100-step EMA) let the teacher actually guide the student.

3. **teacher_temp warmup (0.07 → 0.04 over 30 epochs)** — starting with a sharp teacher temp (0.04) at epoch 0 caused the teacher to instantly commit to arbitrary prototypes before the student had learned anything. Adding a warmup period gave the student time to develop reasonable representations first.

4. **teacher_momentum_start=0.996 → 0.99** — high initial momentum meant the teacher updated too slowly at the start, trailing the student by many steps. Lowering it let the teacher learn faster early on when both networks are random.

5. **n_local_crops=0** — 96px local crops of chest X-rays are mostly uniform lung field with no discriminative structure. They added noise to the loss without useful signal on the small local dataset.

**Bottom line:** DINO has many tightly coupled hyperparameters (teacher temp, centering momentum, EMA momentum, out_dim) that interact badly on small datasets. MoCo's simpler contrastive objective is more robust. DINO would likely be worth revisiting on the full 112k dataset.

**Why DINO is fundamentally harder on homogeneous datasets like chest X-rays:**
DINO learns by matching probability distributions over learned prototypes — the teacher assigns a soft target distribution and the student tries to reproduce it. This only works if different images produce meaningfully different distributions. On ImageNet, a dog and a car look completely different, so the teacher assigns clearly different prototypes and the student has a real signal to chase. Chest X-rays are all the same modality, same grayscale, same anatomy, same rough layout — the visual differences between classes (e.g. a subtle nodule vs. normal) are tiny and often invisible at the patch level. The teacher can't produce reliably different distributions for different images, so the centering mechanism interprets the near-uniform output as drift and suppresses it, and the student receives a near-zero learning signal. This is also why local crops were useless — a 96px crop of a chest X-ray is almost always just homogeneous lung texture with no discriminative content. MoCo sidesteps this entirely: it just asks "are these two augmented views of the same image?" which is a well-defined binary signal regardless of how similar the images look to each other.

---

## PadChest Fine-Tuning

### Disease selection rationale
No diseases were hand-picked. The class list is determined automatically by two filters:

1. **Image filters**: only PA/AP projection, only single-label images (multi-label images are ambiguous — the model can't know which label caused which feature). Hardware, devices, surgical history, and artifacts are excluded so the model must learn actual pathology, not implants.

2. **Count thresholds**: `MIN_TRAIN=15` (need enough examples to support up to 20-shot sampling across 5 trials) and `MIN_VAL=6` (minimum for a reliable held-out accuracy estimate). Whatever survives both filters becomes the evaluation set — no curation.

3. **Patient-level split**: train/val are split by PatientID (80/20), not by image. A patient can have multiple X-rays and if some went to train and others to val, the backbone would have already seen that patient's anatomy during pretraining, inflating val accuracy. Patient-level isolation prevents this.

**Why a label can fail even when train+val total > 23:**
The split is patient-level. A rare disease with only a few patients can end up with most of them in train by random chance, leaving val below MIN_VAL=6. Moving individual images across the split would break patient isolation and cause leakage. The correct fixes are: (a) download more zips to increase total examples, (b) lower MIN_VAL, or (c) increase val_frac to push more patients into val.

MIN_VAL was lowered from 8 → 6 to include nodule, laminar atelectasis, pleural effusion, etc. without needing more data.

### Current status (3 zips: 21, 23, 27 — 9674 images)

| label                         | train | val | total | in eval |
|-------------------------------|------:|----:|------:|---------|
| copd signs                    |  190  |  42 |  232  | YES     |
| scoliosis                     |   91  |  24 |  115  | YES     |
| cardiomegaly                  |   50  |   8 |   58  | YES     |
| air trapping                  |   46  |   9 |   55  | YES     |
| aortic elongation             |   31  |  12 |   43  | YES     |
| nodule                        |   26  |   6 |   32  | YES (MIN_VAL lowered to 6) |
| laminar atelectasis           |   24  |   6 |   30  | YES (MIN_VAL lowered to 6) |
| vertebral degenerative changes|   25  |   3 |   28  | no — val too small         |
| pleural effusion              |   21  |   3 |   24  | no — val too small         |
| chronic changes               |   18  |   5 |   23  | no — val too small         |

Fine-tuning eval not yet run at MIN_VAL=6. Run:
```bash
uv run python -m finetune.finetune_padchest --checkpoint outputs/moco_cloud/best.pt --data-dir datasets/padchest
uv run python -m finetune.finetune_padchest --checkpoint outputs/moco-v2/latest.pt --data-dir datasets/padchest
```
