# Design Reference — PadChest Fine-Tuning Evaluation

Rationale for every major decision in `finetune/finetune_padchest.py` and `finetune/count_padchest.py`.

---

## Dataset Pre-Resize to 256×256 (`data/resize_datasets.py`)

All dataset images have been permanently resized to 256×256 8-bit grayscale PNG in-place.

**Why:**  `dataset.py` already downsizes images to 256×256 at cache-load time (`_CACHE_SIZE = 256`). The original full-resolution files were only ever used to produce this 256×256 array — the training pipeline never sees the original resolution. Pre-resizing makes this permanent, eliminating the per-run conversion overhead and dramatically reducing storage:

| dataset | original | after resize | reduction |
|---------|----------|--------------|-----------|
| NIH (112k images) | ~22 GB | 3.1 GB | **7×** |
| PadChest (29k images) | ~170 GB | 856 MB | **~200×** |

PadChest's extreme reduction is because originals are ~1572×1556 16-bit (2 bytes/px) vs the cached 256×256 8-bit (1 byte/px) — roughly 38× fewer pixels × 2× fewer bytes per pixel = ~76× raw, plus PNG compression on the smaller uniform images.

**Normalisation:** The script min-max normalises each image before quantising to uint8. This correctly handles 16-bit PadChest PNGs (which may not use the full 0-65535 range) and is consistent with how `_load_gray()` in `data/viz/domain_gap.py` handles mixed sources.

**5 corrupt PadChest files** (truncated or unreadable) were skipped — these are broken in the original zips and would have caused errors at training time anyway.

**Idempotent:** Re-running the script detects already-256px files and skips them.

---

## Why the Augmentation Strategy Doubles as Cross-Hospital Transfer

The real-world scenario this project demonstrates: a small hospital somewhere has a handful of labeled X-rays of a rare disease and wants a classifier. They can't collect thousands of labeled examples. They have slightly different X-ray equipment than whatever large public dataset exists. SSL pretraining on the large public dataset (NIH) should still help.

The augmentation choices made for MoCo-v3 (and inherited by BarlowTwins/SparK) were framed as "X-ray physics simulation" but are directly equivalent to **simulating cross-machine, cross-hospital variation**:

| augmentation | X-ray physics framing | cross-hospital framing |
|---|---|---|
| brightness jitter 0.4→0.8 | simulates kVp (tube voltage) variation | different hospitals use different exposure settings |
| contrast jitter 0.4→0.8 | simulates window/level variation | different detectors have different dynamic range |
| gaussian_noise std=0.1 | simulates X-ray quantum (Poisson) noise | different detectors have different noise characteristics |
| aggressive crop [0.08, 1.0] | forces local feature learning | different hospitals position patients differently, different fields of view |
| removed saturation/hue/grayscale | no-ops on grayscale X-rays | correct — all chest X-rays are grayscale regardless of machine |

By making the backbone invariant to these variations during pretraining, the SSL features transfer across the NIH→PadChest domain gap (US hospital → Spanish hospital, different equipment) without ever seeing PadChest data during pretraining. The PadChest evaluation is deliberately from a different country and institution to test this exact scenario.

**This is the core scientific claim**: SSL pretraining on large unlabeled public datasets produces features robust enough to transfer to a small, different-institution labeled dataset for rare disease classification — because the augmentation strategy taught the model to ignore the equipment-specific variation and focus on pathological structure.

### Measured NIH → PadChest Domain Gap

Quantified via `data/viz/domain_gap.py` (500 images per dataset, PA/AP views only, raw pixel values without per-image normalisation):

| metric | NIH | PadChest PA/AP |
|--------|-----|----------------|
| mean intensity | 0.518 | 0.508 |
| std (contrast) | 0.254 | 0.269 |
| p5 | 0.031 | 0.000 |
| p95 | 0.882 | 0.961 |
| **mean shift** | — | **0.010** |
| **std ratio** | — | **0.944** |

**The pixel-level domain gap is negligible** (mean shift 0.010, well below the 0.1 notable threshold). NIH and PadChest PA/AP chest X-rays are nearly identically distributed at the pixel level.

*Note: an earlier (incorrect) measurement reported a mean shift of 0.226. This was caused by two bugs: (1) lateral views included in PadChest (which look very different from PA), and (2) per-image min-max normalisation stretching every image to [0,1] and erasing real brightness differences.*

**Implication for the transfer argument**: the cross-hospital transfer challenge is not about pixel-level brightness/contrast differences — NIH and PadChest images are essentially the same at that level once you compare apples to apples (PA vs PA). The actual transfer challenge is purely about **label scarcity**: PadChest has very few labeled examples of rare findings. The augmentation strategy is still justified for learning robust, location-invariant features, but the "cross-scanner brightness correction" framing overstates the actual pixel distribution gap.

---

## "Rare Diseases" — Clarification of Project Framing

The project describes itself as demonstrating label efficiency on "rare diseases." This needs qualification.

**Most passing classes are clinically common diseases:**
- COPD signs, cardiomegaly, aortic elongation, air trapping, scoliosis, pleural effusion, pneumonia — these are all prevalent conditions affecting large fractions of the population. They appear label-scarce in our dataset only because we have 5 of 54 total PadChest zips (~9% of the full dataset).

**Truly clinically rare diseases exist in the label tail:**
- Pulmonary fibrosis, miliary opacities, tuberculosis sequelae, pulmonary mass — these have 1-5 examples total even across 5 zips, and will never pass any reasonable threshold. They are genuinely rare.

**The accurate project framing is "label-scarce settings," not "rare diseases."**
The scientific contribution is showing that self-supervised pretraining on unlabeled X-rays improves performance when only a small number of labeled examples are available — regardless of whether the disease itself is clinically rare or common. A hospital might have thousands of COPD X-rays but only 10 labeled examples of a specific rare finding they care about. Pretraining helps in both cases.

The "rare disease" framing is a motivating use case, not a strict constraint on the evaluation. The evaluation measures label efficiency — how much better pretrained features are compared to random init when labeled data is scarce — which is the same underlying property whether the disease is common or rare.

---

## Current PadChest Label Counts (8 zips: 13, 15, 16, 21, 22, 23, 27, 35)

25,760 images across 1,223 patients. Common diseases (pneumonia, pleural effusion, cardiomegaly, nodule, atelectasis, laminar atelectasis, emphysema, scoliosis, copd signs, aortic elongation) are now in EXCLUDE_LABELS in both scripts — they have large public labeled datasets and do not demonstrate label efficiency.

### Classes passing threshold (MIN_TRAIN=15, MIN_VAL=8)

| class | train | val | total | verdict |
|-------|------:|----:|------:|---------|
| air trapping | 99 | 28 | 127 | questionable — COPD-adjacent, functional sign |
| vertebral degenerative changes | 50 | 16 | 66 | questionable — extremely common in elderly |
| calcified granuloma | 44 | 13 | 57 | **keep** — prior TB/histoplasma, label-scarce |
| callus rib fracture | 46 | 10 | 56 | **keep** — healed rib fractures, label-scarce |
| costophrenic angle blunting | 48 | 8 | 56 | questionable — sign of pleural effusion (excluded) |
| hiatal hernia | 42 | 10 | 52 | **keep** — retrocardiac finding, label-scarce |
| interstitial pattern | 37 | 12 | 49 | **keep** — ILD indicator, label-scarce |
| chronic changes | 38 | 10 | 48 | questionable — non-specific, not a disease entity |
| hemidiaphragm elevation | 33 | 8 | 41 | **keep** — phrenic nerve/subphrenic disease, label-scarce |
| increased density | 29 | 9 | 38 | questionable — non-specific radiological descriptor |
| bronchiectasis | 23 | 10 | 33 | **keep** — tram-track lines on CXR, no other adult public dataset |
| fibrotic band | 17 | 9 | 26 | **keep** — linear scarring, label-scarce |

**7 clean rare disease classes**: calcified granuloma, callus rib fracture, hiatal hernia, interstitial pattern, hemidiaphragm elevation, bronchiectasis, fibrotic band.

**5 questionable classes** (to consider excluding): air trapping, vertebral degenerative changes, costophrenic angle blunting, chronic changes, increased density.

### Near-misses (below threshold, worth targeting with more zips)

| class | train | val | total |
|-------|------:|----:|------:|
| reticular interstitial pattern | 21 | 2 | 23 |
| kyphosis | 18 | 6 | 24 |
| pulmonary mass | 17 | 2 | 19 |
| pulmonary fibrosis | 9 | 2 | 11 |
| ground glass pattern | 3 | 1 | 4 |
| miliary opacities | 7 | 1 | 8 |

---

## Train/Val Split — Why 80/20 Patient-Level

**Why patient-level, not image-level:**
A patient can have multiple X-rays taken on different dates. If some of their images go to train and others to val, the backbone has already seen that patient's anatomy (rib cage shape, heart size, spinal curvature) during training and will partially "recognise" it at eval time, inflating accuracy. Splitting by PatientID guarantees zero patient overlap between splits.

**Why 80/20:**
- Train needs at least 20 labeled examples per class to support 20-shot sampling across 5 independent trials (5 × 20 = 100 draws, needs a pool larger than 20 to avoid repetition)
- Val needs enough examples for a reliable accuracy estimate — chosen as MIN_VAL=6 (see below)
- 80/20 is the standard ratio; going higher (e.g. 90/10) would leave too few val patients for rare diseases

**Why val sets are small for rare diseases:**
The split is 20% of *patients*, not images. A rare disease appearing in only 12 patients gives 20% × 12 = ~2 val patients, each with 1-2 images = 3-4 val images total. This is a fundamental data scarcity problem — the only real fix is downloading more zips. Lowering MIN_VAL from 8 → 6 was a pragmatic compromise to include more disease classes without new data.

---

## Image Filters — Why Only PA/AP Single-Label

**Projection filter (PA/AP only):**
PadChest contains frontal (PA = posteroanterior, AP = anteroposterior) and lateral views. Lateral views show anatomy from the side — a completely different visual perspective. Mixing them would require the model to learn two different anatomical representations and classify across them, which adds noise without adding useful signal for the downstream task (clinical diagnosis is done on frontal views).

**Single-label filter:**
Multi-label images (e.g. "cardiomegaly, pleural effusion") are ambiguous for classification — the model can't know which label caused which visual feature. A supervised classifier trained on multi-label images using a single-label loss would receive inconsistent gradients. Single-label images give a clean one-to-one mapping between image and class.

---

## Excluded Label Categories — Why These Are Not Diseases

The goal is to measure whether the pretrained backbone has learned *pathological features* from unlabeled X-rays. Including non-disease labels would let the model cheat by recognising hardware or artifacts instead of tissue pathology.

**Implanted hardware and devices:**
`pacemaker`, `nsg tube`, `endotracheal tube`, `central venous catheter`, `chest drain tube`, `tracheostomy tube`, `electrical device`, `artificial heart valve`, `aortic endoprosthesis`, `humeral prosthesis`, `mammary prosthesis`, `osteosynthesis material`, `suture material`, `metal`

These are foreign objects inserted into the patient. A model can identify them by spotting bright metallic regions in the image — this requires no understanding of lung pathology and would produce misleadingly high accuracy.

**Surgical history:**
`sternotomy`, `mastectomy`, `surgery`, `surgery neck/breast/lung/heart/humeral`, `post radiotherapy changes`

These reflect past procedures, not a current finding. A sternotomy scar is visible as a wire pattern on the sternum — again, easy to spot without any pathological understanding.

**Radiological artifacts and normal variants:**
`nipple shadow`, `end on vessel`, `dai`

These are not findings — they are imaging artifacts or normal anatomical structures that appear ambiguous on X-ray. Including them would train the classifier to identify camera/positioning artifacts.

**Non-specific labels:**
`normal`, `unchanged`, `exclude`, `suboptimal study`

`normal` is excluded because it is not a disease. `unchanged` means "same as prior study" — not a standalone diagnosis. `exclude` and `suboptimal study` indicate the image was flagged as unusable.

---

## Why Common Diseases Were Excluded from "Rare" Targeting

The diseases dropped from the rare disease target list were excluded because large publicly available labeled datasets already exist for them. Using them as the downstream evaluation would not demonstrate label efficiency — a researcher could simply fine-tune on one of those datasets directly without any SSL pretraining.

| disease | why excluded | public labeled dataset |
|---------|-------------|----------------------|
| Pneumonia | extremely common, major Kaggle competition | RSNA 2018 (26k labeled X-rays) |
| Pleural effusion | common in heart failure/cancer | CheXpert (75k positive), NIH (13k) |
| Cardiomegaly | standard finding in cardiac workup | NIH ChestX-ray14 (2,776 labeled) |
| Nodule | dedicated screening programs exist | LUNA16, NODE21, JSRT (all large) |
| Atelectasis | one of the most common X-ray findings | NIH (11,559 labeled) |
| Emphysema | common COPD component | NIH (2,516 labeled) |
| Scoliosis | structural, not a lung disease | large orthopedic datasets |
| COPD signs | most prevalent chronic lung disease | multiple large cohort studies |
| Aortic elongation | age-related, not rare | common in elderly population studies |

The point of using PadChest is specifically to evaluate on findings where labeled data is genuinely scarce — either because the disease is rare, or because it has not been the focus of large annotation efforts. Bronchiectasis, pulmonary fibrosis, pulmonary mass, and tuberculosis fit this criterion.

---

## Alternative Sources for Rare Chest Disease X-Rays

If more labeled rare disease data is needed beyond PadChest, these are the best sources:

**Directly downloadable / Kaggle:**
- **SIIM-ACR Pneumothorax Segmentation** (Kaggle) — ~12,000 X-rays with pneumothorax labels and pixel-level masks. Best available source for pneumothorax.
- **VinBigData Chest X-ray** (Kaggle) — 18,000 Vietnamese hospital X-rays with 28 disease labels including bronchiectasis, pulmonary fibrosis, consolidation, nodule. Underused dataset.
- **TBX11K** — 11,200 X-rays specifically for tuberculosis detection, with healthy/sick/active-TB/latent-TB splits. Best TB source.
- **Montgomery County + Shenzhen Hospital TB datasets** — small (138 + 662 images) but clean TB-labeled X-rays, widely cited in literature.

**Require registration / institutional access:**
- **PLCO (Prostate, Lung, Colorectal, Ovarian Cancer Screening Trial)** — NIH-funded, large chest X-ray cohort with lung cancer findings. Requires application.
- **NLST (National Lung Screening Trial)** — CT-based but includes X-ray data with lung cancer outcomes.
- **BRAX** — Brazilian hospital dataset, 40k X-rays with 14 CheXpert labels. Useful for domain diversity.

**Why PadChest is still the best choice for this project:**
PadChest has 174 fine-grained labels (vs 14 in NIH, 14 in CheXpert) — it's the only public dataset with labels specific enough to find bronchiectasis, pulmonary mass, and miliary opacities as distinct classes. The other datasets either use coarse labels or focus on a single disease. PadChest's breadth makes it uniquely suited for multi-class rare disease evaluation.

---

## Full Rare Disease Dataset Survey (sourced from web search)

Datasets with labeled rare chest findings under 400 images per class:

| dataset | diseases from target list | rare class counts | access | format |
|---------|--------------------------|-------------------|--------|--------|
| **VinDr-CXR** | ILD, pneumothorax, lung cavity, pulmonary fibrosis | Lung cavity=21, Pneumothorax=58, ILD=152 | PhysioNet (free, requires credentialing) | DICOM |
| **Shenzhen TB + 2022 annotations** | Tuberculosis, cavitation, miliary TB | Miliary=6, Cavity=45, all findings <165 | Fully open, no registration | PNG + JSON masks |
| **Montgomery County TB** | Tuberculosis | 58 TB positive (138 total) | Fully open | PNG + lung masks |
| **JSRT** | Pulmonary nodule/mass | 154 nodule images (100 malignant, 54 benign) | Free registration | TIFF/PNG |
| **ChestX-Det10** | Fibrosis, mass, nodule, pneumothorax (bounding boxes) | ~hundreds per class from 3,543 total | Free (GitHub) | PNG + JSON |
| **TBX11K** | Tuberculosis | 1,200 TB positive with bounding boxes | Free | PNG 512x512 |
| **VinDr-PCXR** | Bronchiectasis, mediastinal tumor (pediatric) | Very few per class (exact counts need download) | PhysioNet credentialed | DICOM |
| **SIIM-ACR Pneumothorax** | Pneumothorax (pixel masks) | 2,669 pneumothorax | Kaggle account | DICOM + RLE masks |
| **NIH ChestX-ray14** | Fibrosis (1,686), hernia (227), mass, pneumothorax | Hernia=227 under 400; others above | Free, no registration | PNG |
| **MIMIC-CXR-LT / CXR-LT** | Fibrosis, mass, pneumothorax, lung lesion | 12 classes under 1,000; designed for long-tail research | PhysioNet credentialed + MIMIC DUA | JPG |

**Key findings:**
- **Bronchiectasis**: PadChest and VinDr-PCXR (pediatric) are the only public sources. No adult bronchiectasis CXR dataset exists.
- **Miliary TB**: Shenzhen annotations only (6 cases globally in public data). Essentially impossible to evaluate on.
- **Lung cavity / cavitation**: VinDr-CXR (21) and Shenzhen (45) — both extremely small.
- **Bone metastasis on CXR**: No public labeled dataset found anywhere.
- **Ground glass pattern**: Primarily in COVID-19 datasets; PadChest is the only non-COVID source.
- **Mediastinal mass**: No clean adult CXR dataset labels this explicitly. PadChest only.

**Best supplementary downloads for our target diseases:**
1. **Shenzhen TB annotations** — free, open, adds cavitation + miliary labels
2. **VinDr-CXR** — PhysioNet registration, adds ILD (152) and pneumothorax (58) with expert bounding boxes
3. **Montgomery TB** — free, open, 58 clean TB cases for TB evaluation

---

## MoCo-v3 PadChest Results (10 zips, 8 classes, ep304)

8 classes: bronchiectasis, calcified granuloma, callus rib fracture, fibrotic band, hemidiaphragm elevation, hiatal hernia, interstitial pattern, reticular interstitial pattern. Chance = 12.5%.

### Probe mode (frozen backbone + logistic regression)

| shots | pretrained acc | random acc | gap | pretrained AUC | random AUC |
|------:|---------------|-----------|-----|----------------|------------|
| 1 | 21.2% ± 2.6% | 15.7% ± 4.8% | +5.5pp | 0.560 | 0.537 |
| 5 | 21.6% ± 3.8% | 15.1% ± 1.6% | +6.5pp | 0.623 | 0.559 |
| 10 | 28.6% ± 2.7% | 17.8% ± 1.7% | +10.8pp | 0.647 | 0.568 |
| 20 | 30.2% ± 3.5% | 19.4% ± 2.1% | +10.8pp | 0.665 | 0.569 |

### Finetune mode (layers 2-4 unfrozen, prototype head init, 100 epochs)

| shots | pretrained acc | random acc | gap | pretrained AUC | random AUC |
|------:|---------------|-----------|-----|----------------|------------|
| 1 | 21.8% ± 2.0% | 11.6% ± 2.9% | +10.2pp | 0.570 | 0.517 |
| 5 | 22.2% ± 3.9% | 17.3% ± 2.8% | +4.9pp | 0.614 | 0.546 |
| 10 | 28.8% ± 6.0% | 17.1% ± 6.5% | +11.7pp | 0.663 | 0.571 |
| 20 | **34.5% ± 4.6%** | 14.7% ± 5.5% | **+19.8pp** | 0.696 | 0.554 |

### Interpretation

**Finetune is better than probe at 20-shot** (+4.3pp: 34.5% vs 30.2%). With enough labeled data, adapting layers 2-4 to PadChest visuals meaningfully improves features. At 1-10 shot, finetune and probe are equivalent for the pretrained model — too few examples to benefit from backbone adaptation.

**Finetune hurts random init at 20-shot** (19.4% → 14.7%). Without pretrained structure, gradient fine-tuning on 160 total examples just overfits. This validates the SSL pretraining claim: the gap between pretrained and random is *larger* in finetune mode at 20-shot (+19.8pp) than in probe mode (+10.8pp).

**AUC gap widens consistently with shots** (0.023 → 0.064 → 0.079 → 0.096 in probe mode), showing the pretrained model gets progressively better at ranking with more data while random init plateaus.

---

## Threshold Values — MIN_TRAIN and MIN_VAL

| parameter        | value | rationale |
|------------------|------:|-----------|
| MIN_TRAIN        |    15 | supports up to 20-shot × 5 trials with some pool headroom |
| MIN_VAL          |     6 | minimum for a meaningful held-out accuracy estimate; lowered from 8 to include more rare classes |
| val_frac         |  0.20 | standard 80/20; balances train pool size vs val reliability |

MIN_VAL=6 means accuracy is estimated over 6 examples per class. This is low but acceptable for a relative comparison (pretrained vs random init) — both models are evaluated on the same 6 examples so the comparison is still fair even if the absolute accuracy numbers have high variance.

---

## Fine-Tuning Improvements (`finetune/finetune_padchest.py`)

### Two evaluation modes: probe vs finetune

The script now supports `--mode probe|finetune|both`.

**Probe** (original): fully frozen backbone → logistic regression head. Fast; good for comparing many checkpoints.

**Finetune**: gradient-based fine-tuning with layer-wise unfreezing + prototype head init. Slower but expected to give higher accuracy, especially at 10-20 shot where there's enough signal to adapt the backbone.

### Why layer-wise unfreezing (surgical fine-tuning)

Unfreezing the entire backbone on 15-50 training examples causes catastrophic forgetting — the backbone overwrites its SSL features to memorise the tiny training set. The surgical fine-tuning literature shows that selectively unfreezing only the later layers gives better transfer:

| method type | unfrozen layers | reasoning |
|-------------|----------------|-----------|
| Contrastive (MoCo, BarlowTwins, DINO) | layer2, layer3, layer4 | Early layers encode low-level texture already invariant to domain; mid-to-late layers encode task-specific semantics that benefit from PadChest adaptation |
| Restorative (SparK) | layer3, layer4 | Reconstruction forces layer3/4 to encode fine-grained spatial detail useful for identifying subtle findings; layers 1/2 already capture domain-invariant texture |

BatchNorm layers stay frozen throughout regardless of stage — with only 15-50 training examples, updating BN running statistics would corrupt the pretraining distribution learned over 112k images.

### Why prototype head initialisation

A randomly initialised linear head requires many gradient steps to learn the correct direction in feature space. With only N-shot examples, it may never converge. Initialising the head weights as the per-class mean L2-normalised feature vector (the "prototype") puts the head immediately in the right region of feature space — the classifier starts already knowing "class X looks like this on average." This is the cosine classifier approach from the few-shot learning literature.

In practice: prototype init + 100 gradient epochs outperforms random init + 1000 epochs when N ≤ 20.

### Better metrics: F1 and AUC-ROC

The original script reported accuracy only. With 6-7 classes and 6-17 val examples each, class imbalance can cause accuracy to be misleading (a model predicting the most common class gets 25% "for free").

- **Macro F1**: averages F1 per class with equal weight regardless of class size. Penalises ignoring small classes.
- **Macro AUC-ROC (OvR)**: measures ranking quality — whether the model assigns higher probability to the correct class. More robust than accuracy at small val sizes.

Both are now reported alongside accuracy for every shot count.

### Why `few_shot_probe` and `finetune_padchest` are different tools

`data/eval/few_shot_probe.py` evaluates on NIH (same domain as pretraining) — it's a diagnostic to monitor SSL training quality during a run. `finetune/finetune_padchest.py` evaluates on PadChest (different institution, different country) — it's the actual experiment measuring cross-domain label efficiency. Only `finetune_padchest` results go in the paper.

---

## Cloud Training — Difficulties Encountered

### BarlowTwins on L4 GPU

**1. pin_memory=True crashes with Python 3.14**
All non-MoCo dataloaders (`barlow/data.py`, `spark/data.py`, `dino/data.py`) had `pin_memory=True` hardcoded. MoCo had already fixed this but the fix wasn't propagated. On Python 3.14, pin_memory causes a CUDA initialization error that manifests as `CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)` — misleadingly looks like a GPU OOM but the GPU is actually empty. Fix: set `pin_memory=False` in all dataloaders.

**2. Both views need gradient graphs simultaneously (BarlowTwins memory) — 5 OOM attempts**
BarlowTwins computes loss over two views of the same image, both requiring full gradient graphs through the backbone. This is fundamentally 2× the activation memory of MoCo (which uses `no_grad` on the key encoder). Attempts in order:

| attempt | batch_size | result | memory used |
|--------:|----------:|--------|-------------|
| 1 | 512 | OOM (Conv2d, 148 MiB needed) | 21.92 GiB / 21.95 GiB |
| 2 | 384 | OOM (BatchNorm, 246 MiB needed) | 21.76 GiB / 21.95 GiB |
| 3 | 320 | OOM (BatchNorm, 246 MiB needed) | 21.76 GiB / 21.95 GiB |
| 4 | 256 | OOM (2 MiB needed) | 21.55 GiB / 21.95 GiB |
| 5 | 512 + grad checkpointing | **success** | ~10 GiB |

Key insight: the memory barely changed between bs=512 and bs=256 (only ~400 MB difference) because ~21 GB was fixed overhead from two full ResNet50 activation graphs — batch size scaling was tiny relative to the fixed cost. Gradient checkpointing on the 4 ResNet layer blocks reduced activation memory ~5x by recomputing intermediates during backward instead of storing them.

**3. /dev/shm exhaustion with many workers**
With `num_workers=14` and `prefetch_factor=4`, the DataLoader queues 14×4=56 batches in shared memory simultaneously. At bs=512, each batch is ~1.2 GB (two views) = 68 GB in /dev/shm, plus 7 GB for cache_in_ram = ~75 GB total, far exceeding the default /dev/shm limit. Fix: reduced `num_workers=4` and `prefetch_factor=2` in all cloud dataloaders. With `cache_in_ram=True` there is no disk I/O bottleneck so fewer workers are sufficient — confirmed GPU utilization stays at 100% with 4 workers.

**4. ResNet50 too slow with gradient checkpointing**
Even with checkpointing, ResNet50 + BarlowTwins was 6+ seconds/batch = ~112 hours for 300 epochs. Gradient checkpointing adds ~30% compute overhead by recomputing activations during backward. Fix: switched BarlowTwins to ResNet18 (~4x faster forward+backward, lower checkpointing recomputation cost). ResNet18 justified for homogeneous chest X-ray data — downstream performance gap was only 1.7pp at 50-shot vs ResNet50 MoCo.

**5. GPU availability zone issue**
L4 GPU was unavailable in us-central1-a. `gcloud compute instances move` command has been removed from gcloud CLI. Fix: used machine image approach — `gcloud compute machine-images create` then `gcloud compute instances create --source-machine-image` in a new zone with available capacity.

### SparK on L4 GPU

**Expected: single-view = similar memory to MoCo. Actual: OOM at bs=384 and bs=256.**

SparK's U-Net decoder uses skip connections from all 4 encoder stages (f1, f2, f3, f4) simultaneously. This means ALL intermediate ResNet50 feature maps must coexist in memory during the forward pass — much more than MoCo which only keeps the final layer output. Same gradient checkpointing fix as BarlowTwins was needed.

| attempt | batch_size | result |
|--------:|----------:|--------|
| 1 | 384 | OOM — decoder GELU (588 MiB needed) |
| 2 | 256 | OOM — same error, same memory usage |
| 3 | 256 + grad checkpointing on encoder stages | **success** |

---

## SparK Design Decisions

### Why SparK over MAE
MAE (He et al. 2022) requires a ViT backbone. ViTs have no spatial inductive biases — they learn patch relationships entirely from attention, which needs ImageNet-scale data (1M+ images). On 112k chest X-rays a ViT underfit badly. SparK (Tian et al. 2023) adapts masked modeling to ResNet50, keeping the same backbone as MoCo/BarlowTwins/DINO for a fair comparison. `ssl_methods/mae/` was written but deleted — SparK supersedes it entirely.

### Standard convolutions vs sparse convolutions
The original SparK paper uses sparse convolutions that truly ignore masked patch regions. Our implementation zeroes masked patches but uses regular Conv2d. This causes **information leakage**: adjacent visible patches partially bleed information into masked regions through the CNN receptive field. Effects:
- Loss drops fast (task is easier than true SparK — 1.0 → 0.17 by step 1600)
- Encoder doesn't get forced to learn true long-range context
- Features develop more slowly (50-shot still near random at ep6, 8% at ep25)

This is a known trade-off documented in the model code. True sparse convolutions would require a custom CUDA kernel. The standard conv approximation is sufficient for comparison purposes since the features still improve with training.

### Decoder capacity (dec_dim=256)
A lightweight decoder (256 hidden dims) was chosen to minimise GPU memory. The decoder is discarded after pretraining — only the encoder matters. Larger dec_dim (512+) would give stronger gradient signal to the encoder but would OOM on L4 given the skip connection memory pressure. 256 is a reasonable compromise.

### mask_ratio=0.60 vs MAE's 0.75
Standard convolutions leak context across masked boundaries, making the task slightly easier than true SparK. Mask ratio was lowered from 75% to 60% to partially compensate — fewer visible patches means the model sees less leaked context per masked region. In practice the difference is small.

### Why SparK is theoretically well-suited for homogeneous medical images — and why it still underperformed

The theoretical argument for masked image modeling on medical data is compelling:

**Contrastive methods struggle with homogeneous data.** MoCo and BarlowTwins push different images apart in embedding space. For that gradient signal to be useful, different images must look meaningfully different. On chest X-rays — same modality, same anatomy, same grayscale layout — the global visual differences between images are small. All negatives in the contrastive queue are "almost the same" image, which weakens the learning signal. This is exactly what killed DINO here (see DINO section below).

**Masked image modeling is indifferent to inter-image homogeneity.** The task is "reconstruct the missing 60% of *this* image from its visible 40%." It doesn't compare images against each other at all. And crucially, it forces the encoder to reconstruct every patch — including the tiny, subtle findings (a 2% bright spot that is a nodule, a faint linear opacity that is a fibrotic band) that contrastive augmentation would randomly crop out and treat as noise.

| | Contrastive (MoCo) | SparK |
|---|---|---|
| Homogeneous images | Weak negatives — reduces gradient signal | Fine — task is purely per-image |
| Small local findings | May crop out, treated as augmentation noise | Must reconstruct every patch |
| Dataset scale needed | Works well at 112k | Needs 500k+ to shine |

**Why it underperformed in practice at 112k images:** The reconstruction task is indirect — pixels are reconstructed, and only *hopefully* the encoder learns useful features along the way. At small scale, the model can partially solve reconstruction via shallow local texture copying (especially with standard convolution leakage), without learning deep semantic features. Contrastive methods give a direct, efficient gradient signal per step — "do these match or not?" — that works well even at 112k. SparK's mean_cos plateaued at 0.953 from ep80 onward because once the reconstruction loss converged, there was no remaining gradient signal to drive embedding diversity.

**The result is a genuine finding, not a failure:** SparK underperforming MoCo-v3 (9.8% vs 18.5% at 50-shot) at 112k images is consistent with what the literature predicts. For small-to-medium scale medical imaging datasets, contrastive pretraining remains the better choice. Generative methods require scale.

---

### Reconstruction quality as a diagnostic (not an eval metric)
Reconstruction loss and visual quality are training signals, not downstream metrics. The correct eval is still few-shot probe + collapse monitor — same as MoCo and BarlowTwins. Expected reconstruction quality timeline:
- ep1-25: gray blobs, no anatomy — normal
- ep50-100: rough rib/heart/lung outlines should appear
- ep150-200: cleaner anatomy, visible pathological structure

If anatomy is still absent at ep100, the model found a trivial blur-average shortcut and features will plateau.

### Why our reconstructions look patchier than MAE paper demos

SparK reconstructions at ep161 show visible patch grid artifacts that MAE paper figures don't. This is not due to the medical imaging dataset — there are four concrete reasons:

**1. Patch size: 32px vs 16px (biggest factor)**
MAE on ImageNet uses 16×16 patches on 224×224 = 14×14 = 196 patches. Our SparK uses 32×32 = 7×7 = 49 patches. Fewer, larger patches means fewer, more visible seams. The 32px choice was forced by L4 memory constraints — 16px patches would quadruple the number of patches and the decoder's memory footprint.

**2. ViT global attention vs ResNet local receptive field**
MAE's ViT encoder sees the entire image via self-attention — every masked patch prediction is globally consistent with every other patch. Our ResNet encoder processes locally — adjacent patch predictions are made from limited receptive fields and don't "know" each other's absolute brightness. This is SparK's fundamental architectural limitation vs MAE: local processing produces locally-consistent but globally-inconsistent predictions.

**3. Training scale**
MAE paper figures are typically from models trained 800–1600 epochs on 1.28M ImageNet images (~1–2 billion total patch predictions). At ep161 on 112k images we've made ~800M patch predictions — roughly half MAE's minimum budget, and the model is far from fully converged.

**4. Grayscale uniform backgrounds expose discontinuities**
ImageNet's colorful, textured natural images visually mask patch boundary discontinuities — the eye is drawn to content. Chest X-rays are uniform grey; any inter-patch intensity jump is immediately visible against the flat background.

**norm_pix_loss is the root cause of inter-patch brightness jumps:** each patch's target is independently normalised to zero-mean unit-variance before the loss is computed. The model learns to predict each patch in its own coordinate system, not in a globally consistent brightness space. Adjacent patches can be perfectly predicted in their own normalised spaces while still having different absolute brightness values at their shared boundary. This is irreducible without changing the training objective. A post-processing Gaussian blur (sigma=2) on the reconstruction and soft mask blending (sigma=3) on the composite reduce the visual artifact without changing the underlying model.

**MAE paper demos are also cherry-picked** — their supplementary figures show the best examples, not random samples. Random samples from MAE at early training look very similar to ours.

---

## Why DINO Failed on This Dataset

DINO was attempted but never produced a useful checkpoint. The root cause is structural: DINO's learning mechanism is fundamentally mismatched to homogeneous datasets.

### How DINO learns

DINO maintains a teacher network (slow EMA of student) that assigns soft probability distributions over learned prototypes. The student sees a different augmented view and tries to match the teacher's distribution. The cross-entropy between teacher and student distributions is the loss signal.

This works on ImageNet because a dog and a car produce genuinely different teacher distributions — the teacher assigns high probability to very different prototypes. The student gets a rich, class-specific gradient signal.

### Why it breaks on chest X-rays

Chest X-rays are all the same modality, same anatomy, same grayscale layout. The visual differences between a normal lung and a lung with early interstitial pattern are subtle — a slight texture change in the parenchyma. The teacher network, looking at two augmented crops of two different X-rays, produces nearly identical prototype distributions for both. The centering buffer (which subtracts a running mean to prevent collapse) interprets this near-uniform output as drift and suppresses it further. The student receives near-zero gradient signal.

Local crops (96px patches) made this worse: a 96px crop of a chest X-ray is almost always homogeneous lung texture with no discriminative content. The teacher assigns the same distribution to every local crop regardless of what image it came from.

### Hyperparameter fixes attempted (none solved the root cause)

| change | what it fixed | why insufficient |
|--------|--------------|-----------------|
| out_dim 2048 → 256 | teacher probabilities became sharper | still near-uniform across images |
| center_momentum 0.9 → 0.99 | centering adapted more slowly | didn't change the signal quality |
| teacher_temp warmup 0.07 → 0.04 over 30ep | teacher less diffuse early on | still couldn't distinguish similar images |
| teacher_momentum_start 0.996 → 0.99 | teacher learned faster early | didn't address homogeneity |
| n_local_crops = 0 | removed uninformative local crop noise | helped but underlying problem remained |

**Bottom line:** DINO requires inter-image visual diversity to produce meaningful teacher distributions. MoCo sidesteps this — it only asks "are these two views from the same image?" which is a well-defined binary signal regardless of how similar different images look to each other. On homogeneous medical data, MoCo's simpler contrastive objective is fundamentally more robust than DINO's prototype-matching approach.

### Quantitative proof of failure

**Training loss:** Flatlined at 5.42 from epoch 0 through epoch 15. The best checkpoint is epoch 0 — the model never improved at all. Loss did not decrease by a single meaningful step.

**Collapse monitor (ep1 → ep10 → ep15):**

| epoch | std   | mean_cos | eff_rank |
|------:|-------|----------|----------|
| 1     | 0.146 | 0.974    | 438.8    |
| 10    | 0.137 | 0.976    | 411.9    |
| 15    | 0.130 | 0.978    | 396.0    |

Both trends are moving in the wrong direction: mean_cos is **increasing** (embeddings becoming more similar over time, not more diverse) and eff_rank is **decreasing** (losing effective dimensions). DINO is actively collapsing — training makes the features progressively worse.

**Few-shot probe (ep15 checkpoint vs random init):**

| shots | DINO ep15 | random init | gap |
|------:|-----------|-------------|-----|
| 1     | 3.0 ± 0.4% | 7.4 ± 2.7% | **−4.4pp** |
| 5     | 5.7 ± 5.5% | 11.5 ± 9.0% | **−5.8pp** |
| 10    | 7.3 ± 8.2% | 3.7 ± 0.7% | +3.6pp |
| 20    | 7.1 ± 6.0% | 6.0 ± 4.2% | +1.1pp |
| 50    | 2.9 ± 1.6% | 3.5 ± 1.1% | **−0.6pp** |

Chance = 6.7% (15 classes). The DINO pretrained model scores **below chance** at 1-shot (3.0%) and 50-shot (2.9%), and is **worse than random init** at 1, 5, and 50-shot. SSL training actively destroyed the features — random initialisation is a better starting point than 15 epochs of DINO. The extreme variance (±8.2% at 10-shot) confirms the embeddings carry no stable signal; the linear classifier is fitting noise.

---

## RAM Cache & Shared Memory (`data/dataset.py`)

### Why this is the most important performance decision in the pipeline

On the cloud VM (L4, 16 vCPUs, 14 dataloader workers), without caching all 14 workers simultaneously hit the SSD to read images. Each epoch over 112k images = 112k random reads, multiplied by 14 workers = I/O saturation, GPU idle.

The fix is to load everything into a shared memory tensor once at startup:

```python
def _load_gray256(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((_CACHE_SIZE, _CACHE_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

# In UnlabeledChestXrayDataset.__init__, if cache_in_ram=True:
stacked = np.stack(arrays, axis=0)          # (112120, 256, 256) uint8
self._cache = torch.from_numpy(stacked).share_memory_()
```

`.share_memory_()` is the key call. It pins the tensor in OS shared memory so all 14 worker processes can read from it with zero copies — no pickling, no IPC overhead. 112k × 256×256 × 1 byte ≈ 7GB, fits comfortably in the VM's 64GB RAM.

### Two views in __getitem__

```python
img = Image.fromarray(self._cache[idx].numpy(), mode="L").convert("RGB")
return self.transform(img), self.transform(img)
```

`self.transform` is called **twice on the same image**. Because `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` etc. are stochastic, each call produces a different result. These two independently augmented views are the positive pair for contrastive learning — the model must learn representations invariant to those augmentations.

### Why `.convert("RGB")`

The ResNet50 backbone's first conv layer has shape `[64, 3, 7, 7]` — it expects 3-channel input. Converting grayscale to RGB triplicates the single channel into 3 identical channels. This is wasteful (3× data, same information) but necessary to use ImageNet-pretrained weights without restructuring the model. The backbone adapts within a few epochs.

---

## MoCo Training Loop (`ssl_methods/moco/train.py`)

MoCo pretraining is split across three files with a clean separation of concerns. `data.py` builds the DataLoader — it applies a chest X-ray specific augmentation pipeline (aggressive crop, brightness/contrast jitter, gaussian noise) to each image **twice independently**, producing two differently-distorted views `(x_q, x_k)` that serve as the positive pair for contrastive learning. `model.py` defines the architecture: two identical ResNet50 encoders where `encoder_q` is trained by backprop and `encoder_k` is a slow exponential moving average shadow that never gets gradients, plus a 65,536-slot ring buffer queue of stored key vectors; on each forward pass the query is compared against its matching key (positive, should be similar) and all 65k queued keys (negatives, should be dissimilar), producing logits that cross entropy drives toward column 0 — this is InfoNCE. `train.py` runs the loop: SGD with cosine LR decay and linear warmup optimises only `encoder_q`, mixed precision halves memory, the VICReg variance term penalises any of the 128 projection dimensions whose std drops below 1.0 across the batch to prevent dimensional collapse, and every batch logs step-level loss components to wandb while every epoch saves a checkpoint containing the full training state — model, optimizer, scheduler, scaler, and config — so training can resume exactly where it left off after a VM shutdown.

### Optimizer — SGD not Adam

SGD with momentum is used deliberately, not by default. Adam adapts the learning rate per-parameter based on gradient history, which can interfere with the EMA dynamics of the momentum encoder. SGD with a cosine schedule is empirically more stable for large-batch contrastive training. Only `encoder_q.parameters()` are passed to the optimizer — `encoder_k` is excluded because it is updated by EMA, not gradient descent. Passing `model.parameters()` would cause the optimizer to fight against the EMA update every step.

### Learning Rate Schedule — Warmup + Cosine Decay

Linear warmup for `warmup_epochs` then cosine decay to zero. The warmup matters because at epoch 0 the queue is full of randomly initialised keys — taking large gradient steps against meaningless negatives pushes weights in random directions. Starting with a small LR gives the queue time to fill with real encoder-consistent keys before aggressive updates begin. Cosine decay stays high longer than linear decay and drops off more gradually, keeping the model learning aggressively through mid-training before tapering off.

### Mixed Precision — `autocast` + `GradScaler`

`autocast` runs the forward pass in float16, halving memory and speeding up matrix multiplications. Float16 has a smaller numeric range than float32, so gradients can underflow to zero. `GradScaler` compensates by multiplying the loss by a large scale factor before backward, dividing back before the optimizer step, and automatically adjusting the scale factor based on whether overflow is detected.

### Variance Loss (VICReg term)

`variance_loss(q_raw)` takes the batch of unnormalised projection vectors `(N, 128)`, computes the per-dimension standard deviation across the batch, and applies a hinge penalty to any dimension whose std falls below 1.0. This prevents dimensional collapse — the failure mode where InfoNCE alone allows the encoder to output nearly the same value in every dimension for every image, technically minimising the loss but producing useless features. The `1e-4` epsilon inside `torch.sqrt` prevents NaN gradients when a dimension has exactly zero variance.

### Checkpointing Strategy

Three checkpoints written simultaneously every epoch: `latest.pt` (overwritten every epoch — crash recovery), `best.pt` (only written when epoch loss improves — used for evaluation), and `epoch_N.pt` every N epochs (swept by `collapse_monitor.py` to plot embedding diversity over training). Every checkpoint stores model, optimizer, scheduler, scaler, global_step, best_loss, wandb_run_id, and config — the full state needed to resume seamlessly, including continuing the same wandb plot rather than starting a new one.

### Logging — wandb and tqdm

`wandb.log` sends per-step loss components (`loss/step`, `loss/infonce`, `loss/var`) using `global_step` as the x-axis rather than wandb's internal counter, so resumed runs append to the same plot without overlap. `loss/infonce` and `loss/var` are logged separately so their individual contributions are visible — if total loss improves but `infonce` is stuck, the variance term is dominating. `tqdm` displays a live terminal readout of the same numbers as a heartbeat while training runs in the background via `nohup`.

---

## `ssl_methods/moco/model.py` — Architecture Summary

`model.py` implements MoCo v2: two identical ResNet encoders where only one (`encoder_q`) is trained by backprop, while the other (`encoder_k`) is a slow-moving shadow that updates via exponential moving average. Both encoders end in a 2-layer MLP projection head that maps 2048-dim backbone features down to 128-dim for contrastive learning. On each forward pass, a query image and an augmented key image are encoded separately, L2-normalised onto the unit sphere, then compared — the query is dotted against its matching key (positive pair, should score high) and against 65,536 stored keys from a ring buffer queue (negatives, should score low). These scores are assembled into a logit matrix and fed to cross entropy with labels always pointing to column 0, which is the InfoNCE loss. After each forward pass the fresh key vectors are written into the queue, overwriting the oldest entries, so the model always has a large and approximately consistent bank of negatives to contrast against without needing a massive batch size.

---

## Augmentation Pipeline (`ssl_methods/moco/data.py`)

Every transform in the pipeline has a specific justification for chest X-rays. None of them are copied blindly from ImageNet defaults.

**`RandomResizedCrop(224, scale=(0.08, 1.0))`**
Crops a random rectangle covering between 8% and 100% of the image area, then resizes it to 224×224. The wide scale range forces the model to recognise findings at all zoom levels — an 8% crop is a tiny patch of lung texture, a 100% crop is the whole chest. This is the most important augmentation for learning local features. Standard ImageNet uses `(0.2, 1.0)` — we went lower because chest X-rays have fine-grained local findings (nodules, fibrotic bands) rather than object-level features that are visible at any crop size.

**`RandomHorizontalFlip()`**
Mirrors the image left-right with 50% probability. Anatomically, the left and right lungs are symmetric enough that this is valid — the model should recognise a fibrotic band regardless of which lung it's in. Would be inappropriate for modalities where left-right asymmetry is diagnostically meaningful (e.g. ECG, brain MRI with lateralised lesions).

**`RandomRotation(degrees=10)`**
Rotates up to ±10°. Simulates real patient positioning variation — patients aren't always perfectly straight in the scanner. Small rotation makes the model invariant to this without introducing anatomically implausible orientations (a 90° rotation would look nothing like a real X-ray).

**`RandomApply([ColorJitter(brightness=0.8, contrast=0.8)], p=...)`**
Randomly applies brightness and contrast jitter. Brightness maps to tube kVp variation (how much voltage drives the X-ray beam — different hospitals use different settings). Contrast maps to radiologist window/level settings (different technicians adjust the display differently). Both are 0.8, doubled from the standard ImageNet value of 0.4, because X-ray acquisition variance is larger than natural photo variance. No saturation or hue — those are no-ops on grayscale input where all three RGB channels are identical after `convert("RGB")`. Wrapping in `RandomApply` means it's skipped entirely some fraction of the time, so the model also sees clean unmodified images.

**`RandomApply([GaussianBlur(kernel_size=...)], p=...)`**
Simulates motion blur from patient breathing during the scan, or defocus from detector calibration issues. Applied randomly so the model sees both sharp and blurred versions and learns features that survive mild blurring.

**`ToTensor()`**
Converts the PIL image to a float32 tensor and scales pixel values from [0, 255] to [0, 1]. Required before any tensor operations.

**`Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`**
Subtracts the ImageNet RGB channel means and divides by their standard deviations. These are technically wrong for X-rays (which have different pixel distributions) but practically fine — the backbone only needs its expected input scale to be roughly correct at initialisation, and the batch norm layers adapt the internal statistics within a few epochs.

**`GaussianNoise(std=0.1)`** *(appended last if configured)*
Custom transform that adds zero-mean Gaussian noise to the normalised float tensor. Simulates X-ray quantum (Poisson) noise — the speckle pattern that appears in low-dose acquisitions from insufficient photon counts. Applied after `Normalize` so the noise is added in the model's working float space, not to the raw uint8 pixel values. The only augmentation in the pipeline with no ImageNet equivalent — entirely X-ray-specific.
