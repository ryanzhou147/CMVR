# PadChest Binary Fine-Tune Results

Per-disease binary classification with partial backbone unfreeze + gradient updates.
10 trials per cell, AUC reported as mean ± std.

---

## Methodology and design rationale

### Why fine-tune at all if probing already works?

The probe establishes that SSL features have signal. Fine-tuning asks a different question: given a small labeled set, can we specialize those features further for a specific disease? In deployment, a radiologist would provide 20–50 confirmed examples of a rare disease they want to detect. Fine-tuning simulates that workflow. The expected result is that fine-tuning improves on the probe at higher shot counts (20–50) where there is enough signal to update the backbone meaningfully, while at 1–5 shots the probe may be competitive or better because fine-tuning risks overfitting.

### Why partial unfreeze instead of full fine-tuning?

With only 5–50 labeled examples, fully unfreezing all layers causes catastrophic forgetting — the backbone overwrites pretraining knowledge to fit the tiny labeled set. Early ResNet layers (stem, layer1) capture low-level X-ray features (edges, bone/soft tissue contrast, lung field boundaries) that required 112k images to learn. These do not need updating for a disease-specific task. Freezing them and only updating later layers preserves this knowledge while allowing the high-level semantic layers to adapt. This is supported empirically in Raghu et al. (2019), which showed that medical imaging transfer learning benefits from keeping early layers frozen, particularly in low-data regimes.

**Empirical support from Grad-CAM layer visualization:**

We ran multi-layer Grad-CAM on the frozen moco-v2 backbone (no fine-tuning) across 4 pulmonary mass val images and observed the following representational hierarchy:

- **layer1**: Diffuse, noisy activation across the entire image — ribs, lung edges, laterality markers. No spatial selectivity.
- **layer2**: Consistently activates the mediastinum (the vertical boundary between lung fields and the heart/great vessels) across all images. Generic structural anchoring, not disease-specific.
- **layer3**: Activation narrows to one or two lung zones. The model begins lateralizing toward the affected side.
- **layer4**: Tight focal hotspot landing in a single lung quadrant, consistent with the location of the mass in the original X-ray.

This directly supports the freeze/unfreeze boundary. layer1 and layer2 are doing generic work (edge detection, structural landmarks) that transfers without modification — updating them with 20–50 examples would only corrupt useful representations. layer3 and layer4 are where disease-specific spatial selectivity lives, and these are the layers we unfreeze. The Grad-CAM result gives visual confirmation that the SSL backbone has already built a coherent representational hierarchy from pretraining alone, and that fine-tuning only needs to sharpen the top of that hierarchy.

### Why does SparK unfreeze later than contrastive methods?

Contrastive methods (MoCo, BarlowTwins, DINO) push discriminative signal into deep layers — the earlier layers learn generic visual features and the later layers learn the task-specific representation. Layer2 in a contrastive backbone is relatively generic and safe to unfreeze.

SparK's pretraining task — reconstructing masked patches from context — requires spatial coherence at every layer, including layer2. The encoder must learn to preserve and propagate spatial structure through all stages to enable accurate reconstruction. Unfreezing layer2 on SparK with only 20–50 examples corrupts these spatial representations. Layer3 onwards carries higher-level semantics that can be adapted more safely.

### Why prototype initialization for the head?

With 5–20 training examples, random head initialization means the first gradient steps are mostly correcting a bad starting point rather than learning a decision boundary. Initializing head weights from per-class mean L2-normalized feature vectors (the class prototype, or centroid in feature space) gives the optimizer a geometrically sensible starting point. The prototype is already a reasonable classifier — it places the decision boundary between the class centroids — and gradient updates then refine it. This is the core idea from prototypical networks (Snell et al., 2017) and is standard in few-shot fine-tuning literature.

### Why keep BatchNorm frozen?

BatchNorm running statistics (mean and variance per channel) were estimated over 112k images during pretraining. Fine-tuning with batches of 10–100 images would update these statistics with high-variance estimates, corrupting the normalization that the pretrained weights expect. Freezing BatchNorm in eval mode throughout fine-tuning is standard practice when fine-tuning with small batches (He et al., 2019; standard in detectron2, timm). Only the affine parameters (gamma, beta) can optionally be unfrozen; here we freeze both for simplicity.

### Why AdamW with separate LRs for head and backbone?

The head is randomly initialized (relative to the task) and needs to move quickly — lr=1e-3. The backbone is already well-trained and should move slowly to avoid forgetting — lr=1e-4, one order of magnitude lower. This two-rate setup is standard in transfer learning. AdamW (Adam with decoupled weight decay) is preferred over SGD in low-data fine-tuning because it adapts per-parameter learning rates, which helps when different layers have very different gradient magnitudes.

### Why cosine LR decay?

With only 100 epochs and small batches, a step schedule wastes training at a high LR then drops too abruptly. Cosine annealing smoothly decays the LR from the initial value to near zero, which works well across the range of shot counts without per-shot tuning. Used in the original MoCo, BarlowTwins, and DINO fine-tuning protocols.

### Why balanced training AND balanced val?

Same reasoning as the probe. PadChest has roughly 1–5% prevalence of rare diseases in the wild. An unbalanced evaluation lets a trivial "always predict negative" baseline score >95% accuracy. Balancing train and val forces the model to actually learn the disease boundary. Train is balanced at exactly N positives + N negatives. Val uses all available positives + an equal number of sampled negatives, giving an unbiased AUC estimate at the target operating prevalence.

### Why augmentation during fine-tuning but not probing?

The probe uses the frozen backbone with no weight updates, so augmentation would only add noise to feature extraction without benefit. During fine-tuning, augmentation regularizes the backbone updates — with only 20–50 examples, the backbone would otherwise overfit to the specific crops and exposures in the training set. Augmentations are chosen for X-ray physics: random resized crop (simulates different FOV and collimation), horizontal flip (left/right anatomical mirroring is valid for most chest findings), small rotation (±10°, patient positioning variation), brightness/contrast jitter (exposure variation across scanners), and Gaussian blur (simulates resolution differences).

---

## Backbone architecture

### Why ResNet50?

ResNet50 was chosen for four reasons:

1. **Established SSL baseline**: MoCo, BarlowTwins, SimCLR, and most contrastive SSL papers use ResNet50 as the standard benchmark backbone. Using it makes our results directly comparable to the literature.

2. **Fits available hardware**: ResNet50 with batch_size=384 fits in 22GB VRAM (L4 GPU) with room for the SSL projection head and augmentation pipeline. ResNet101/152 would require batch_size reductions that hurt contrastive learning quality.

3. **2048-d feature space**: The avgpool output is 2048-d, which gives the linear probe and fine-tune head enough capacity to separate disease classes while remaining tractable with few examples. ViT-B/16 produces 768-d, ViT-L produces 1024-d — both are viable but require different SSL training recipes.

4. **SparK compatibility**: SparK's masked autoencoder was specifically designed around ResNet's hierarchical stage structure (stem→layer1→layer2→layer3→layer4). The U-Net decoder attaches skip connections at each stage boundary. This architecture doesn't translate directly to ViT backbones without significant redesign.

### What other backbones were available?

| Backbone | Params | Feature dim | Why not used |
|----------|--------|-------------|--------------|
| ResNet18 | 11M | 512 | Barlow local run used this — 4× smaller, ~1.7pp behind ResNet50 at 50-shot in NIH eval |
| ResNet50 | 25M | 2048 | **Used** |
| ResNet101 | 44M | 2048 | OOM at batch_size=384 on L4, would need batch_size≤192 hurting SSL quality |
| ViT-B/16 | 86M | 768 | Requires DINO-style or MAE training recipe; no direct MoCo/Barlow implementation in this codebase |
| ViT-L/16 | 307M | 1024 | Far too large for available hardware |
| DenseNet121 | 8M | 1024 | CheXNet backbone — strong medical imaging baseline but less studied under SSL |

ResNet18 results from the Barlow local run show it is competitive at high shot counts despite being 4× smaller, suggesting the feature dimension matters less than the pretraining quality past a certain model size.

### ResNet50 layer structure and what each stage learns

ResNet50 has 5 named stages. Understanding what each learns is what motivates the freeze/unfreeze decisions:

```
Stage       Output size    Parameters    What it learns
─────────────────────────────────────────────────────────────────────
stem        112×112        ~9k           Edges, gradients, basic textures
layer1      56×56          ~215k         Corners, blobs, simple patterns
layer2      28×28          ~1.2M         Textures, local structures (ribs, vessels)
layer3      14×14          ~7.1M         Object parts (lung fields, diaphragm shape)
layer4      7×7            ~14.3M        Semantic concepts (disease patterns)
avgpool     1×1            0             Global average pooling — no parameters
fc          —              ~2M           Task head — replaced entirely
─────────────────────────────────────────────────────────────────────
```

The freeze cut at layer2 (contrastive) / layer3 (SparK) is not arbitrary — it reflects where generic visual features end and domain/task-specific features begin. Raghu et al. (2019) showed empirically that in medical imaging, layers up to and including layer1 transfer with zero degradation from any source domain. Layer2 is the first stage worth considering for domain adaptation.

### Exact layers frozen and unfrozen

**MoCo / BarlowTwins / DINO / ImageNet:**
```
stem     ████ FROZEN    — 9k params locked
layer1   ████ FROZEN    — 215k params locked
layer2   ░░░░ UNFROZEN  — 1.2M params update (except BatchNorm stats)
layer3   ░░░░ UNFROZEN  — 7.1M params update (except BatchNorm stats)
layer4   ░░░░ UNFROZEN  — 14.3M params update (except BatchNorm stats)
head     ░░░░ UNFROZEN  — 2-class Linear(2048, 2), prototype-initialized
```

**SparK:**
```
stem     ████ FROZEN    — 9k params locked
layer1   ████ FROZEN    — 215k params locked
layer2   ████ FROZEN    — 1.2M params locked (spatial features from reconstruction)
layer3   ░░░░ UNFROZEN  — 7.1M params update (except BatchNorm stats)
layer4   ░░░░ UNFROZEN  — 14.3M params update (except BatchNorm stats)
head     ░░░░ UNFROZEN  — 2-class Linear(2048, 2), prototype-initialized
```

**BatchNorm exception**: within all unfrozen layers, BatchNorm running mean/variance stay frozen in eval mode. Only convolution weights and BatchNorm affine parameters (gamma/beta) are excluded from gradients. Running stats were computed over 112k images and would be corrupted by fine-tuning batches of 10–100.

---

## Exact fine-tuning setup per model

### MoCo-v1, MoCo-v2, MoCo-v3, BarlowTwins, ImageNet (contrastive group)

```
Frozen:   stem, layer1, layer2's BatchNorm stats
Unfrozen: layer2 weights, layer3, layer4, avgpool
Head:     nn.Linear(2048, 2), prototype-initialized
Optimizer: AdamW — backbone lr=1e-4, head lr=1e-3, weight_decay=0.01
Scheduler: CosineAnnealingLR(T_max=100)
Epochs:   100
Batch:    min(32, n_train_samples)
Augment:  RandomResizedCrop(224, scale=(0.08,1.0)), RandomHorizontalFlip,
          RandomRotation(10), ColorJitter(brightness=0.8, contrast=0.8) p=0.8,
          GaussianBlur(kernel=9) p=0.5
```

Rationale for unfreezing layer2: contrastive pretraining pushes discriminative signal
into layer3+, making layer2 relatively generic. With 20+ examples it can be safely
specialized to X-ray disease features.

### SparK (restorative/reconstructive group)

```
Frozen:   stem, layer1, layer2, layer3's BatchNorm stats
Unfrozen: layer3 weights, layer4, avgpool
Head:     nn.Linear(2048, 2), prototype-initialized
Optimizer: AdamW — backbone lr=1e-4, head lr=1e-3, weight_decay=0.01
Scheduler: CosineAnnealingLR(T_max=100)
Epochs:   100
Batch:    min(32, n_train_samples)
Augment:  same as above
```

Rationale for freezing layer2: SparK's masked reconstruction objective requires
spatial coherence through all encoder stages including layer2. Unfreezing it with
few examples corrupts the spatial feature maps that the pretraining relied on.

### Random init (control)

Same setup as the method group it mirrors (contrastive unfreeze schedule), but
backbone weights are randomly initialized — no pretraining. This is the floor:
if fine-tuning a random backbone approaches SSL fine-tuning performance, the
pretraining added little value.

---

## Results

### moco-v2 ep304

*Pending — run with:*
```
uv run python -m finetune.finetune_padchest --checkpoint outputs/moco-v2/best.pt --data-dir datasets/padchest --n 1 5 10 20 50 all
```
