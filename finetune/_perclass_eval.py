"""Quick per-class accuracy breakdown at 20-shot, probe mode."""
import random
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from pathlib import Path

from finetune.finetune_padchest import load_padchest_splits, ImageDataset
from data.load_backbone import load_feature_extractor


@torch.no_grad()
def extract(model: nn.Module, paths, device) -> np.ndarray:
    loader = DataLoader(ImageDataset(paths), batch_size=64,
                        num_workers=0, pin_memory=False)
    feats = []
    for imgs in loader:
        feats.append(model(imgs.to(device)).cpu().numpy())
    return normalize(np.concatenate(feats))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("outputs/moco-v3/best.pt", map_location=device, weights_only=False)

    train_by_label, val_by_label = load_padchest_splits(Path("datasets/padchest"))
    classes = sorted(train_by_label)
    label_to_idx = {c: i for i, c in enumerate(classes)}

    train_paths = [p for c in classes for p in train_by_label[c]]
    train_labels = np.array([label_to_idx[c] for c in classes for _ in train_by_label[c]])
    val_paths   = [p for c in classes for p in val_by_label[c]]
    val_labels  = np.array([label_to_idx[c] for c in classes for _ in val_by_label[c]])

    print(f"\nVal examples per class:")
    for c in classes:
        print(f"  {c:<42s} train={len(train_by_label[c]):>3d}  val={len(val_by_label[c]):>3d}")

    for init_name, is_random in [("Pretrained", False), ("Random init", True)]:
        backbone = load_feature_extractor(ckpt, device, random_init=is_random)
        train_feats = extract(backbone, train_paths, device)
        val_feats   = extract(backbone, val_paths,   device)

        all_preds = []
        for t in range(5):
            rng = random.Random(42 + t)
            idx = []
            for cls in range(len(classes)):
                cls_idx = np.where(train_labels == cls)[0].tolist()
                idx.extend(rng.sample(cls_idx, min(20, len(cls_idx))))
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(train_feats[idx], train_labels[idx])
            all_preds.append(clf.predict(val_feats))

        preds = stats.mode(np.stack(all_preds), axis=0).mode.flatten()
        print(f"\n{'='*60}")
        print(f"{init_name} — 20-shot per-class (majority vote, 5 trials)")
        print(f"{'='*60}")
        print(classification_report(val_labels, preds, target_names=classes,
                                    digits=2, zero_division=0))


if __name__ == "__main__":
    main()
