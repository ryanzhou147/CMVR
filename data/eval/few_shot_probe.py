"""Few-shot linear probe evaluation for MoCo pretrained encoder.

Freezes the pretrained backbone, trains a logistic regression classifier on
N labeled examples per class, and evaluates on the val split. Compares
against a randomly initialised encoder as a baseline.

Single-label images only are used so this is a clean multiclass problem.

Usage:
  uv run python -m data.few_shot_probe --checkpoint outputs/moco/best.pt
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.load_backbone import load_feature_extractor, method_name

FEW_SHOT_NS = [1, 5, 10, 20, 50]
N_TRIALS = 5   # repeat each N-shot with different random seeds and average
MIN_TEST_PER_CLASS = 10

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return _TRANSFORM(Image.open(self.paths[idx]).convert("RGB"))


def load_split(data_dir: Path, txt_file: str) -> set[str]:
    """Load a set of image filenames from a split txt file."""
    return set(Path(data_dir / txt_file).read_text().strip().splitlines())


def load_single_label_samples(
    data_dir: Path, split_names: set[str]
) -> dict[str, list[Path]]:
    """Return {label: [paths]} for single-label images in the given split."""
    index = {p.name: p for p in data_dir.glob("images*/**/*.png")}
    label_to_paths: dict[str, list[Path]] = {}
    with open(data_dir / "Data_Entry_2017.csv") as f:
        for row in csv.DictReader(f):
            name = row["Image Index"]
            if name not in split_names or name not in index:
                continue
            labels = row["Finding Labels"].split("|")
            if len(labels) != 1:
                continue  # skip multi-label
            label = labels[0]
            label_to_paths.setdefault(label, []).append(index[name])
    return label_to_paths


@torch.no_grad()
def extract_features(
    model: nn.Module, paths: list[Path], device: torch.device
) -> np.ndarray:
    """Extract L2-normalised backbone features for a list of image paths."""
    dataset = ImageDataset(paths)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    feats = []
    for imgs in loader:
        feats.append(model(imgs.to(device)).cpu().numpy())
    return normalize(np.concatenate(feats))




def run_probe(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    test_feats: np.ndarray,
    test_labels: np.ndarray,
    n_shot: int,
    rng: random.Random,
) -> float:
    """Sample n_shot per class, fit logistic regression, return test accuracy."""
    classes = np.unique(train_labels)
    idx = []
    for cls in classes:
        cls_idx = np.where(train_labels == cls)[0].tolist()
        idx.extend(rng.sample(cls_idx, min(n_shot, len(cls_idx))))

    X_train, y_train = train_feats[idx], train_labels[idx]
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    return clf.score(test_feats, test_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot linear probe for CMVR checkpoints")
    parser.add_argument("--checkpoint", type=str, default="outputs/barlow/best.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")

    # Load split membership
    train_names = load_split(data_dir, "train.txt")
    val_names = load_split(data_dir, "val.txt")

    print("Loading single-label images...")
    train_by_label = load_single_label_samples(data_dir, train_names)
    val_by_label = load_single_label_samples(data_dir, val_names)

    # Keep only classes with enough val examples for reliable evaluation
    classes = sorted(
        c for c in train_by_label
        if c in val_by_label and len(val_by_label[c]) >= MIN_TEST_PER_CLASS
    )
    print(f"Classes ({len(classes)}): {classes}")
    label_to_idx = {c: i for i, c in enumerate(classes)}

    train_paths = [p for c in classes for p in train_by_label[c]]
    train_labels_raw = [c for c in classes for _ in train_by_label[c]]
    val_paths = [p for c in classes for p in val_by_label[c]]
    val_labels_raw = [c for c in classes for _ in val_by_label[c]]

    train_labels = np.array([label_to_idx[l] for l in train_labels_raw])
    val_labels = np.array([label_to_idx[l] for l in val_labels_raw])

    print(f"Train pool: {len(train_paths)} images | Val: {len(val_paths)} images")

    results = {}
    for name, is_random in [("Pretrained", False), ("Random init", True)]:
        print(f"\nExtracting features — {name}...")
        backbone = load_feature_extractor(checkpoint, device, random_init=is_random)

        train_feats = extract_features(backbone, train_paths, device)
        val_feats = extract_features(backbone, val_paths, device)

        accs = {}
        for n in FEW_SHOT_NS:
            trial_accs = []
            for trial in range(N_TRIALS):
                rng = random.Random(args.seed + trial)
                acc = run_probe(train_feats, train_labels, val_feats, val_labels, n, rng)
                trial_accs.append(acc)
            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            accs[n] = (mean_acc, std_acc)
            print(f"  {n:3d}-shot: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")

        results[name] = accs

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Pretrained": "#2196F3", "Random init": "#FF5722"}
    for name, accs in results.items():
        ns = list(accs.keys())
        means = [accs[n][0] * 100 for n in ns]
        stds = [accs[n][1] * 100 for n in ns]
        ax.plot(ns, means, marker="o", label=name, color=colors[name])
        ax.fill_between(ns,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=colors[name])

    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Few-shot linear probe — {len(classes)} NIH classes\n"
                 f"(epoch {checkpoint['epoch']+1}, {N_TRIALS} trials each)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(FEW_SHOT_NS)
    ax.set_xticklabels(FEW_SHOT_NS)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(f"{method_name(checkpoint)}_few_shot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
