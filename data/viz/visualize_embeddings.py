"""Visualize MoCo pretrained encoder embeddings via t-SNE.

Loads a checkpoint, extracts backbone features from labeled NIH images,
runs t-SNE to 2D, and plots coloured by disease label.

Usage:
  uv run python -m data.visualize_embeddings
  uv run python -m data.visualize_embeddings --checkpoint outputs/moco/best.pt --n 3000
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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ssl_methods.moco.model import MoCo

N_SAMPLES = 2000

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class LabeledDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, str]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return _TRANSFORM(img), label


def load_nih_labeled(data_dir: Path, n: int, rng: random.Random) -> list[tuple[Path, str]]:
    """Sample n images from the NIH CSV with their primary disease label."""
    index = {p.name: p for p in data_dir.glob("images*/**/*.png")}
    rows = []
    with open(data_dir / "Data_Entry_2017.csv") as f:
        for row in csv.DictReader(f):
            path = index.get(row["Image Index"])
            if path:
                primary = row["Finding Labels"].split("|")[0]
                rows.append((path, primary))
    return rng.sample(rows, min(n, len(rows)))


def extract_features(model: MoCo, samples: list[tuple[Path, str]], device: torch.device):
    """Run all images through the query encoder backbone, return (features, labels)."""
    # Temporarily strip the projection head so we get 2048-dim backbone features
    orig_fc = model.encoder_q.fc
    model.encoder_q.fc = nn.Identity()
    model.eval()

    dataset = LabeledDataset(samples)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            feats = model.encoder_q(imgs.to(device)).cpu().numpy()
            all_features.append(feats)
            all_labels.extend(labels)

    model.encoder_q.fc = orig_fc  # restore
    return np.concatenate(all_features), all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE of MoCo embeddings")
    parser.add_argument("--checkpoint", type=str, default="outputs/moco/best.pt")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    moco_cfg = config["moco"]

    model = MoCo(
        encoder_name=moco_cfg["encoder"],
        dim=moco_cfg["dim"],
        K=moco_cfg["queue_size"],
        m=moco_cfg["momentum"],
        T=moco_cfg["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"] + 1
    print(f"Loaded checkpoint from epoch {epoch}")

    print(f"Sampling {args.n} labeled images...")
    samples = load_nih_labeled(Path(args.data_dir), args.n, rng)
    print(f"Sampled {len(samples)} images")

    print("Extracting features...")
    features, labels = extract_features(model, samples, device)
    print(f"Features: {features.shape}")

    print("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=40, random_state=args.seed, max_iter=1000).fit_transform(features)

    # Plot
    unique_labels = sorted(set(labels))
    cmap = plt.colormaps["tab20"].resampled(len(unique_labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            color=cmap(label_to_idx[label]),
            label=label,
            alpha=0.5,
            s=8,
        )

    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_title(
        f"MoCo ResNet50 t-SNE — {len(samples)} NIH chest X-rays (epoch {epoch})",
        fontsize=13,
    )
    ax.axis("off")

    plt.tight_layout()
    out_path = Path("moco_tsne.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
