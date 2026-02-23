"""Augmentation viewer: display MoCo view pairs to inspect augmentation diversity.

Samples N random chest X-rays and renders each as a row of three panels:
  [original grayscale | augmented view 1 (x_q) | augmented view 2 (x_k)]

Use this to visually confirm whether the two views look sufficiently different.
If the views look nearly identical, the contrastive task is too easy and the
model will overfit to low-level statistics rather than semantic content.

Usage:
  uv run python -m data.augmentation_viewer
  uv run python -m data.augmentation_viewer --config configs/moco.yaml --n 12
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from ssl_methods.moco.data import get_moco_transforms

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor) -> np.ndarray:
    """Invert ImageNet normalisation so pixel values are in [0, 1] for display."""
    img = tensor.permute(1, 2, 0).numpy()
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)


def load_image_paths(data_dir: Path, n: int, rng: random.Random) -> list[Path]:
    paths = sorted(data_dir.glob("images*/**/*.png"))
    if not paths:
        raise RuntimeError(f"No PNG images found under {data_dir}")
    return rng.sample(paths, min(n, len(paths)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MoCo augmentation pairs")
    parser.add_argument("--config", type=str, default="configs/moco.yaml")
    parser.add_argument("--data-dir", type=str, default="datasets/nih-chest-xrays")
    parser.add_argument("--n", type=int, default=8, help="Number of images to show")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    transform = get_moco_transforms(config)
    paths = load_image_paths(Path(args.data_dir), args.n, rng)
    print(f"Loaded {len(paths)} images — rendering augmentation pairs...")

    fig, axes = plt.subplots(args.n, 3, figsize=(9, args.n * 3), squeeze=False)

    for col, title in enumerate(["Original", "View 1 (x_q)", "View 2 (x_k)"]):
        axes[0][col].set_title(title, fontsize=12, fontweight="bold")

    for row, path in enumerate(paths):
        img = Image.open(path).convert("RGB")

        # Apply the transform independently twice — same randomness source as training
        v1 = unnormalize(transform(img))
        v2 = unnormalize(transform(img))

        axes[row][0].imshow(img.convert("L"), cmap="gray")
        axes[row][1].imshow(v1)
        axes[row][2].imshow(v2)

        for ax in axes[row]:
            ax.axis("off")

        # Show filename on the left margin
        axes[row][0].set_ylabel(path.name, fontsize=7, rotation=0,
                                labelpad=65, va="center")

    plt.suptitle(f"MoCo augmentation pairs — {args.config}", fontsize=13, y=1.005)
    plt.tight_layout()

    out = Path("moco_augmentations.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
