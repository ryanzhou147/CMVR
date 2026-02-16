"""Display a 25x25 grid of sample chest X-rays from the dataset.

Usage:
  uv run python scripts/show_samples.py [--data-dir datasets/nih-chest-xrays] [--split train]
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

DEFAULT_DATA_DIR = "datasets/nih-chest-xrays"
GRID = 25
TOTAL = GRID * GRID


def find_images(data_dir: Path, split: str | None) -> list[Path]:
    """Get image paths, optionally filtered by split file."""
    if split:
        split_file = data_dir / f"{split}.txt"
        if split_file.exists():
            filenames = split_file.read_text().strip().splitlines()
            # Images are spread across images_001..images_012 subdirs
            all_pngs = {p.name: p for p in data_dir.glob("**/*.png")}
            return [all_pngs[f] for f in filenames if f in all_pngs]

    return sorted(data_dir.glob("**/*.png"))


def main():
    parser = argparse.ArgumentParser(description="Show sample chest X-ray grid")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default=None, help="train/val/test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paths = find_images(data_dir, args.split)
    print(f"Found {len(paths)} images")

    random.seed(args.seed)
    sample = random.sample(paths, min(TOTAL, len(paths)))

    fig, axes = plt.subplots(GRID, GRID, figsize=(30, 30))
    fig.suptitle(
        f"Sample Chest X-rays ({args.split or 'all'}) — {len(sample)} images",
        fontsize=20,
    )

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(sample):
            img = Image.open(sample[i]).convert("L")
            ax.imshow(img, cmap="gray")

    plt.tight_layout()
    out_path = data_dir / "sample_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
