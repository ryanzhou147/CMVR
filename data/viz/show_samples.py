"""Display a 5x5 grid of sample chest X-rays from the dataset.

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


def reservoir_sample(iterator, k: int, rng: random.Random) -> list:
    """Pick k items from an iterator without loading it all into memory."""
    result = []
    for i, item in enumerate(iterator):
        if i < k:
            result.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                result[j] = item
    return result


def sample_images(data_dir: Path, split: str | None, n: int, rng: random.Random) -> list[Path]:
    """Sample n image paths without loading the full list into memory."""
    if split:
        split_file = data_dir / f"{split}.txt"
        if split_file.exists():
            filenames = split_file.read_text().strip().splitlines()
            # Sample filenames first (cheap strings), then resolve paths
            chosen = rng.sample(filenames, min(n, len(filenames)))
            subdirs = sorted(data_dir.glob("images*")) or [data_dir]
            name_to_path: dict[str, Path] = {}
            for subdir in subdirs:
                for p in subdir.glob("*.png"):
                    if p.name in chosen:
                        name_to_path[p.name] = p
                    if len(name_to_path) == len(chosen):
                        break
                if len(name_to_path) == len(chosen):
                    break
            return [name_to_path[f] for f in chosen if f in name_to_path]

    return reservoir_sample(data_dir.glob("**/*.png"), n, rng)


def main():
    parser = argparse.ArgumentParser(description="Show sample chest X-ray grid")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default=None, help="train/val/test split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = random.Random(args.seed)
    sample = sample_images(data_dir, args.split, TOTAL, rng)
    print(f"Showing {len(sample)} images")

    thumb_size = (128, 128)

    fig, axes = plt.subplots(GRID, GRID, figsize=(10, 10))
    fig.suptitle(
        f"Sample Chest X-rays ({args.split or 'all'}) — {len(sample)} images",
        fontsize=20,
    )

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(sample):
            img = Image.open(sample[i])
            img.thumbnail(thumb_size)
            ax.imshow(img, cmap="gray")

    plt.tight_layout()
    out_path = data_dir / "sample_grid.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved grid to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
