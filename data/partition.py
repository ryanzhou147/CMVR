"""Partition NIH Chest X-rays into train/val/test splits (80/10/10).

Usage:
  uv run python -m data.partition [--data-dir DATA_DIR] [--seed SEED]

Writes train.txt, val.txt, test.txt to the data directory,
each containing one image filename per line.
"""

import argparse
import random
from pathlib import Path

DEFAULT_DATA_DIR = "datasets/nih-chest-xrays"


def partition(data_dir: Path, seed: int = 42):
    """Split all images into 80/10/10 train/val/test."""
    image_files = sorted(
        p.name for p in data_dir.glob("**/*.png")
    )

    if not image_files:
        raise RuntimeError(f"No .png images found in {data_dir}")

    print(f"Found {len(image_files)} images in {data_dir}")

    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = image_files[:n_train]
    val = image_files[n_train:n_train + n_val]
    test = image_files[n_train + n_val:]

    for name, split in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
        out_path = data_dir / name
        out_path.write_text("\n".join(split) + "\n")
        print(f"  {name}: {len(split)} images")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Partition NIH Chest X-rays into train/val/test")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    partition(Path(args.data_dir), seed=args.seed)


if __name__ == "__main__":
    main()
