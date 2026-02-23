"""Download the PadChest chest X-ray sample from Kaggle.

PadChest contains 174 radiographic labels from a Spanish hospital — many
findings have fewer than 20 examples, making it ideal for few-shot fine-tuning
after SSL pretraining on NIH.

Setup (same Kaggle credentials as NIH download):
  1. Add KAGGLE_API_TOKEN=<your-key> to .env (already done if NIH is downloaded)
  2. Run: uv run python -m data.download_padchest

This downloads the ~175MB Kaggle sample (~1000 images).
For the full 160k-image dataset (~1TB), register at:
  http://bimcv.cipf.es/bimcv-projects/padchest/

Usage:
  uv run python -m data.download_padchest
  uv run python -m data.download_padchest --dest datasets/padchest
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DATASET_SLUG = "raddar/padchest-chest-xrays-sample"
DEFAULT_DEST = "datasets/padchest"


def download(dest: Path) -> None:
    load_dotenv()
    dest.mkdir(parents=True, exist_ok=True)

    existing = list(dest.glob("**/*.png")) + list(dest.glob("**/*.jpg"))
    if len(existing) > 100:
        print(f"PadChest appears already downloaded ({len(existing)} images in {dest})")
        return

    if not os.environ.get("KAGGLE_API_TOKEN"):
        print("ERROR: Kaggle credentials not found.")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to API > Create New Token")
        print("  3. Add KAGGLE_API_TOKEN=<your-key> to .env")
        sys.exit(1)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading PadChest sample (~175MB) to {dest}...")
    api.dataset_download_files(
        dataset=DATASET_SLUG,
        path=str(dest),
        unzip=True,
        quiet=False,
    )

    images = list(dest.glob("**/*.png")) + list(dest.glob("**/*.jpg"))
    csvs = list(dest.glob("**/*.csv"))
    print(f"\nDownload complete: {len(images)} images, {len(csvs)} CSV file(s) in {dest}")
    if csvs:
        print(f"Label CSV: {csvs[0]}")
        print("\nFirst few lines of CSV:")
        with open(csvs[0]) as f:
            for i, line in enumerate(f):
                print(" ", line.rstrip())
                if i >= 3:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PadChest sample from Kaggle")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST)
    args = parser.parse_args()
    download(Path(args.dest))


if __name__ == "__main__":
    main()
