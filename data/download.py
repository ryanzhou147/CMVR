"""
  1. Create a Kaggle account at https://www.kaggle.com
  2. Add your API key to .env:
       KAGGLE_API_TOKEN=<your-key>
  3. Run: uv run python -m data.download [--dest DATA_DIR]
The dataset is ~45GB and contains 112,120 chest X-ray images.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DATASET_SLUG = "nih-chest-xrays/data"
DEFAULT_DEST = "datasets/nih-chest-xrays"


def download(dest: Path):
    """Download and extract the NIH Chest X-rays dataset."""
    load_dotenv()
    dest.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(dest.glob("**/*.png"))
    if len(existing_images) > 100_000:
        print(f"Dataset appears already downloaded ({len(existing_images)} images in {dest})")
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

    print(f"Downloading NIH Chest X-rays (~45GB) to {dest}...")
    print("This will take a while depending on your connection speed.")
    print()

    api.dataset_download_files(
        dataset=DATASET_SLUG,
        path=str(dest),
        unzip=True,
        quiet=False,
    )

    # Verify
    images = list(dest.glob("**/*.png"))
    print(f"Download complete: {len(images)} images in {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download NIH Chest X-rays from Kaggle")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST, help=f"Destination directory (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    dest = Path(args.dest)
    download(dest)

    print()
    print("Next step: update configs/default.yaml with:")
    print(f'  root_dir: {dest.resolve()}')


if __name__ == "__main__":
    main()
