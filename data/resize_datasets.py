"""Pre-resize all dataset images to 256×256 8-bit grayscale PNG in-place.

Why:
  cache_in_ram=True in dataset.py already downsizes every image to 256×256
  uint8 grayscale at startup.  This script makes that permanent on disk so:
    - No repeated resize on every run / cloud VM restart
    - PadChest ~1572×1556 16-bit  → 40× storage reduction per image
    - NIH 1024×1024 8-bit          → 16× storage reduction per image
  The resulting files are byte-for-byte equivalent to what cache_in_ram=True
  loads into memory, so training behaviour is unchanged.

The resize uses the same normalise-then-quantise approach as _load_gray() in
data/viz/domain_gap.py: min-max normalise the raw pixel array to [0, 1] then
scale to [0, 255] uint8.  This correctly handles 16-bit PadChest PNGs
regardless of whether they use the full 0-65535 range.

Usage:
    # Resize NIH only
    python -m data.resize_datasets --nih-dir datasets/nih-chest-xrays

    # Resize PadChest only
    python -m data.resize_datasets --padchest-dir datasets/padchest

    # Resize both
    python -m data.resize_datasets \
        --nih-dir datasets/nih-chest-xrays \
        --padchest-dir datasets/padchest

    # Preview only (no writes)
    python -m data.resize_datasets --nih-dir datasets/nih-chest-xrays --dry-run
"""

import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_DEFAULT_SIZE = 256  # must match _CACHE_SIZE in data/dataset.py


def _resize_one(args: tuple[Path, int]) -> tuple[Path, str]:
    """Resize a single image in-place.  Returns (path, status)."""
    path, target = args
    try:
        img = Image.open(path)

        # Already the right size and mode — skip (idempotent)
        if img.size == (target, target) and img.mode == "L":
            return path, "skip"

        # Normalise raw pixel values to [0, 255] uint8.
        # Works for 8-bit (NIH) and 16-bit (PadChest) alike.
        arr = np.array(img).astype(np.float32)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo + 1e-8)
        arr = (arr * 255).astype(np.uint8)

        out = Image.fromarray(arr).convert("L").resize(
            (target, target), Image.BILINEAR
        )

        # Write atomically: temp file in same dir, then rename
        dir_ = path.parent
        with tempfile.NamedTemporaryFile(
            dir=dir_, suffix=".png", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        out.save(tmp_path, format="PNG")
        os.replace(tmp_path, path)

        return path, "ok"

    except Exception as e:
        return path, f"error: {e}"


def _collect_pngs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.png"))


def _resize_dir(root: Path, target: int, dry_run: bool, workers: int) -> None:
    paths = _collect_pngs(root)
    print(f"  {root}: {len(paths):,} PNGs found")

    if dry_run:
        if paths:
            img = Image.open(paths[0])
            print(f"  Sample: {paths[0].name}  size={img.size}  mode={img.mode}")
        print("  --dry-run: no files written")
        return

    ok = skip = errors = 0
    work = [(p, target) for p in paths]
    with mp.Pool(workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(_resize_one, work, chunksize=64)):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                errors += 1
                print(f"  WARN {path.name}: {status}")
            if (i + 1) % 5000 == 0 or (i + 1) == len(paths):
                print(f"  {i+1:,}/{len(paths):,}  ok={ok}  skip={skip}  err={errors}")

    print(f"  Done: {ok} resized, {skip} already {target}px, {errors} errors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-resize dataset images to 256×256 8-bit grayscale")
    parser.add_argument("--nih-dir", default=None)
    parser.add_argument("--padchest-dir", default=None)
    parser.add_argument("--size", type=int, default=_DEFAULT_SIZE,
                        help=f"Target size (default {_DEFAULT_SIZE}, must match _CACHE_SIZE in dataset.py)")
    parser.add_argument("--workers", type=int, default=min(8, mp.cpu_count()),
                        help="Parallel workers (default min(8, cpu_count))")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files and show sample info; do not write anything")
    args = parser.parse_args()

    target = args.size
    if target != _DEFAULT_SIZE:
        print(f"WARNING: --size {target} differs from _CACHE_SIZE={_DEFAULT_SIZE} in dataset.py. "
              f"Update _CACHE_SIZE to match or the cache will re-resize at runtime.")

    if not args.nih_dir and not args.padchest_dir:
        parser.error("Specify at least one of --nih-dir or --padchest-dir")

    if not args.dry_run:
        print(f"Resizing images IN-PLACE to {target}×{target} 8-bit grayscale.")
        print(f"This is irreversible — originals will be overwritten.")
        print(f"Workers: {args.workers}")
        print()

    if args.nih_dir:
        print(f"NIH ({args.nih_dir}):")
        _resize_dir(Path(args.nih_dir), target, args.dry_run, args.workers)
        print()

    if args.padchest_dir:
        print(f"PadChest ({args.padchest_dir}):")
        _resize_dir(Path(args.padchest_dir), target, args.dry_run, args.workers)
        print()


if __name__ == "__main__":
    main()
