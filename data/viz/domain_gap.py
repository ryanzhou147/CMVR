"""Visualise the domain gap between NIH and PadChest datasets.

Two panels:
  1. Sample images side-by-side — qualitative look at visual differences
  2. Pixel intensity histograms — quantify distribution shift
  3. Per-dataset statistics table printed to stdout

Optionally, if a checkpoint is provided, also plots a UMAP of backbone
features coloured by dataset — shows whether SSL features are domain-invariant.

Usage:
    # Pixel-level only
    uv run python -m data.viz.domain_gap \
        --nih-dir datasets/nih-chest-xrays \
        --padchest-dir datasets/padchest

    # With feature-level UMAP
    uv run python -m data.viz.domain_gap \
        --nih-dir datasets/nih-chest-xrays \
        --padchest-dir datasets/padchest \
        --checkpoint outputs/moco-v3/best.pt
"""

import argparse
import csv
import gzip
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.dataset import collect_image_paths  # noqa: F401 (used for NIH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gray(path: Path, size: int = 224) -> np.ndarray:
    """Load an image as grayscale float32 in [0, 1], resized to size×size.

    No per-image normalisation — raw pixel values divided by 255 so that
    the actual brightness distribution is preserved for comparison.
    """
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    img_pil = Image.fromarray((arr * 255).astype(np.uint8))
    img_pil = img_pil.resize((size, size), Image.BILINEAR)
    return np.array(img_pil, dtype=np.float32) / 255.0


def _collect_padchest_pa(root: Path) -> list[Path]:
    """Return PA/AP-only PadChest image paths using the CSV projection column.

    Falls back to all PNGs if no CSV is found (e.g. no labels file).
    """
    root = Path(root)
    gz_candidates = sorted(root.glob("*.csv.gz"))
    csv_candidates = sorted(root.glob("*.csv"))

    if not gz_candidates and not csv_candidates:
        print("  WARNING: no PadChest CSV found — including all views (may include laterals)")
        return sorted(root.rglob("*.png"))

    # Build a filename→path index for all images on disk
    disk_index = {p.name: p for p in root.rglob("*.png")}

    def open_csv():
        if gz_candidates:
            return gzip.open(gz_candidates[0], "rt", encoding="utf-8", errors="replace")
        return open(csv_candidates[0], encoding="utf-8", errors="replace")

    pa_paths = []
    with open_csv() as f:
        for row in csv.DictReader(f):
            if row.get("Projection", "").strip() not in ("PA", "AP"):
                continue
            image_id = row.get("ImageID", "").strip()
            if image_id and image_id in disk_index:
                pa_paths.append(disk_index[image_id])

    return sorted(set(pa_paths))


def _sample_paths(paths: list, n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return rng.sample(paths, min(n, len(paths)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Domain gap: NIH vs PadChest")
    parser.add_argument("--nih-dir", default="datasets/nih-chest-xrays")
    parser.add_argument("--padchest-dir", default="datasets/padchest")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional SSL checkpoint for feature-level UMAP")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Images to show per dataset in the sample grid")
    parser.add_argument("--n-hist", type=int, default=500,
                        help="Images to sample for histogram / statistics")
    parser.add_argument("--output", default="domain_gap.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---- Collect paths -------------------------------------------------------
    print("Collecting image paths...")
    nih_paths = collect_image_paths("nih", args.nih_dir)
    padchest_paths = _collect_padchest_pa(Path(args.padchest_dir))
    print(f"  NIH:      {len(nih_paths):,} images")
    print(f"  PadChest: {len(padchest_paths):,} PA/AP images")

    sample_nih = _sample_paths(nih_paths, args.n_samples)
    sample_pc  = _sample_paths(padchest_paths, args.n_samples)

    hist_nih = _sample_paths(nih_paths, args.n_hist)
    hist_pc  = _sample_paths(padchest_paths, args.n_hist)

    # ---- Load pixel arrays ---------------------------------------------------
    print(f"Loading {args.n_hist} images per dataset for statistics...")
    nih_pixels = np.concatenate([_load_gray(p).ravel() for p in hist_nih])
    pc_pixels  = np.concatenate([_load_gray(p).ravel() for p in hist_pc])

    # ---- Statistics ----------------------------------------------------------
    def stats(arr: np.ndarray, name: str) -> None:
        print(f"\n{name}:")
        print(f"  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"p5={np.percentile(arr,5):.3f}  p95={np.percentile(arr,95):.3f}")

    stats(nih_pixels, "NIH pixel stats")
    stats(pc_pixels,  "PadChest pixel stats")

    mean_shift = abs(nih_pixels.mean() - pc_pixels.mean())
    std_ratio  = nih_pixels.std() / (pc_pixels.std() + 1e-8)
    print(f"\nDomain gap indicators:")
    print(f"  Mean shift:  {mean_shift:.3f}  (0=identical, >0.1=notable)")
    print(f"  Std ratio:   {std_ratio:.3f}  (1.0=identical)")

    # ---- Plot ----------------------------------------------------------------
    has_ckpt = args.checkpoint is not None
    n_rows = 3 if has_ckpt else 2
    fig = plt.figure(figsize=(14, 5 * n_rows))

    # Row 1: sample images
    n = args.n_samples
    axes_nih = [fig.add_subplot(n_rows, n * 2, i + 1)          for i in range(n)]
    axes_pc  = [fig.add_subplot(n_rows, n * 2, n + i + 1)      for i in range(n)]

    for ax, path in zip(axes_nih, sample_nih):
        ax.imshow(_load_gray(path), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes_nih[n // 2].set_title("NIH", fontsize=12, fontweight="bold", pad=8)

    for ax, path in zip(axes_pc, sample_pc):
        ax.imshow(_load_gray(path), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    axes_pc[n // 2].set_title("PadChest", fontsize=12, fontweight="bold", pad=8)

    # Row 2: histograms
    ax_hist = fig.add_subplot(n_rows, 1, 2)
    ax_hist.hist(nih_pixels, bins=100, alpha=0.6, color="steelblue",
                 density=True, label=f"NIH  (μ={nih_pixels.mean():.3f}, σ={nih_pixels.std():.3f})")
    ax_hist.hist(pc_pixels,  bins=100, alpha=0.6, color="tomato",
                 density=True, label=f"PadChest (μ={pc_pixels.mean():.3f}, σ={pc_pixels.std():.3f})")
    ax_hist.set_xlabel("Pixel intensity (normalised to [0,1])")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Pixel intensity distribution — NIH vs PadChest")
    ax_hist.legend()

    # Row 3 (optional): feature UMAP
    if has_ckpt:
        try:
            from umap import UMAP
            from data.load_backbone import load_feature_extractor

            print(f"\nExtracting features from {args.checkpoint}...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            extractor = load_feature_extractor(ckpt, device)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            n_feat = min(300, len(nih_paths), len(padchest_paths))
            feat_nih_paths = _sample_paths(nih_paths,      n_feat, seed=args.seed + 1)
            feat_pc_paths  = _sample_paths(padchest_paths, n_feat, seed=args.seed + 2)

            def extract(paths):
                feats = []
                extractor.eval()
                with torch.no_grad():
                    for p in paths:
                        img = Image.open(p).convert("RGB")
                        t = transform(img).unsqueeze(0).to(device)
                        feats.append(extractor(t).cpu().numpy())
                return np.concatenate(feats, axis=0)

            f_nih = extract(feat_nih_paths)
            f_pc  = extract(feat_pc_paths)
            all_feats = np.concatenate([f_nih, f_pc], axis=0)
            labels = np.array([0] * n_feat + [1] * n_feat)

            print("Running UMAP...")
            emb = UMAP(n_components=2, random_state=42).fit_transform(all_feats)

            ax_umap = fig.add_subplot(n_rows, 1, 3)
            ax_umap.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                            c="steelblue", s=8, alpha=0.5, label="NIH")
            ax_umap.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                            c="tomato", s=8, alpha=0.5, label="PadChest")
            ax_umap.set_title("Feature space UMAP — are NIH and PadChest features mixed or separated?")
            ax_umap.legend()
            ax_umap.axis("off")

            # Overlap score: fraction of each point's k nearest neighbours from the other domain
            from sklearn.neighbors import NearestNeighbors
            k = 10
            nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
            _, idx = nn.kneighbors(emb)
            cross = (labels[idx[:, 1:]] != labels[:, None]).mean()
            print(f"  Feature overlap score: {cross:.3f}  "
                  f"(0=fully separated, 0.5=perfectly mixed, 1=all cross-domain neighbours)")

        except ImportError:
            print("umap-learn or sklearn not installed — skipping feature UMAP. "
                  "Install with: uv add umap-learn")

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
