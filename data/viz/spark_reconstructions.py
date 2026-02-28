"""Visualise SparK masked image reconstructions.

Shows a grid of: Original | Masked (60%) | Reconstructed | Composite
for N random NIH images. Use this to sanity-check that the model has
learned meaningful anatomy rather than trivial local texture copying.

Usage:
    uv run python -m data.viz.spark_reconstructions --checkpoint outputs/spark_cloud/latest.pt
    uv run python -m data.viz.spark_reconstructions --checkpoint outputs/spark_cloud/latest.pt --n-images 8 --output recon.png
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from data.dataset import collect_image_paths
from ssl_methods.spark.model import SparK

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm(t: torch.Tensor) -> np.ndarray:
    """ImageNet-denormalise a (3,H,W) tensor → (H,W) float32 in [0,1]."""
    img = (t * _STD + _MEAN).clamp(0, 1)
    return img.mean(0).numpy()


def _norm_display(t: torch.Tensor) -> np.ndarray:
    """Min-max normalise a (3,H,W) prediction tensor for display.

    Used for norm_pix_loss=True models where pred is in patch-normalised
    space (not ImageNet space) — applying ImageNet denorm gives gray blobs.
    """
    img = t.mean(0).numpy()
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise SparK reconstructions")
    parser.add_argument("--checkpoint", required=True, help="Path to SparK .pt checkpoint")
    parser.add_argument("--data-dir", default="datasets/nih-chest-xrays")
    parser.add_argument("--n-images", type=int, default=6)
    parser.add_argument("--output", default="spark_reconstructions.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----------------------------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    sp_cfg = config["spark"]

    model = SparK(
        img_size=config["data"]["image_size"],
        patch_size=sp_cfg["patch_size"],
        encoder_name=sp_cfg["encoder"],
        dec_dim=sp_cfg["dec_dim"],
        mask_ratio=sp_cfg["mask_ratio"],
        norm_pix_loss=False,   # raw pixel space for display — we only care about the visual output
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # ---- Load images ---------------------------------------------------------
    paths = collect_image_paths("nih", args.data_dir)
    paths = random.sample(paths, min(args.n_images, len(paths)))

    size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imgs = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ]).to(device)

    # ---- Forward pass --------------------------------------------------------
    with torch.no_grad():
        _, pred, mask_px = model(imgs)

    imgs   = imgs.cpu()
    pred   = pred.cpu()
    mask_px = mask_px.cpu()

    # ---- Plot ----------------------------------------------------------------
    n = len(paths)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None]   # ensure 2-D indexing

    col_titles = ["Original", f"Masked ({int(sp_cfg['mask_ratio']*100)}%)", "Reconstruction", "Composite"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for i in range(n):
        orig_full = _denorm(imgs[i])
        mask      = mask_px[i, 0].numpy()   # (H,W) binary, 1=masked

        # pred is in patch-normalised space — min-max to [0,1] first
        recon_raw = _norm_display(pred[i])

        # Re-scale predicted patches to match the brightness/contrast of the
        # visible patches in this image.  This removes the per-patch colour
        # discontinuity at boundaries — both visible and predicted regions are
        # now in the same absolute intensity space.
        vis_pixels = orig_full[mask == 0]
        if vis_pixels.size > 0:
            vis_lo, vis_hi = vis_pixels.min(), vis_pixels.max()
            recon_display = recon_raw * (vis_hi - vis_lo) + vis_lo
        else:
            recon_display = recon_raw

        recon_display = recon_display.clip(0, 1)

        # Smooth the reconstruction to reduce per-patch brightness jumps caused
        # by norm_pix_loss independent normalisation. sigma=2 blurs the ~1px
        # boundary region without destroying anatomical structure inside patches.
        recon_smooth = gaussian_filter(recon_display, sigma=2.0)

        # Soft mask: feather the patch edges so the composite blends gradually
        # instead of cutting hard at each 32px boundary.
        soft_mask = gaussian_filter(mask.astype(np.float32), sigma=3.0)

        # Composite: real visible patches + smoothed predictions for masked patches
        composite = orig_full * (1 - soft_mask) + recon_smooth * soft_mask

        for col, display_img in enumerate([
            orig_full,
            orig_full * (1 - mask),   # black where masked
            recon_smooth,
            composite,
        ]):
            axes[i, col].imshow(display_img, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[i, col].axis("off")

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
