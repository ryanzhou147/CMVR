"""SparK pretraining loop."""

import math
from pathlib import Path

import torch
import wandb
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from ssl_methods.spark.data import build_spark_dataloader
from ssl_methods.spark.model import SparK


def train_spark(config: dict) -> None:
    """Full SparK pretraining loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print("Building dataloader...")
    dataloader = build_spark_dataloader(config)
    print(f"Total batches per epoch: {len(dataloader)}")

    sp_cfg = config["spark"]
    model = SparK(
        img_size=config["data"]["image_size"],
        patch_size=sp_cfg["patch_size"],
        encoder_name=sp_cfg["encoder"],
        dec_dim=sp_cfg["dec_dim"],
        mask_ratio=sp_cfg["mask_ratio"],
        norm_pix_loss=sp_cfg.get("norm_pix_loss", True),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    enc_params = sum(p.numel() for p in [
        *model.stem.parameters(), *model.layer1.parameters(),
        *model.layer2.parameters(), *model.layer3.parameters(),
        *model.layer4.parameters(),
    ]) / 1e6
    print(f"Model parameters: {n_params:.1f}M  (encoder: {enc_params:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["training"]["weight_decay"],
    )

    warmup_epochs = config["training"]["warmup_epochs"]
    total_epochs = config["training"]["epochs"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    wandb_run_id = None

    resume_path = output_dir / "latest.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", start_epoch * len(dataloader))
        best_loss = checkpoint.get("best_loss", float("inf"))
        wandb_run_id = checkpoint.get("wandb_run_id")

    wandb.init(
        project="cmvr-ssl",
        name=config["training"]["run_name"],
        id=wandb_run_id,
        resume="allow",
        config=config,
    )

    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc="SparK Pretraining", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0

        batch_pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{total_epochs}", leave=False)
        for imgs in batch_pbar:
            imgs = imgs.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                loss, _, _ = model(imgs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if grad_clip := config["training"].get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            wandb.log({"loss/step": loss.item()}, step=global_step)
            global_step += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        lr = optimizer.param_groups[0]["lr"]

        epoch_pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
        wandb.log({"loss/epoch": avg_loss, "lr": lr}, step=global_step)

        checkpoint_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_loss": min(best_loss, avg_loss),
            "wandb_run_id": wandb.run.id,
            "config": config,
        }
        torch.save(checkpoint_data, output_dir / "latest.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint_data, output_dir / "best.pt")

        if (epoch + 1) % config["training"]["checkpoint_every"] == 0:
            torch.save(checkpoint_data, output_dir / f"epoch_{epoch+1}.pt")

    wandb.finish()
    print(f"Training complete. Checkpoints saved to {output_dir}")
