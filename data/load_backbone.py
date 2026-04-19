"""Shared backbone loader for SSL checkpoints.

Supports MoCo, DINO, BarlowTwins, and SparK. Auto-detects the method from the
checkpoint config dict. Both frozen feature extractors (for probing) and
unfrozen backbones (for gradient fine-tuning) are provided.
"""

import torch
import torch.nn as nn


class _GetFeaturesWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_features(x)


class _SparKBackbone(nn.Module):
    # SparK encoder stages aren't direct children of the model, so re-register
    # them here so unfreeze_for_finetuning can target .layer1/.layer2/etc.
    def __init__(self, model: nn.Module):
        super().__init__()
        self.stem    = model.stem
        self.layer1  = model.layer1
        self.layer2  = model.layer2
        self.layer3  = model.layer3
        self.layer4  = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return self.avgpool(h).flatten(1)


def _build_ssl_model(ckpt: dict, random_init: bool) -> tuple[nn.Module, str]:
    """Instantiate and optionally load the SSL model from a checkpoint.

    Returns (model, method_key) where method_key is one of:
    'moco', 'dino', 'barlow', 'spark'.
    """
    config = ckpt["config"]

    if "moco" in config:
        from ssl_methods.moco.model import MoCo
        cfg = config["moco"]
        model = MoCo(
            encoder_name=cfg["encoder"],
            dim=cfg["dim"],
            K=cfg["queue_size"],
            m=cfg["momentum"],
            T=cfg["temperature"],
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        return model, "moco"

    if "dino" in config:
        from ssl_methods.dino.model import DINO
        cfg = config["dino"]
        model = DINO(encoder_name=cfg["encoder"], out_dim=cfg["out_dim"])
        if not random_init:
            model.load_state_dict(ckpt["model"])
        return model, "dino"

    if "barlow" in config:
        from ssl_methods.barlow.model import BarlowTwins
        cfg = config["barlow"]
        model = BarlowTwins(
            encoder_name=cfg["encoder"],
            proj_dim=cfg["proj_dim"],
            proj_hidden_dim=cfg["proj_hidden_dim"],
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        return model, "barlow"

    if "spark" in config:
        from ssl_methods.spark.model import SparK
        cfg = config["spark"]
        model = SparK(
            img_size=config["data"]["image_size"],
            patch_size=cfg["patch_size"],
            encoder_name=cfg["encoder"],
            dec_dim=cfg["dec_dim"],
            mask_ratio=cfg["mask_ratio"],
            norm_pix_loss=cfg.get("norm_pix_loss", True),
        )
        if not random_init:
            model.load_state_dict(ckpt["model"])
        return model, "spark"

    raise ValueError(
        f"Unrecognised SSL method. Config top-level keys: {list(config.keys())}"
    )


def _freeze_all(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def load_feature_extractor(
    ckpt: dict,
    device: torch.device,
    random_init: bool = False,
) -> nn.Module:
    """Frozen feature extractor from a checkpoint. forward(x) -> (B, D)."""
    model, method = _build_ssl_model(ckpt, random_init)

    if method == "moco":
        model.encoder_q.fc = nn.Identity()
        model.encoder_k = None
        extractor: nn.Module = model.encoder_q
    elif method == "dino":
        extractor = model.student[0]
    elif method == "barlow":
        extractor = _GetFeaturesWrapper(model)
    else:  # spark
        extractor = _GetFeaturesWrapper(model)

    extractor = extractor.to(device)
    _freeze_all(extractor)
    return extractor


def load_raw_backbone(
    ckpt: dict,
    device: torch.device,
    random_init: bool = False,
) -> tuple[nn.Module, str]:
    """Backbone with named ResNet stages for gradient-based fine-tuning.

    All parameters start frozen. Call unfreeze_for_finetuning afterwards.
    Returns (backbone, method) where method is one of: moco, dino, barlow, spark.
    """
    model, method = _build_ssl_model(ckpt, random_init)

    if method == "moco":
        backbone: nn.Module = model.encoder_q
        backbone.fc = nn.Identity()
    elif method == "dino":
        backbone = model.student[0]
    elif method == "barlow":
        backbone = model.backbone
    else:  # spark
        backbone = _SparKBackbone(model)

    backbone = backbone.to(device)
    _freeze_all(backbone)
    return backbone, method


def load_imagenet_feature_extractor(device: torch.device) -> nn.Module:
    """Frozen ResNet50 pretrained on ImageNet1K. forward(x) -> (B, 2048)."""
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.to(device)
    _freeze_all(model)
    return model


def load_imagenet_raw_backbone(device: torch.device) -> tuple[nn.Module, str]:
    """ResNet50 pretrained on ImageNet1K with named stages for fine-tuning.

    All parameters start frozen. Call unfreeze_for_finetuning afterwards.
    """
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.to(device)
    _freeze_all(model)
    return model, "imagenet"


def unfreeze_for_finetuning(backbone: nn.Module, method: str) -> list[str]:
    """Selectively unfreeze ResNet stages for gradient fine-tuning.

    Contrastive and supervised methods unfreeze from layer2 up.
    SparK (restorative) keeps layer2 frozen: the U-Net decoder already
    adapted those features during pretraining.
    """
    if method in ("moco", "barlow", "dino", "imagenet"):
        unfreeze = {"layer2", "layer3", "layer4"}
    else:
        unfreeze = {"layer3", "layer4"}

    for name, child in backbone.named_children():
        for p in child.parameters():
            p.requires_grad_(name in unfreeze)

    # BatchNorm stats were computed over 112k images; don't corrupt them.
    for m in backbone.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                p.requires_grad_(False)

    return sorted(unfreeze)


def method_name(ckpt: dict) -> str:
    """Return the SSL method name from a checkpoint dict."""
    config = ckpt["config"]
    for key in ("moco", "dino", "barlow", "spark"):
        if key in config:
            return key
    return "unknown"
