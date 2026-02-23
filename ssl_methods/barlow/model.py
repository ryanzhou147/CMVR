"""BarlowTwins model: shared backbone + 3-layer BatchNorm projector.

Architecture follows Zbontar et al. 2021.  The projector uses BatchNorm
(not LayerNorm) because the cross-correlation loss is computed over the
batch dimension — BatchNorm ensures the projections are roughly zero-mean
and unit-variance, which keeps the correlation matrix well-conditioned.

At fine-tuning time the projector is discarded; only the backbone is kept.
"""

import torch
import torch.nn as nn
from torchvision import models


def _build_backbone(encoder_name: str) -> tuple[nn.Module, int]:
    """Instantiate a backbone and return ``(backbone, feature_dim)``.

    The classification head is replaced with ``nn.Identity`` so the backbone
    outputs raw feature vectors.
    """
    if encoder_name.startswith("vit"):
        backbone = getattr(models, encoder_name)(weights=None)
        feature_dim: int = backbone.hidden_dim
        backbone.heads = nn.Identity()
    else:
        backbone = getattr(models, encoder_name)(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    return backbone, feature_dim


class BarlowTwins(nn.Module):
    """BarlowTwins student model.

    Both views pass through the *same* backbone and projector — there is no
    teacher, no queue, and no momentum EMA.  The loss alone prevents collapse.

    Args:
        encoder_name:     torchvision backbone (e.g. ``"resnet50"``).
        proj_dim:         Output dimension of the projector (prototype space).
        proj_hidden_dim:  Hidden width of the projector MLP.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        proj_dim: int = 2048,
        proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.backbone, feature_dim = _build_backbone(encoder_name)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_dim, bias=False),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both views.

        Returns:
            z1, z2: ``(N, proj_dim)`` projection tensors for view 1 and view 2.
        """
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        return z1, z2

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features for downstream evaluation.

        Returns ``(N, feature_dim)`` — projector is discarded.
        """
        return self.backbone(x)
