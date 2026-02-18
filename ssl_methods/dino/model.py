"""DINO model components: backbone builder, DINOHead, and DINO student-teacher."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def build_backbone(encoder_name: str) -> tuple[nn.Module, int]:
    """Instantiate a backbone and return ``(backbone, feature_dim)``.

    Supports torchvision ResNets (``"resnet18"``, ``"resnet50"``) and
    Vision Transformers (``"vit_b_16"``, ``"vit_b_32"``, ``"vit_l_16"``).
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


class DINOHead(nn.Module):
    """DINO projection head.

    Architecture::

        Linear(in_dim → hidden_dim)
        [GELU → Linear(hidden_dim → hidden_dim)] × (n_layers - 2)
        GELU → Linear(hidden_dim → bottleneck_dim)
        L2-normalise
        weight-norm Linear(bottleneck_dim → out_dim, no bias)

    The weight-norm layer has its magnitude component frozen at 1 so only
    the direction of each prototype vector is learned.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            layers += [nn.GELU(), nn.Linear(hidden_dim, hidden_dim)]
        layers += [nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim)]
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINO(nn.Module):
    """Student-teacher DINO model.

    The teacher is an EMA of the student and never receives gradients.
    Both share the same backbone + DINOHead architecture.

    Args:
        encoder_name: Backbone name passed to :func:`build_backbone`.
        out_dim:      Prototype dimension for both student and teacher heads.
    """

    def __init__(self, encoder_name: str = "vit_b_16", out_dim: int = 65536):
        super().__init__()
        backbone_s, feature_dim = build_backbone(encoder_name)
        backbone_t, _ = build_backbone(encoder_name)

        # ViT requires a fixed input size; store it so forward() can resize local crops
        self._fixed_size: int | None = (
            getattr(backbone_s, "image_size", None) if encoder_name.startswith("vit") else None
        )

        self.student = nn.Sequential(backbone_s, DINOHead(feature_dim, out_dim))
        self.teacher = nn.Sequential(backbone_t, DINOHead(feature_dim, out_dim))

        for p in self.teacher.parameters():
            p.requires_grad = False
        self._copy_student_to_teacher()

    def _copy_student_to_teacher(self) -> None:
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.copy_(p_s.data)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA update of the teacher network."""
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data = p_t.data * momentum + p_s.data * (1.0 - momentum)

    def forward(
        self, views: list[torch.Tensor], n_global: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass over all views.

        The student sees every view; the teacher sees only the first
        ``n_global`` (large-crop) views.

        Returns:
            student_output: ``list`` of ``(B, out_dim)`` tensors — one per view.
            teacher_output: ``list`` of ``(B, out_dim)`` tensors — global views only.
        """
        if self._fixed_size is not None:
            s = self._fixed_size
            views = [
                F.interpolate(v, size=(s, s), mode="bilinear", align_corners=False)
                if v.shape[-1] != s else v
                for v in views
            ]
        student_out = [self.student(v) for v in views]
        with torch.no_grad():
            teacher_out = [self.teacher(views[i]) for i in range(n_global)]
        return student_out, teacher_out
