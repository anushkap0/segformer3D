"""3D boundary-aware refinement (v2).

Changes vs v1: (1) the edge prior uses sigmoid, not softmax, because WT/TC/ET are
overlapping multi-label regions; (2) the branch runs at the decoder's native
resolution (1/4 of the input) and its outputs are upsampled to full resolution,
so the boundary loss is still computed at full resolution.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel3D(nn.Module):
    """Fixed 3D Sobel gradient magnitude, per channel, with per-axis voxel spacing."""

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        smooth = torch.tensor([1.0, 2.0, 1.0])
        deriv = torch.tensor([-1.0, 0.0, 1.0])
        kd = torch.einsum("i,j,k->ijk", deriv, smooth, smooth)
        kh = torch.einsum("i,j,k->ijk", smooth, deriv, smooth)
        kw = torch.einsum("i,j,k->ijk", smooth, smooth, deriv)
        sd, sh, sw = voxel_spacing
        kernels = torch.stack([kd / max(sd, 1e-6), kh / max(sh, 1e-6), kw / max(sw, 1e-6)], dim=0)
        self.register_buffer("kernels", kernels.unsqueeze(1))  # (3, 1, 3, 3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float().reshape(b * c, 1, d, h, w)
            g = F.conv3d(x32, self.kernels, padding=1)
            mag = torch.sqrt(torch.clamp((g ** 2).sum(dim=1, keepdim=True), min=1e-12))
        return mag.reshape(b, c, d, h, w)


class BoundaryBranch3D(nn.Module):
    def __init__(self, decoder_dim: int, num_classes: int,
                 voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 feature_stride: int = 4):
        super().__init__()
        # one low-res voxel spans `feature_stride` input voxels
        eff_spacing = tuple(s * feature_stride for s in voxel_spacing)
        self.sobel = Sobel3D(eff_spacing)
        self.conv = nn.Sequential(
            nn.Conv3d(decoder_dim + num_classes, decoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_dim // 2, decoder_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(decoder_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv3d(decoder_dim // 2, 1, kernel_size=1)

    def forward(self, decoder_feat: torch.Tensor, region_logits_lr: torch.Tensor):
        region_prob = torch.sigmoid(region_logits_lr.float())  # multi-label, not softmax
        edge_prior = self.sobel(region_prob).to(decoder_feat.dtype)
        feat = self.conv(torch.cat([decoder_feat, edge_prior], dim=1))
        return self.boundary_head(feat), feat


class BoundaryGatedFusion3D(nn.Module):
    def __init__(self, decoder_dim: int, num_classes: int):
        super().__init__()
        bdim = decoder_dim // 2
        self.gate = nn.Sequential(nn.Conv3d(decoder_dim + bdim, decoder_dim, kernel_size=1), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv3d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv3d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, decoder_feat: torch.Tensor, boundary_feat: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([decoder_feat, boundary_feat], dim=1))
        return self.classifier(self.refine(decoder_feat * gate))


class SegFormer3DWithBoundary(nn.Module):
    def __init__(self, base_model: nn.Module, decoder_dim: int, num_classes: int,
                 voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 feature_stride: int = 4):
        super().__init__()
        self.base_model = base_model
        self.boundary_branch = BoundaryBranch3D(decoder_dim, num_classes, voxel_spacing, feature_stride)
        self.fusion = BoundaryGatedFusion3D(decoder_dim, num_classes)

    def forward(self, x: torch.Tensor):
        out_shape = x.shape[2:]
        region_logits, feat_lr, region_logits_lr = self.base_model(x, return_decoder_features=True)
        boundary_logits_lr, boundary_feat = self.boundary_branch(feat_lr, region_logits_lr)
        refined_logits_lr = self.fusion(feat_lr, boundary_feat)

        def up(t):
            return F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)

        return {
            "region_logits": region_logits,
            "boundary_logits": up(boundary_logits_lr),
            "refined_logits": up(refined_logits_lr),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from segformer3d import SegFormer3D

    base = SegFormer3D(in_channels=4, num_classes=3, decoder_dim=128)
    model = SegFormer3DWithBoundary(base, decoder_dim=128, num_classes=3)
    print(f"params: {model.num_parameters() / 1e6:.2f}M (base {base.num_parameters() / 1e6:.2f}M)")
    out = model(torch.randn(1, 4, 96, 96, 96))
    for k, v in out.items():
        print(k, tuple(v.shape))
