"""
Clean-room reimplementation of SegFormer3D (Perera, Navard & Yilmaz, CVPR-W 2024,
arXiv:2404.10156), a hierarchical 3D transformer for medical image segmentation.

Reference points taken from the paper text:
  - Overlapping patch embedding (3D conv, stride < kernel) at each of 4 hierarchical
    stages -> preserves local continuity of voxels, avoids the "torn" edges of
    non-overlapping ViT patchify.
  - Positional-encoding-free design: local continuity from the overlap conv + the
    3x3x3 depthwise conv inside Mix-FFN substitutes for explicit position embeddings,
    which is what lets the model handle train/test resolution mismatch gracefully
    (a common issue in medical imaging where volumes vary in size).
  - Efficient self-attention via spatial ("sequence") reduction: keys/values are
    downsampled by a strided conv before the attention product, cutting the O(N^2)
    cost of full 3D volumetric self-attention to something tractable at 4.5M params.
  - All-MLP decoder: no transformer / no complex upsampling path in the decoder,
    just per-stage linear projections to a common channel dim, upsample to a common
    resolution, concat, and a final MLP -> per-voxel class logits.

This module exposes both the assembled model (`SegFormer3D`) and the individual
building blocks (`OverlapPatchEmbed3D`, `EfficientSelfAttention3D`, `MixFFN3D`,
`TransformerBlock3D`, `AllMLPDecoder3D`) so that Phase 2 can hook the boundary
refinement branch onto intermediate decoder features without re-deriving the whole
forward pass.

NOT dropped in verbatim from the paper -- this is written from the architectural
description, so validate against the official repo (github.com/OSUPCVLab/SegFormer3D)
before trusting absolute numbers; use this for the boundary-branch surgery in Phase 2.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Overlapping patch embedding
# --------------------------------------------------------------------------- #
class OverlapPatchEmbed3D(nn.Module):
    """3D conv patch embedding with overlap (stride < kernel_size).

    Downsamples spatial dims by `stride` while keeping receptive-field overlap
    between adjacent patches, which is what preserves local voxel continuity
    relative to a non-overlapping ViT-style patchify.
    """

    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int = 7, stride: int = 4):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x: (B, C, D, H, W)
        x = self.proj(x)
        d, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, (d, h, w)


# --------------------------------------------------------------------------- #
# Efficient self-attention with spatial reduction
# --------------------------------------------------------------------------- #
class EfficientSelfAttention3D(nn.Module):
    """Multi-head self-attention with a spatial-reduction (SR) step on K/V.

    Full 3D self-attention over a volumetric token sequence is O(N^2) where
    N = D*H*W tokens -- prohibitive at higher resolutions. Following the
    SegFormer(2D)/PVT lineage that SegFormer3D extends into 3D, we reduce the
    K/V sequence length by `sr_ratio` via a strided conv before computing
    attention, which keeps memory/compute tractable at the early, high-resolution
    stages while leaving Q at full resolution (so output granularity is preserved).
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        # x: (B, N, C), shape = (D, H, W) with N == D*H*W
        b, n, c = x.shape
        d, h, w = shape

        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, d, h, w)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv_input = x_
        else:
            kv_input = x

        kv = self.kv(kv_input).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# --------------------------------------------------------------------------- #
# Mix-FFN (position-free FFN with a depthwise conv for local structure)
# --------------------------------------------------------------------------- #
class MixFFN3D(nn.Module):
    """MLP with an interleaved 3x3x3 depthwise conv.

    The depthwise conv provides implicit positional information via local
    neighborhood mixing, which is part of why the encoder needs no explicit
    positional embedding (and therefore degrades gracefully under train/test
    resolution mismatch, per the paper's stated motivation).
    """

    def __init__(self, dim: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        b, n, c = x.shape
        d, h, w = shape
        x = self.fc1(x)
        hidden_dim = x.shape[-1]
        x = x.transpose(1, 2).reshape(b, hidden_dim, d, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention3D(dim, num_heads, sr_ratio, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN3D(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), shape))
        x = x + self.drop_path(self.ffn(self.norm2(x), shape))
        return x


class DropPath(nn.Module):
    """Stochastic depth, per-sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# --------------------------------------------------------------------------- #
# Hierarchical encoder (4 stages)
# --------------------------------------------------------------------------- #
class SegFormer3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        depths: Sequence[int] = (2, 2, 2, 2),
        patch_kernel: Sequence[int] = (7, 3, 3, 3),
        patch_stride: Sequence[int] = (4, 2, 2, 2),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.num_stages = len(embed_dims)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_ch = in_channels
        cursor = 0
        for i in range(self.num_stages):
            self.patch_embeds.append(
                OverlapPatchEmbed3D(in_ch, embed_dims[i], patch_kernel[i], patch_stride[i])
            )
            stage_blocks = nn.ModuleList(
                [
                    TransformerBlock3D(
                        embed_dims[i], num_heads[i], sr_ratios[i], mlp_ratio,
                        drop_rate, attn_drop_rate, dpr[cursor + j],
                    )
                    for j in range(depths[i])
                ]
            )
            self.blocks.append(stage_blocks)
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            in_ch = embed_dims[i]
            cursor += depths[i]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, C, D, H, W). Returns list of 4 feature maps (B, C_i, D_i, H_i, W_i)
        outs = []
        for i in range(self.num_stages):
            x, shape = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, shape)
            x = self.norms[i](x)
            b, n, c = x.shape
            d, h, w = shape
            x = x.transpose(1, 2).reshape(b, c, d, h, w)
            outs.append(x)
        return outs


# --------------------------------------------------------------------------- #
# All-MLP decoder
# --------------------------------------------------------------------------- #
class AllMLPDecoder3D(nn.Module):
    """Aggregates multi-stage encoder features with simple linear projections
    + upsampling + concat + fused MLP -- deliberately avoiding a heavy decoder,
    per the paper's design goal of keeping the whole model lightweight.
    """

    def __init__(self, embed_dims: Sequence[int], decoder_dim: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.linears = nn.ModuleList(
            [nn.Conv3d(dim, decoder_dim, kernel_size=1) for dim in embed_dims]
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_dim * len(embed_dims), decoder_dim, kernel_size=1),
            nn.BatchNorm3d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout3d(dropout)
        self.classifier = nn.Conv3d(decoder_dim, num_classes, kernel_size=1)
        self.decoder_dim = decoder_dim

    def forward(self, feats: List[torch.Tensor], out_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        # feats: list of (B, C_i, D_i, H_i, W_i), highest-res feature (feats[0]) sets fusion resolution
        target_shape = feats[0].shape[2:]
        projected = []
        for feat, lin in zip(feats, self.linears):
            p = lin(feat)
            if p.shape[2:] != target_shape:
                p = F.interpolate(p, size=target_shape, mode="trilinear", align_corners=False)
            projected.append(p)
        fused = torch.cat(projected, dim=1)
        fused = self.fuse(fused)
        decoder_feat = self.dropout(fused)
        logits = self.classifier(decoder_feat)
        logits = F.interpolate(logits, size=out_shape, mode="trilinear", align_corners=False)
        # also return the pre-classifier fused feature map (upsampled) for the
        # boundary refinement branch to consume in Phase 2
        fused_up = F.interpolate(fused, size=out_shape, mode="trilinear", align_corners=False)
        return logits, fused_up


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SegFormer3D(nn.Module):
    """Assembled SegFormer3D: hierarchical encoder + all-MLP decoder.

    Default config (~4-5M params) mirrors the paper's stated lightweight target.
    Set `in_channels=4` for BraTS's 4 MRI modalities (T1, T1ce, T2, FLAIR) and
    `num_classes=3` for the standard BraTS region formulation (WT, TC, ET) — or
    `num_classes=4` if you prefer predicting the raw label set (NCR/NET, ED, ET,
    background) and deriving WT/TC/ET regions at eval time; pick one and be
    consistent between training targets and evaluation metric code.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        embed_dims: Sequence[int] = (32, 64, 160, 256),
        num_heads: Sequence[int] = (1, 2, 5, 8),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        depths: Sequence[int] = (2, 2, 2, 2),
        decoder_dim: int = 128,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = SegFormer3DEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            sr_ratios=sr_ratios,
            depths=depths,
            drop_path_rate=drop_path_rate,
        )
        self.decoder = AllMLPDecoder3D(embed_dims, decoder_dim, num_classes)

    def forward(self, x: torch.Tensor, return_decoder_features: bool = False):
        in_shape = x.shape[2:]
        feats = self.encoder(x)
        logits, decoder_feat = self.decoder(feats, out_shape=in_shape)
        if return_decoder_features:
            return logits, decoder_feat
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # smoke test shape check (run this on a machine with torch installed)
    model = SegFormer3D(in_channels=4, num_classes=3)
    print(f"SegFormer3D params: {model.num_parameters() / 1e6:.2f}M")
    dummy = torch.randn(1, 4, 128, 128, 128)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
