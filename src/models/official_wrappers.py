"""Uses the OFFICIAL SegFormer3D class unchanged; the boundary branch attaches via forward hooks."""
from __future__ import annotations
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.boundary_refinement import BoundaryBranch3D, BoundaryGatedFusion3D


def _official_class(repo_dir):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from architectures.segformer3d import SegFormer3D
    return SegFormer3D


class OfficialSegFormer3DWithBoundary(nn.Module):
    def __init__(self, base, fuse_dim, num_classes, boundary_dim=128,
                 voxel_spacing=(1.0, 1.0, 1.0), feature_stride=4):
        super().__init__()
        self.base_model = base
        self._feat = None
        self._logits_lr = None
        dec = base.segformer_decoder
        dec.linear_fuse.register_forward_hook(self._save_feat)
        dec.linear_pred.register_forward_hook(self._save_logits)
        # own width: project decoder features down first, so the add-on stays small
        self.proj = nn.Sequential(nn.Conv3d(fuse_dim, boundary_dim, 1),
                                  nn.BatchNorm3d(boundary_dim), nn.ReLU(inplace=True))
        self.boundary_branch = BoundaryBranch3D(boundary_dim, num_classes, voxel_spacing, feature_stride)
        self.fusion = BoundaryGatedFusion3D(boundary_dim, num_classes)

    def _save_feat(self, module, inputs, output):
        self._feat = output

    def _save_logits(self, module, inputs, output):
        self._logits_lr = output

    def forward(self, x):
        out_shape = x.shape[2:]
        region_logits = self.base_model(x)
        feat = self.proj(self._feat)
        boundary_lr, boundary_feat = self.boundary_branch(feat, self._logits_lr)
        refined_lr = self.fusion(feat, boundary_feat)

        def up(t):
            return F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)

        return {"region_logits": region_logits,
                "boundary_logits": up(boundary_lr),
                "refined_logits": up(refined_lr)}


def build_official_model(m, init_ckpt=None):
    Off = _official_class(m.get("official_repo_dir", "/workspace/SegFormer3D-official"))
    base = Off(in_channels=m["in_channels"], sr_ratios=m["sr_ratios"], embed_dims=m["embed_dims"],
               patch_kernel_size=m["patch_kernel_size"], patch_stride=m["patch_stride"],
               patch_padding=m["patch_padding"], mlp_ratios=m["mlp_ratios"], num_heads=m["num_heads"],
               depths=m["depths"], decoder_head_embedding_dim=m["decoder_head_embedding_dim"],
               num_classes=m["num_classes"], decoder_dropout=m["decoder_dropout"])
    if init_ckpt and os.path.exists(init_ckpt):
        base.load_state_dict(torch.load(init_ckpt, map_location="cpu")["model_state_dict"], strict=True)
        print(f"Warm-started from {init_ckpt}")
    if m["type"] == "segformer3d_official_boundary":
        return OfficialSegFormer3DWithBoundary(
            base, fuse_dim=m["decoder_head_embedding_dim"], num_classes=m["num_classes"],
            boundary_dim=m.get("boundary_dim", 128),
            voxel_spacing=tuple(m.get("voxel_spacing", (1.0, 1.0, 1.0))))
    return base
