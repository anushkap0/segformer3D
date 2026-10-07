"""Loss functions for region segmentation and boundary-aware training.

BraTS is conventionally trained with soft Dice + BCE on the three overlapping regions
(WT, TC, ET) using sigmoid outputs (each region predicted independently, not softmax
mutually-exclusive classes) -- see the original BraTS nnU-Net / SegResNet formulations.
This file defaults to that sigmoid multi-label formulation; set `mutually_exclusive=True`
if you instead train on the 4-way raw label set with softmax (background, NCR/NET, ED, ET).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, mutually_exclusive: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.mutually_exclusive = mutually_exclusive
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits, target: (B, C, D, H, W). target is one-hot / multi-label float {0,1}.
        if self.mutually_exclusive:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target, dim=dims)
        cardinality = torch.sum(probs + target, dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, mutually_exclusive: bool = False, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(mutually_exclusive)
        self.mutually_exclusive = mutually_exclusive
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, target)
        if self.mutually_exclusive:
            ce_loss = F.cross_entropy(logits, target.argmax(dim=1))
        else:
            ce_loss = F.binary_cross_entropy_with_logits(logits, target)
        return self.dice_weight * dice_loss + self.bce_weight * ce_loss


class BoundaryLoss(nn.Module):
    """BCE + Dice on the predicted boundary map against a boundary target derived from
    the ground-truth mask (e.g. a morphological-gradient edge map computed once during
    data prep -- see `src/data/brats_dataset.py::compute_boundary_target`).
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice = SoftDiceLoss(mutually_exclusive=False)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, boundary_logits: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(boundary_logits, boundary_target)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class DualSupervisionCompositeLoss(nn.Module):
    """Composite loss for the Phase 2 boundary-refinement model. Mirrors PFF-Net's
    dual-supervision ablation (their variant C, applying supervision to both the
    region and boundary decoders, was their best-performing configuration).

    total = region_weight * DiceBCE(region_logits, target)
          + refined_weight * DiceBCE(refined_logits, target)
          + boundary_weight * BoundaryLoss(boundary_logits, boundary_target)
    """

    def __init__(
        self,
        region_weight: float = 1.0,
        refined_weight: float = 1.0,
        boundary_weight: float = 0.5,
        mutually_exclusive: bool = False,
    ):
        super().__init__()
        self.region_loss = DiceBCELoss(mutually_exclusive)
        self.refined_loss = DiceBCELoss(mutually_exclusive)
        self.boundary_loss = BoundaryLoss()
        self.region_weight = region_weight
        self.refined_weight = refined_weight
        self.boundary_weight = boundary_weight

    def forward(self, outputs: dict, target: torch.Tensor, boundary_target: torch.Tensor) -> dict:
        l_region = self.region_loss(outputs["region_logits"], target)
        l_refined = self.refined_loss(outputs["refined_logits"], target)
        l_boundary = self.boundary_loss(outputs["boundary_logits"], boundary_target)
        total = (
            self.region_weight * l_region
            + self.refined_weight * l_refined
            + self.boundary_weight * l_boundary
        )
        return {
            "total": total,
            "region_loss": l_region.detach(),
            "refined_loss": l_refined.detach(),
            "boundary_loss": l_boundary.detach(),
        }
