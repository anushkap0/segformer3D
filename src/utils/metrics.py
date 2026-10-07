"""Evaluation metrics for 3D segmentation: Dice, HD95, sensitivity, precision.

HD95 uses `medpy.metric.binary.hd95` since it's a well-tested, widely-cited
implementation (used by nnU-Net and most BraTS leaderboard submissions) rather than
reimplementing distance-transform-based Hausdorff distance from scratch.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:
    from medpy.metric.binary import hd95 as _medpy_hd95
except ImportError:  # pragma: no cover
    _medpy_hd95 = None


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * intersection + smooth) / (denom + smooth))


def hd95_score(pred: np.ndarray, target: np.ndarray, voxel_spacing=(1.0, 1.0, 1.0)) -> float:
    """Returns NaN if either mask is empty (HD95 undefined) -- callers should
    exclude NaNs from the mean rather than treating them as 0, since a 0 would
    falsely reward the degenerate empty-prediction case on an empty ground truth,
    and a naive large-penalty substitute biases the aggregate metric arbitrarily.
    """
    if _medpy_hd95 is None:
        raise ImportError("medpy is required for HD95 -- pip install medpy")
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    return float(_medpy_hd95(pred, target, voxelspacing=voxel_spacing))


def sensitivity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def precision_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def compute_region_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    region_names=("WT", "TC", "ET"),
    threshold: float = 0.5,
    voxel_spacing=(1.0, 1.0, 1.0),
    mutually_exclusive: bool = False,
) -> Dict[str, Dict[str, float]]:
    """pred_logits, target: (C, D, H, W) single-sample tensors (no batch dim).
    Returns {region_name: {"dice": ..., "hd95": ..., "sensitivity": ..., "precision": ...}}
    """
    if mutually_exclusive:
        probs = torch.softmax(pred_logits, dim=0)
        pred_bin = torch.argmax(probs, dim=0)
        pred_np = pred_bin.cpu().numpy()
        target_np = target.argmax(dim=0).cpu().numpy() if target.ndim == 4 else target.cpu().numpy()
    else:
        probs = torch.sigmoid(pred_logits)
        pred_np = (probs > threshold).cpu().numpy()
        target_np = target.cpu().numpy().astype(bool)

    results = {}
    for i, name in enumerate(region_names):
        if mutually_exclusive:
            p = (pred_np == i + 1)  # assumes class 0 = background
            t = (target_np == i + 1)
        else:
            p = pred_np[i]
            t = target_np[i]
        results[name] = {
            "dice": dice_score(p, t),
            "hd95": hd95_score(p, t, voxel_spacing),
            "sensitivity": sensitivity_score(p, t),
            "precision": precision_score(p, t),
        }
    return results


def aggregate_metrics(per_case_results: list) -> Dict[str, Dict[str, float]]:
    """per_case_results: list of dicts as returned by compute_region_metrics.
    Averages each metric per region, using nanmean so undefined HD95 cases
    (empty masks) don't corrupt the aggregate.
    """
    if not per_case_results:
        return {}
    region_names = list(per_case_results[0].keys())
    metric_names = list(per_case_results[0][region_names[0]].keys())
    out = {r: {} for r in region_names}
    for r in region_names:
        for m in metric_names:
            vals = np.array([case[r][m] for case in per_case_results], dtype=float)
            out[r][m] = float(np.nanmean(vals))
    return out
