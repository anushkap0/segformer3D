"""Shared utilities for the inference-time improvement pipeline.

Everything downstream of `infer_probs.py` works on saved probability maps, so tuning,
ensembling and statistics never need the GPU again.

Data layout (same as src/evaluate.py):  <data_dir>/<case_id>/{image.npy,label.npy,spacing.json}
Label convention: 0 background, 1 NCR/NET, 2 oedema, 3 enhancing tumour.
Region channels:  0 = WT {1,2,3}, 1 = TC {1,3}, 2 = ET {3}.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

REGIONS = ("WT", "TC", "ET")
HD95_PENALTY = 373.13  # diagonal of 240 x 240 x 155 voxels at 1 mm: official BraTS 2021 rule


# ----------------------------------------------------------------------------- data
def load_case(data_dir: str, case_id: str):
    d = os.path.join(data_dir, case_id)
    image = np.load(os.path.join(d, "image.npy"))
    label = np.load(os.path.join(d, "label.npy"))
    with open(os.path.join(d, "spacing.json")) as f:
        spacing = tuple(float(s) for s in json.load(f)["spacing"])
    return image, label, spacing


def labels_to_regions(label: np.ndarray) -> np.ndarray:
    return np.stack([np.isin(label, [1, 2, 3]), np.isin(label, [1, 3]), label == 3], axis=0)


def case_list(manifest: str | None, split: str | None, cases_file: str | None) -> List[str]:
    """Cases from a split manifest (json {split: [ids]}) or from a plain text file."""
    if cases_file:
        with open(cases_file) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]
    with open(manifest) as f:
        return list(json.load(f)[split])


# ----------------------------------------------------------------------- probabilities
def save_probs(path: str, probs: Dict[str, np.ndarray]) -> None:
    """Stores each (3, D, H, W) probability map as uint8 (resolution 1/255 is far below
    any threshold we tune) in a compressed npz."""
    np.savez_compressed(path, **{k: np.clip(np.rint(v * 255.0), 0, 255).astype(np.uint8) for k, v in probs.items()})


def load_probs(pred_dirs: Sequence[str], case_id: str, key: str) -> np.ndarray:
    """Averages the probability maps of several runs (an ensemble when len > 1)."""
    acc = None
    for d in pred_dirs:
        with np.load(os.path.join(d, f"{case_id}.npz")) as z:
            p = z[key].astype(np.float32) / 255.0
        acc = p if acc is None else acc + p
    return acc / len(pred_dirs)


# ------------------------------------------------------------------------- metrics
def dice_brats(p: np.ndarray, t: np.ndarray) -> float:
    ps, ts = int(p.sum()), int(t.sum())
    if ps == 0 and ts == 0:
        return 1.0
    if ps == 0 or ts == 0:
        return 0.0
    return 2.0 * float(np.logical_and(p, t).sum()) / (ps + ts)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Border voxels with a 6-connected structuring element, as in medpy.metric.binary
    (used by src/utils/metrics.py), so HD95 matches the frozen evaluation."""
    from scipy.ndimage import binary_erosion, generate_binary_structure
    return np.logical_xor(mask, binary_erosion(mask, structure=generate_binary_structure(3, 1), iterations=1))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing) -> Tuple[np.ndarray, np.ndarray]:
    """Distances (mm) from surface voxels of a to the surface of b and vice versa."""
    from scipy.ndimage import distance_transform_edt
    sa, sb = _surface(a), _surface(b)
    dt_b = distance_transform_edt(~sb, sampling=spacing)
    dt_a = distance_transform_edt(~sa, sampling=spacing)
    return dt_b[sa], dt_a[sb]


def hd95_brats(p: np.ndarray, t: np.ndarray, spacing) -> float:
    """95th-percentile Hausdorff distance (medpy definition) with the official BraTS 2021
    rule for empty masks (both empty -> 0, exactly one empty -> 373.13)."""
    ps, ts = bool(p.any()), bool(t.any())
    if not ps and not ts:
        return 0.0
    if not ps or not ts:
        return HD95_PENALTY
    try:  # identical to the frozen evaluation when medpy is installed
        from medpy.metric.binary import hd95 as medpy_hd95
        return float(medpy_hd95(p, t, voxelspacing=spacing))
    except ImportError:  # same definition as medpy: 95th percentile of the pooled distances
        d_ab, d_ba = _surface_distances(p, t, spacing)
        return float(np.percentile(np.hstack((d_ab, d_ba)), 95))


def surface_dice(p: np.ndarray, t: np.ndarray, spacing, tol: float = 1.0) -> float:
    """Normalised surface Dice at tolerance `tol` mm (Nikolov et al. 2021). Empty
    masks follow the same convention as Dice. Check against src/utils/metrics.py
    (surface_dice_1mm) before reporting: the two should agree to ~1e-3."""
    ps, ts = bool(p.any()), bool(t.any())
    if not ps and not ts:
        return 1.0
    if not ps or not ts:
        return 0.0
    d_ab, d_ba = _surface_distances(p, t, spacing)
    return float(((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba)))


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing, with_distances: bool = True) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, r in enumerate(REGIONS):
        out[f"dice_{r}"] = dice_brats(pred[i], gt[i])
        if with_distances:
            out[f"hd95_{r}"] = hd95_brats(pred[i], gt[i], spacing)
            out[f"sd1_{r}"] = surface_dice(pred[i], gt[i], spacing)
        out[f"empty_gt_{r}"] = int(not gt[i].any())
        out[f"empty_pred_{r}"] = int(not pred[i].any())
    out["dice_mean"] = float(np.mean([out[f"dice_{r}"] for r in REGIONS]))
    if with_distances:
        out["sd1_mean"] = float(np.mean([out[f"sd1_{r}"] for r in REGIONS]))
    return out
