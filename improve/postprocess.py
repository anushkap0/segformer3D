"""Post-processing applied to a (3, D, H, W) probability map -> binary region masks.

All parameters are chosen on the validation split only (tune_val.py) and then frozen.

  thr_wt, thr_tc, thr_et  per-region probability thresholds
  wt_min_cc               remove WT connected components smaller than this (voxels)
  et_min_vol              if the total predicted ET volume is below this, relabel it as
                          tumour core (standard BraTS rule: removes spurious ET blobs on
                          cases without enhancing tumour, which the BraTS HD95 rule
                          penalises with 373.13 mm)
  hierarchy               enforce ET within TC within WT
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label as cc_label


@dataclass
class PP:
    thr_wt: float = 0.5
    thr_tc: float = 0.5
    thr_et: float = 0.5
    wt_min_cc: int = 0
    et_min_vol: int = 0
    hierarchy: bool = False

    def to_dict(self):
        return asdict(self)


IDENTITY = PP()


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or not mask.any():
        return mask
    lab, n = cc_label(mask, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def apply_pp(probs: np.ndarray, pp: PP) -> np.ndarray:
    wt = probs[0] > pp.thr_wt
    tc = probs[1] > pp.thr_tc
    et = probs[2] > pp.thr_et
    wt = remove_small_components(wt, pp.wt_min_cc)
    if pp.hierarchy:
        tc &= wt
        et &= tc
    if pp.et_min_vol > 0 and 0 < et.sum() < pp.et_min_vol:
        tc |= et          # relabel the enhancing voxels as tumour core
        et = np.zeros_like(et)
    return np.stack([wt, tc, et], axis=0)
