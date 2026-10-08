"""BraTS-style dataset loader.

Expects preprocessed data from `scripts/prepare_brats.py`: each case is a folder
containing `image.npy` (C=4, D, H, W; z-score normalized per modality) and
`label.npy` (D, H, W integer labels: 0=background, 1=NCR/NET, 2=ED, 4=ET in the
raw BraTS convention, remapped to 0/1/2/3 during prep) plus a `spacing.json` with
the voxel spacing in mm.

The standard BraTS evaluation regions are derived from the raw labels as:
  WT (whole tumor)  = labels {1, 2, 3}   (i.e. raw {1, 2, 4})
  TC (tumor core)   = labels {1, 3}      (i.e. raw {1, 4})
  ET (enhancing)    = labels {3}         (i.e. raw {4})
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.ndimage import binary_erosion
except ImportError:  # pragma: no cover
    binary_erosion = None


def labels_to_brats_regions(label: np.ndarray) -> np.ndarray:
    """label: (D, H, W) with values {0,1,2,3} (remapped from raw {0,1,2,4}).
    Returns (3, D, H, W) float32 multi-label array: [WT, TC, ET].
    """
    wt = np.isin(label, [1, 2, 3]).astype(np.float32)
    tc = np.isin(label, [1, 3]).astype(np.float32)
    et = (label == 3).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def compute_boundary_target(region_target: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Derives a boundary map from a multi-label region target by taking each region's
    interior-minus-eroded-interior (morphological gradient), then taking the union
    across regions. region_target: (C, D, H, W) -> returns (1, D, H, W) float32.
    """
    if binary_erosion is None:
        raise ImportError("scipy is required for boundary target computation")
    c = region_target.shape[0]
    boundary = np.zeros(region_target.shape[1:], dtype=np.float32)
    struct = np.ones((3, 3, 3), dtype=bool)
    for i in range(c):
        mask = region_target[i].astype(bool)
        if mask.sum() == 0:
            continue
        eroded = binary_erosion(mask, structure=struct, iterations=thickness)
        edge = np.logical_and(mask, np.logical_not(eroded))
        boundary = np.logical_or(boundary, edge)
    return boundary.astype(np.float32)[None, ...]


class BraTSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        case_ids: List[str],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        train: bool = True,
        return_boundary_target: bool = False,
    ):
        self.data_dir = data_dir
        self.case_ids = case_ids
        self.patch_size = patch_size
        self.train = train
        self.return_boundary_target = return_boundary_target

    def __len__(self) -> int:
        return len(self.case_ids)

    def _load_case(self, case_id: str):
        case_dir = os.path.join(self.data_dir, case_id)
        image = np.load(os.path.join(case_dir, "image.npy"))  # (4, D, H, W)
        label = np.load(os.path.join(case_dir, "label.npy"))  # (D, H, W)
        with open(os.path.join(case_dir, "spacing.json")) as f:
            spacing = json.load(f)["spacing"]
        return image, label, spacing

    def _random_crop(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, d, h, w = image.shape
        pd, ph, pw = self.patch_size
        pd, ph, pw = min(pd, d), min(ph, h), min(pw, w)

        # oversample foreground: 70% of crops centered near a nonzero label voxel
        if self.train and label.sum() > 0 and np.random.rand() < 0.7:
            fg_coords = np.argwhere(label > 0)
            cz, cy, cx = fg_coords[np.random.randint(len(fg_coords))]
        else:
            cz = np.random.randint(0, d)
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)

        z0 = np.clip(cz - pd // 2, 0, max(d - pd, 0))
        y0 = np.clip(cy - ph // 2, 0, max(h - ph, 0))
        x0 = np.clip(cx - pw // 2, 0, max(w - pw, 0))

        image_crop = image[:, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        label_crop = label[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]

        # pad if the volume was smaller than patch_size along any axis
        pad_d, pad_h, pad_w = pd - image_crop.shape[1], ph - image_crop.shape[2], pw - image_crop.shape[3]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image_crop = np.pad(image_crop, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)))
            label_crop = np.pad(label_crop, ((0, pad_d), (0, pad_h), (0, pad_w)))

        return image_crop, label_crop

    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # random flips along each spatial axis
        for axis in (1, 2, 3):
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis - 1).copy()
        # random intensity scale + shift (per-modality), a standard BraTS augmentation
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1, size=(image.shape[0], 1, 1, 1))
            shift = np.random.uniform(-0.1, 0.1, size=(image.shape[0], 1, 1, 1))
            image = image * scale + shift
        # gaussian noise
        if np.random.rand() < 0.15:
            image = image + np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
        return image.astype(np.float32), label

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[idx]
        image, label, spacing = self._load_case(case_id)

        if self.train:
            image, label = self._random_crop(image, label)
            image, label = self._augment(image, label)
        else:
            # center crop / pad to patch_size for val (sliding-window inference is
            # handled separately in evaluate.py for full-volume metrics)
            image, label = self._random_crop(image, label)

        region_target = labels_to_brats_regions(label)

        sample = {
            "image": torch.from_numpy(image).float(),
            "target": torch.from_numpy(region_target).float(),
            "case_id": case_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32),
        }
        if self.return_boundary_target:
            boundary_target = compute_boundary_target(region_target)
            sample["boundary_target"] = torch.from_numpy(boundary_target).float()
        return sample


def load_split_manifest(manifest_path: str) -> Dict[str, List[str]]:
    with open(manifest_path) as f:
        return json.load(f)
