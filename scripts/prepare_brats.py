"""Preprocess raw BraTS (or MSD) NIfTI volumes into the .npy format expected by
`src/data/brats_dataset.py`, and write a case-level train/val/test split manifest.

Usage:
    python scripts/prepare_brats.py --raw_dir data/raw/BraTS --out_dir data/processed/BraTS
    python scripts/prepare_brats.py --raw_dir data/raw/MSD_Task01 --out_dir data/processed/MSD_Task01 --dataset msd

Expected raw BraTS layout (official challenge structure):
    raw_dir/
      BraTS-GLI-00000-000/
        BraTS-GLI-00000-000-t1n.nii.gz
        BraTS-GLI-00000-000-t1c.nii.gz
        BraTS-GLI-00000-000-t2w.nii.gz
        BraTS-GLI-00000-000-t2f.nii.gz
        BraTS-GLI-00000-000-seg.nii.gz
      ...

If your download uses the older naming convention (*_t1.nii.gz / *_t1ce.nii.gz /
*_t2.nii.gz / *_flair.nii.gz / *_seg.nii.gz), pass --legacy_names.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from glob import glob

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


MODALITY_SUFFIXES_NEW = {
    "t1": "t1n.nii.gz", "t1ce": "t1c.nii.gz", "t2": "t2w.nii.gz", "flair": "t2f.nii.gz",
}
MODALITY_SUFFIXES_LEGACY = {
    "t1": "t1.nii.gz", "t1ce": "t1ce.nii.gz", "t2": "t2.nii.gz", "flair": "flair.nii.gz",
}
MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]


def find_modality_file(case_dir: str, case_id: str, suffix: str) -> str:
    candidates = glob(os.path.join(case_dir, f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No file matching *{suffix} in {case_dir}")
    return candidates[0]


def resample_to_spacing(volume: np.ndarray, orig_spacing, target_spacing=(1.0, 1.0, 1.0), order: int = 1) -> np.ndarray:
    zoom_factors = [o / t for o, t in zip(orig_spacing, target_spacing)]
    return zoom(volume, zoom_factors, order=order)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float32)
    mean, std = volume[mask].mean(), volume[mask].std()
    std = std if std > 1e-6 else 1.0
    out = (volume - mean) / std
    out[~mask] = 0.0
    return out.astype(np.float32)


def crop_to_nonzero(image: np.ndarray, label: np.ndarray):
    # image: (4, D, H, W), label: (D, H, W)
    nonzero_mask = (image != 0).any(axis=0)
    coords = np.argwhere(nonzero_mask)
    if len(coords) == 0:
        return image, label
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    return image[:, z0:z1, y0:y1, x0:x1], label[z0:z1, y0:y1, x0:x1]


def remap_labels(label: np.ndarray) -> np.ndarray:
    # raw BraTS labels {0,1,2,4} -> contiguous {0,1,2,3}
    out = label.copy()
    out[out == 4] = 3
    return out.astype(np.uint8)


def process_case(case_dir: str, case_id: str, out_dir: str, legacy_names: bool, target_spacing):
    suffixes = MODALITY_SUFFIXES_LEGACY if legacy_names else MODALITY_SUFFIXES_NEW
    modality_volumes = []
    orig_spacing = None
    for mod in MODALITY_ORDER:
        path = find_modality_file(case_dir, case_id, suffixes[mod])
        nii = nib.load(path)
        vol = nii.get_fdata().astype(np.float32)
        if orig_spacing is None:
            orig_spacing = nii.header.get_zooms()[:3]
        vol = resample_to_spacing(vol, orig_spacing, target_spacing, order=1)
        vol = zscore_normalize(vol)
        modality_volumes.append(vol)

    seg_path = find_modality_file(case_dir, case_id, "seg.nii.gz")
    seg_nii = nib.load(seg_path)
    seg = seg_nii.get_fdata().astype(np.uint8)
    seg = resample_to_spacing(seg, orig_spacing, target_spacing, order=0)  # nearest-neighbor for labels
    seg = remap_labels(seg)

    image = np.stack(modality_volumes, axis=0)  # (4, D, H, W)
    image, seg = crop_to_nonzero(image, seg)

    case_out_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_out_dir, exist_ok=True)
    np.save(os.path.join(case_out_dir, "image.npy"), image.astype(np.float32))
    np.save(os.path.join(case_out_dir, "label.npy"), seg.astype(np.uint8))
    with open(os.path.join(case_out_dir, "spacing.json"), "w") as f:
        json.dump(
    {
        "spacing": [float(x) for x in target_spacing],
        "orig_spacing": [float(x) for x in orig_spacing],
    },
    f,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset", default="brats", choices=["brats", "msd"])
    parser.add_argument("--legacy_names", action="store_true")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    case_dirs = sorted([d for d in glob(os.path.join(args.raw_dir, "*")) if os.path.isdir(d)])
    case_ids = [os.path.basename(d) for d in case_dirs]

    print(f"Found {len(case_ids)} cases in {args.raw_dir}")
    for case_dir, case_id in zip(case_dirs, case_ids):
        try:
            process_case(case_dir, case_id, args.out_dir, args.legacy_names, tuple(args.target_spacing))
            print(f"  processed {case_id}")
        
        except Exception as e:
            print(f"  SKIPPED {case_id}: {type(e).__name__}: {e}")

    processed_ids = sorted(
        d for d in os.listdir(args.out_dir) if os.path.isdir(os.path.join(args.out_dir, d))
    )
    random.Random(args.seed).shuffle(processed_ids)
    n = len(processed_ids)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    manifest = {
        "train": processed_ids[:n_train],
        "val": processed_ids[n_train:n_train + n_val],
        "test": processed_ids[n_train + n_val:],
    }
    with open(os.path.join(args.out_dir, "split_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote split manifest: {len(manifest['train'])} train / "
          f"{len(manifest['val'])} val / {len(manifest['test'])} test -> "
          f"{os.path.join(args.out_dir, 'split_manifest.json')}")


if __name__ == "__main__":
    main()
