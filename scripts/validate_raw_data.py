"""Scans a raw BraTS directory and checks every case for:
  1. All 5 required files present (4 modalities + seg)
  2. Each file actually loads with nibabel (catches truncated/corrupted downloads)
  3. Each modality volume has nonzero data (catches empty/all-zero files, which
     load fine but are useless)
  4. All modalities + seg share the same shape (catches misaligned/mismatched files)

Produces a clean report: which cases pass, which fail and why. Use this BEFORE
running prepare_brats.py on a large download, rather than discovering corruption
mid-preprocessing or (worse) silently training on a few broken cases.

Usage:
    python scripts/validate_raw_data.py --raw_dir data/raw/BraTS2021_full
    python scripts/validate_raw_data.py --raw_dir data/raw/BraTS2021_full --legacy_names
    python scripts/validate_raw_data.py --raw_dir data/raw/BraTS2021_full --quarantine_dir data/raw/BraTS2021_full_quarantine

--quarantine_dir moves (not deletes) failing case folders out of raw_dir into a
separate folder, so prepare_brats.py can be pointed straight at raw_dir afterward
without re-touching anything, and nothing is destructively deleted -- you can
inspect/restore quarantined cases later if you want.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from glob import glob

import nibabel as nib
import numpy as np

MODALITY_SUFFIXES_NEW = {
    "t1": "t1n.nii.gz", "t1ce": "t1c.nii.gz", "t2": "t2w.nii.gz", "flair": "t2f.nii.gz",
}
MODALITY_SUFFIXES_LEGACY = {
    "t1": "t1.nii.gz", "t1ce": "t1ce.nii.gz", "t2": "t2.nii.gz", "flair": "flair.nii.gz",
}


def find_file(case_dir: str, suffix: str):
    candidates = glob(os.path.join(case_dir, f"*{suffix}"))
    return candidates[0] if candidates else None


def validate_case(case_dir: str, case_id: str, legacy_names: bool) -> dict:
    suffixes = MODALITY_SUFFIXES_LEGACY if legacy_names else MODALITY_SUFFIXES_NEW
    result = {"case_id": case_id, "ok": True, "errors": []}

    paths = {}
    for mod, suffix in suffixes.items():
        path = find_file(case_dir, suffix)
        if path is None:
            result["ok"] = False
            result["errors"].append(f"missing {mod} file (*{suffix})")
        else:
            paths[mod] = path

    seg_path = find_file(case_dir, "seg.nii.gz")
    if seg_path is None:
        result["ok"] = False
        result["errors"].append("missing seg file (*seg.nii.gz)")
    else:
        paths["seg"] = seg_path

    if not result["ok"]:
        return result  # no point loading files that don't exist

    shapes = {}
    for name, path in paths.items():
        try:
            nii = nib.load(path)
            data = nii.get_fdata()
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"{name} failed to load ({type(e).__name__}: {e})")
            continue

        shapes[name] = data.shape

        if name != "seg" and np.count_nonzero(data) == 0:
            result["ok"] = False
            result["errors"].append(f"{name} is all-zero (empty/corrupted file)")

        if np.isnan(data).any():
            result["ok"] = False
            result["errors"].append(f"{name} contains NaN values")

    if len(set(shapes.values())) > 1:
        result["ok"] = False
        result["errors"].append(f"shape mismatch across files: {shapes}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--legacy_names", action="store_true")
    parser.add_argument("--quarantine_dir", default=None,
                         help="if set, move failing case folders here instead of leaving them in raw_dir")
    parser.add_argument("--report_path", default=None,
                         help="where to write the JSON validation report (default: <raw_dir>/validation_report.json)")
    args = parser.parse_args()

    case_dirs = sorted([d for d in glob(os.path.join(args.raw_dir, "*")) if os.path.isdir(d)])
    print(f"Found {len(case_dirs)} case folders in {args.raw_dir}\n")

    passed, failed = [], []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        result = validate_case(case_dir, case_id, args.legacy_names)
        if result["ok"]:
            passed.append(case_id)
        else:
            failed.append(result)
            print(f"FAIL {case_id}: {'; '.join(result['errors'])}")

    print(f"\n{'='*60}")
    print(f"PASSED: {len(passed)} / {len(case_dirs)}")
    print(f"FAILED: {len(failed)} / {len(case_dirs)}")
    print(f"{'='*60}")

    report_path = args.report_path or os.path.join(args.raw_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({"passed": passed, "failed": failed, "total": len(case_dirs)}, f, indent=2)
    print(f"\nWrote validation report to {report_path}")

    if args.quarantine_dir and failed:
        os.makedirs(args.quarantine_dir, exist_ok=True)
        for result in failed:
            src = os.path.join(args.raw_dir, result["case_id"])
            dst = os.path.join(args.quarantine_dir, result["case_id"])
            if os.path.exists(src):
                shutil.move(src, dst)
        print(f"Moved {len(failed)} failing case folders to {args.quarantine_dir}")
        print(f"{args.raw_dir} now contains only the {len(passed)} passing cases -- "
              f"safe to point prepare_brats.py at it directly.")
    elif failed:
        print(f"\n{len(failed)} cases failed validation but were NOT moved (no --quarantine_dir given). "
              f"prepare_brats.py will currently either error or silently skip these -- "
              f"re-run with --quarantine_dir to cleanly separate them first.")


if __name__ == "__main__":
    main()
    