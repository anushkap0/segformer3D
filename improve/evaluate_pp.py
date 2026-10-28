"""Step 3 (CPU): evaluate a configuration ONCE on the test split or the external set.

Writes one row per case with Dice, HD95 (BraTS rule) and surface Dice at 1 mm for WT,
TC and ET, plus empty-mask flags, so stats.py can compare configurations case by case.

Examples
  # sanity check: must reproduce the frozen main table (plain pass, default threshold)
  python improve/evaluate_pp.py --pred_dirs preds/full_s1/test --key p_plain --pp none ...
  # tuned ensemble with flip-TTA
  python improve/evaluate_pp.py --pred_dirs preds/full_s1/test preds/full_s2/test --key p_tta \
      --pp pp/full_ens_tta.json --data_dir data/processed/BraTS \
      --manifest data/processed/BraTS/split_manifest.json --split test --out_csv results_v2/test_full_ens_tta.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import REGIONS, case_list, case_metrics, labels_to_regions, load_case, load_probs
from postprocess import PP, apply_pp


def _one(args_tuple):
    cid, pred_dirs, key, pp_dict, data_dir = args_tuple
    _, label, spacing = load_case(data_dir, cid)
    probs = load_probs(pred_dirs, cid, key)
    pred = apply_pp(probs, PP(**pp_dict))
    m = case_metrics(pred, labels_to_regions(label), spacing)
    m["case_id"] = cid
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dirs", nargs="+", required=True)
    ap.add_argument("--key", default="p_tta", choices=["p_plain", "p_tta"])
    ap.add_argument("--pp", default="none", help="JSON from tune_val.py, or 'none' for the default 0.5 threshold")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split")
    ap.add_argument("--cases_file")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pp = PP().to_dict() if args.pp == "none" else json.load(open(args.pp))["pp"]
    cases = case_list(args.manifest, args.split, args.cases_file)
    jobs = [(c, args.pred_dirs, args.key, pp, args.data_dir) for c in cases]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    cols = ["case_id"] + [k for k in rows[0] if k != "case_id"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(args.out_csv.replace(".csv", "_config.json"), "w") as f:
        json.dump({"pred_dirs": args.pred_dirs, "key": args.key, "pp": pp, "n": len(rows)}, f, indent=2)

    summ = {k: float(np.mean([r[k] for r in rows])) for k in cols if k != "case_id"}
    print(f"n = {len(rows)}   mean Dice = {summ['dice_mean']:.4f}   " +
          "  ".join(f"{r}: Dice {summ['dice_' + r]:.4f} HD95 {summ['hd95_' + r]:.2f} SD1 {summ['sd1_' + r]:.4f}" for r in REGIONS))
    for r in REGIONS:
        fp = sum(1 for x in rows if x[f"empty_gt_{r}"] and not x[f"empty_pred_{r}"])
        miss = sum(1 for x in rows if not x[f"empty_gt_{r}"] and x[f"empty_pred_{r}"])
        print(f"  {r}: false positives on empty ground truth = {fp}, missed regions = {miss}")


if __name__ == "__main__":
    main()
