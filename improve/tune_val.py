"""Step 2 (CPU): choose post-processing parameters on the VALIDATION split only.

Coordinate search that maximises validation mean Dice (BraTS rule for empty masks):
WT threshold and minimum component size on WT Dice, TC threshold on TC Dice, ET
threshold and minimum ET volume on ET Dice, then whether to enforce ET within TC within
WT on mean Dice. The result is written to a JSON that evaluate_pp.py must use unchanged.

Example (ensemble of two seeds, flip-TTA probabilities):
  python improve/tune_val.py --pred_dirs preds/full_s1/val preds/full_s2/val --key p_tta \
      --data_dir data/processed/BraTS --manifest data/processed/BraTS/split_manifest.json \
      --split val --out pp/full_ens_tta.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import case_list, dice_brats, labels_to_regions, load_case, load_probs
from postprocess import PP, apply_pp, remove_small_components

THR = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
WT_CC = [0, 100, 250, 500, 1000, 2000]
ET_MIN = [0, 25, 50, 100, 200, 300, 500, 750, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dirs", nargs="+", required=True)
    ap.add_argument("--key", default="p_tta", choices=["p_plain", "p_tta"])
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split", default="val")
    ap.add_argument("--cases_file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split != "val" and not args.cases_file:
        raise SystemExit("Tuning must use the validation split.")

    cases = case_list(args.manifest, args.split, args.cases_file)
    P, G = [], []
    for cid in cases:
        P.append(np.rint(load_probs(args.pred_dirs, cid, args.key) * 255).astype(np.uint8))
        G.append(labels_to_regions(load_case(args.data_dir, cid)[1]))
    print(f"loaded {len(cases)} validation cases")
    q = lambda t: int(np.floor(t * 255 + 1e-9))

    # WT: threshold x minimum connected-component size
    best_wt = max(((np.mean([dice_brats(remove_small_components(p[0] > q(t), cc), g[0]) for p, g in zip(P, G)]), t, cc)
                   for t in THR for cc in WT_CC))
    # TC: threshold
    best_tc = max(((np.mean([dice_brats(p[1] > q(t), g[1]) for p, g in zip(P, G)]), t) for t in THR))
    # ET: threshold x minimum total ET volume
    def et_dice(p, g, t, vmin):
        e = p[2] > q(t)
        if vmin > 0 and 0 < e.sum() < vmin:
            e = np.zeros_like(e)
        return dice_brats(e, g[2])
    best_et = max(((np.mean([et_dice(p, g, t, v) for p, g in zip(P, G)]), t, v) for t in THR for v in ET_MIN))

    def mean_dice(pp):
        return float(np.mean([np.mean([dice_brats(m, gg) for m, gg in zip(apply_pp(p / 255.0, pp), g)]) for p, g in zip(P, G)]))

    cands = [PP(best_wt[1], best_tc[1], best_et[1], best_wt[2], best_et[2], h) for h in (False, True)]
    scores = [mean_dice(c) for c in cands]
    chosen = cands[int(np.argmax(scores))]
    base_plain = mean_dice(PP())
    result = {
        "pred_dirs": args.pred_dirs, "key": args.key, "n_val": len(cases),
        "pp": chosen.to_dict(),
        "val_mean_dice_default_pp": base_plain,
        "val_mean_dice_tuned_pp": max(scores),
        "per_region_best": {"WT": best_wt, "TC": best_tc, "ET": best_et},
    }
    if result["val_mean_dice_tuned_pp"] <= base_plain:
        result["pp"] = PP().to_dict()
        result["note"] = "tuned post-processing did not beat the default on validation; default kept"
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
