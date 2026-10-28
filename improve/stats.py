"""Step 4 (CPU): paired comparison of two per-case result files.

Reports mean and median difference (A - B) with paired bootstrap 95% CIs over cases,
the Wilcoxon signed-rank p value, the win rate, and a non-inferiority test of A
against B with margin delta: A is non-inferior if the lower CI bound of A - B is above
-delta (the margin must be fixed BEFORE looking at the test results; see PROTOCOL.md).

Example
  python improve/stats.py --a results_v2/test_full_ens_tta.csv --b results_v2/test_unet_tta.csv \
      --metric dice_mean --margin 0.01 --label "BR-SegFormer3D ensemble vs U-Net"
  # restrict to a subset, e.g. the 187 zero-shot cases
  ... --subset groups/zero_shot_187.txt
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.stats import wilcoxon


def read(path, metric):
    with open(path) as f:
        return {r["case_id"]: float(r[metric]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", default="dice_mean")
    ap.add_argument("--subset", help="text file of case ids to restrict to")
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--lower_is_better", action="store_true", help="for HD95: report B - A so positive still favours A")
    args = ap.parse_args()

    a, b = read(args.a, args.metric), read(args.b, args.metric)
    ids = sorted(set(a) & set(b))
    if args.subset:
        keep = {l.strip() for l in open(args.subset) if l.strip()}
        ids = [i for i in ids if i in keep]
    d = np.array([(b[i] - a[i]) if args.lower_is_better else (a[i] - b[i]) for i in ids])
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(d), size=(args.n_boot, len(d)))
    means, medians = d[idx].mean(1), np.median(d[idx], 1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    mlo, mhi = np.percentile(medians, [2.5, 97.5])
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"{args.label}  metric={args.metric}  n={len(d)}" + ("  (differences are B - A: positive favours A)" if args.lower_is_better else ""))
    print(f"  mean diff   = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'detectable' if lo > 0 or hi < 0 else 'not detectable'}")
    print(f"  median diff = {np.median(d):+.4f}  95% CI [{mlo:+.4f}, {mhi:+.4f}]  Wilcoxon p = {p:.2g}  "
          f"A better in {100 * np.mean(d > 0):.0f}% of cases")
    print(f"  non-inferiority (margin {args.margin}): {'YES' if lo > -args.margin else 'no'} "
          f"(lower bound {lo:+.4f} vs {-args.margin:+.4f});  superiority: {'YES' if lo > 0 else 'no'}")


if __name__ == "__main__":
    main()
