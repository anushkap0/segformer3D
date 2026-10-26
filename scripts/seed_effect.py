import json, sys
import numpy as np
w1, w2, n1, n2 = [json.load(open(p))["per_case"] for p in sys.argv[1:5]]
cases = sorted(set(w1) & set(w2) & set(n1) & set(n2))
print("cases:", len(cases), "| order: with_loss_s1 with_loss_s2 noloss_s1 noloss_s2")
rng = np.random.default_rng(0)
def vec(pc, metric, region):
    if region == "mean":
        return np.array([np.mean([pc[c][r][metric] for r in ("WT", "TC", "ET")]) for c in cases], float)
    return np.array([pc[c][region][metric] for c in cases], float)
def ci(d):
    d = d[~np.isnan(d)]
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi
for metric in ("dice", "surface_dice_1mm"):
    print(f"\n== {metric}  (effect = with loss - without loss)")
    for region in ("WT", "TC", "ET", "mean"):
        e1 = vec(w1, metric, region) - vec(n1, metric, region)
        e2 = vec(w2, metric, region) - vec(n2, metric, region)
        m, lo, hi = ci((e1 + e2) / 2)
        sw = vec(w2, metric, region) - vec(w1, metric, region)
        sn = vec(n2, metric, region) - vec(n1, metric, region)
        print(f"{region:4s} effect seed1 {e1.mean():+.4f} | seed2 {e2.mean():+.4f} | 2-seed mean {m:+.4f} CI[{lo:+.4f},{hi:+.4f}] | seed-to-seed diff: with-loss {sw.mean():+.4f}, no-loss {sn.mean():+.4f}")
