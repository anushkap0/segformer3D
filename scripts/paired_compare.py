import json, sys
import numpy as np
from scipy.stats import wilcoxon
a_path, b_path = sys.argv[1], sys.argv[2]
a = json.load(open(a_path))["per_case"]; b = json.load(open(b_path))["per_case"]
cases = sorted(set(a) & set(b))
metrics = [k for k in ("dice", "hd95", "surface_dice_1mm") if k in a[cases[0]]["WT"] and k in b[cases[0]]["WT"]]
print(f"A={a_path}\nB={b_path}\npaired cases: {len(cases)}  (diff = B - A)")
rng = np.random.default_rng(0)
def report(name, x, y):
    ok = ~(np.isnan(x) | np.isnan(y)); d = (y - x)[ok]
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try: p = wilcoxon(d).pvalue
    except ValueError: p = float("nan")
    print(f"{name:18s} n={len(d):3d} mean={d.mean():+.4f} median={np.median(d):+.4f} 95%CI=[{lo:+.4f},{hi:+.4f}] p={p:.4f}")
for r in ("WT", "TC", "ET"):
    for m in metrics:
        report(f"{r} {m}", np.array([a[c][r][m] for c in cases], float), np.array([b[c][r][m] for c in cases], float))
x = np.array([np.mean([a[c][r]["dice"] for r in ("WT", "TC", "ET")]) for c in cases])
y = np.array([np.mean([b[c][r]["dice"] for r in ("WT", "TC", "ET")]) for c in cases])
report("mean dice", x, y)
