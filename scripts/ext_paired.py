import json
import numpy as np
ov = json.load(open("logs/ext/overlap.json"))
G = {"not_in_train": [c for c, v in ov.items() if "train" not in v["splits"]],
     "no_match": [c for c, v in ov.items() if not v["matches"]]}
def load(n):
    return json.load(open(f"logs/ext/{n}_ext.json"))["per_case"]
def vec(pc, cs, m):
    return np.array([np.mean([pc[c][r][m] for r in ("WT", "TC", "ET")]) for c in cs], float)
pairs = [("v2_abl_noloss", "v2_phase2"), ("v2_continued", "v2_phase2"), ("v2_phase1", "v2_continued"), ("v2_phase1", "unet"), ("v2_phase2", "unet")]
rng = np.random.default_rng(0)
for g, cs in G.items():
    print(f"\n== group {g} (n={len(cs)})")
    for a, b in pairs:
        pa, pb = load(a), load(b)
        row = [f"{a} -> {b}"]
        for m in ("dice", "surface_dice_1mm"):
            d = vec(pb, cs, m) - vec(pa, cs, m)
            boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row.append(f"{m}: {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]")
        print("  " + " | ".join(row))
