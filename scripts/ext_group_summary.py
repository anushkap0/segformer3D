import json, os, sys
import numpy as np
ov = json.load(open("logs/ext/overlap.json"))
groups = {"matched_train": [c for c, v in ov.items() if "train" in v["splits"]],
          "not_in_train": [c for c, v in ov.items() if "train" not in v["splits"]]}
print({k: len(v) for k, v in groups.items()})
for path in sys.argv[1:]:
    pc = json.load(open(path))["per_case"]
    row = [os.path.basename(path)]
    for g, cs in groups.items():
        cs = [c for c in cs if c in pc]
        if not cs:
            row.append(f"{g}: n=0"); continue
        d = np.array([np.mean([pc[c][r]["dice"] for r in ("WT", "TC", "ET")]) for c in cs])
        s = np.array([np.mean([pc[c][r]["surface_dice_1mm"] for r in ("WT", "TC", "ET")]) for c in cs])
        row.append(f"{g}: n={len(cs)} dice {d.mean():.3f} surfDice {s.mean():.3f}")
    print(" | ".join(row))
