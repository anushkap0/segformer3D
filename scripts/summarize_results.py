import json, sys
import numpy as np
for path in sys.argv[1:]:
    pc = json.load(open(path))["per_case"]; cases = list(pc)
    print(path, f"({len(cases)} cases)")
    for r in ("WT", "TC", "ET"):
        row = []
        for m in ("dice", "hd95", "surface_dice_1mm"):
            if m not in pc[cases[0]][r]: continue
            v = np.array([pc[c][r][m] for c in cases], float); v = v[~np.isnan(v)]
            row.append(f"{m}: mean {v.mean():.4f} median {np.median(v):.4f}")
        print(f"  {r}  " + " | ".join(row))
