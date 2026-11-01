import json, sys, numpy as np
D = "data/processed/BraTS2017_repro"
for path in sys.argv[1:]:
    pc = json.load(open(path))["per_case"]
    G = {}
    for c in pc:
        y = np.load(f"{D}/{c}/label.npy"); G[c] = {"WT": (y > 0).any(), "TC": np.isin(y, [1, 3]).any(), "ET": (y == 3).any()}
    print(path, "| cases:", len(pc), "| with empty ET:", sum(1 for c in G if not G[c]["ET"]))
    for lab, skip in (("ours (empty ground truth counted)", False), ("authors-style (empty ground truth skipped)", True)):
        r = {k: np.mean([pc[c][k]["dice"] for c in pc if (G[c][k] or not skip)]) for k in ("WT", "TC", "ET")}
        print(f"  {lab:44s}", *[f"{k} {v:.4f}" for k, v in r.items()], "| mean", f"{np.mean(list(r.values())):.4f}")
