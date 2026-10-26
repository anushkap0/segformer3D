import glob, json
import numpy as np
REG = ("WT", "TC", "ET")
def load(phase):
    out = {}
    for f in sorted(glob.glob(f"logs/cv/fold_*/{phase}_test.json")):
        out[int(f.split("fold_")[1].split("/")[0])] = json.load(open(f))["aggregate"]
    return out
p1, p2 = load("phase1"), load("phase2")
folds = sorted(set(p1) & set(p2))
print("folds with both models:", folds)
def st(v):
    v = np.array(v)
    return f"{v.mean():.4f} +/- {(v.std(ddof=1) if len(v) > 1 else 0):.4f}"
for metric in ("dice", "hd95"):
    print(f"\n== {metric} ==")
    for r in REG:
        a = [p1[k][r][metric] for k in folds]
        b = [p2[k][r][metric] for k in folds]
        print(f"{r}: baseline {st(a)} | boundary {st(b)} | diff {st([y - x for x, y in zip(a, b)])}")
    a = [np.mean([p1[k][r][metric] for r in REG]) for k in folds]
    b = [np.mean([p2[k][r][metric] for r in REG]) for k in folds]
    print(f"mean: baseline {st(a)} | boundary {st(b)} | diff {st([y - x for x, y in zip(a, b)])}")
