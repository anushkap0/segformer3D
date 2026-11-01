import json, itertools, collections, numpy as np
ov = json.load(open("logs/ext/overlap.json"))
E, O = "data/processed/BraTS2017ext", "data/processed/BraTS2021_FULL"
flips = [f for r in range(4) for f in itertools.combinations(range(3), r)]   # 8 flip combinations
def dice(a, b):
    s = a.sum() + b.sum(); return 1.0 if s == 0 else 2 * (a & b).sum() / s
rows = []
for c, v in ov.items():
    if not v["matches"]: continue
    m = v["matches"][0]
    xe = np.asarray(np.load(f"{E}/{c}/image.npy", mmap_mode="r")[3], dtype=np.float32)
    xo = np.asarray(np.load(f"{O}/{m}/image.npy", mmap_mode="r")[3], dtype=np.float32)
    best = (-2, ())
    for f in flips:
        a = np.flip(xe, axis=f) if f else xe
        r = np.corrcoef(a.ravel(), xo.ravel())[0, 1]
        if r > best[0]: best = (r, f)
    r, f = best
    ye = np.load(f"{E}/{c}/label.npy"); yo = np.load(f"{O}/{m}/label.npy")
    if f: ye = np.flip(ye, axis=f)
    rows.append((c, m, str(f).replace(",", ";"), r, dice(ye > 0, yo > 0), dice(ye == 3, yo == 3), int(v["label_equal"][0])))
R = np.array([(r[3], r[4], r[5], r[6]) for r in rows], dtype=float)
corr, wt, et, leq = R.T
print("matched cases:", len(rows))
print("best-flip FLAIR corr  min/median:", corr.min().round(4), np.median(corr).round(4), "| corr>0.99:", int((corr > 0.99).sum()), "| corr>0.9:", int((corr > 0.9).sum()))
print("best flip axes:", collections.Counter(r[2] for r in rows).most_common())
for name, mask in (("label counts equal", leq == 1), ("label counts differ", leq == 0)):
    print(f"{name:20s} n={int(mask.sum())}  corr median {np.median(corr[mask]):.3f}  WT-Dice median {np.median(wt[mask]):.3f}  ET-Dice median {np.median(et[mask]):.3f}")
with open("results/match_verification_flip.csv", "w") as f:
    f.write("ext_case,match_2021,best_flip,flair_corr,wt_dice,et_dice,label_counts_equal\n")
    for r in rows: f.write(",".join(map(str, r)) + "\n")
