import json, numpy as np
ov = json.load(open("logs/ext/overlap.json"))
E, O = "data/processed/BraTS2017ext", "data/processed/BraTS2021_FULL"
def dice(a, b):
    s = a.sum() + b.sum(); return 1.0 if s == 0 else 2 * (a & b).sum() / s
rows = []
for c, v in ov.items():
    if not v["matches"]: continue
    m = v["matches"][0]
    xe = np.load(f"{E}/{c}/image.npy", mmap_mode="r"); xo = np.load(f"{O}/{m}/image.npy", mmap_mode="r")
    if xe.shape != xo.shape:
        rows.append((c, m, 0, np.nan, np.nan, np.nan, v["label_equal"][0])); continue
    corr = np.corrcoef(np.asarray(xe[3]).ravel(), np.asarray(xo[3]).ravel())[0, 1]
    ye, yo = np.load(f"{E}/{c}/label.npy"), np.load(f"{O}/{m}/label.npy")
    rows.append((c, m, 1, corr, dice(ye > 0, yo > 0), dice(ye == 3, yo == 3), v["label_equal"][0]))
print("label values  ext:", np.unique(ye), " 2021:", np.unique(yo))
R = np.array([(r[2], r[3], r[4], r[5], r[6]) for r in rows], dtype=float)
same, corr, wt, et, leq = R.T
print("matched cases:", len(R), "| same array shape:", int(same.sum()))
print("FLAIR corr  min/median:", np.nanmin(corr).round(4), np.nanmedian(corr).round(4), "| corr>0.99:", int((corr > 0.99).sum()))
for name, mask in (("label counts equal", leq == 1), ("label counts differ", leq == 0)):
    print(f"{name:20s} n={int(mask.sum())}  WT-Dice(ext vs 2021) median {np.nanmedian(wt[mask]):.3f}  ET-Dice median {np.nanmedian(et[mask]):.3f}")
with open("results/match_verification.csv", "w") as f:
    f.write("ext_case,match_2021,same_shape,flair_corr,wt_dice,et_dice,label_counts_equal\n")
    for r in rows: f.write(",".join(map(str, r)) + "\n")
