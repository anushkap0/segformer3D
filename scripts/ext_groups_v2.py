import json, csv, sys, numpy as np
R = sys.argv[1]
ov = json.load(open("logs/ext/overlap.json"))
ver = {x["ext_case"] for x in csv.DictReader(open("results/match_verification_flip.csv")) if float(x["flair_corr"]) > 0.99}
G = {"trained":           [c for c in ver if "train" in ov[c]["splits"]],
     "matched_untrained": [c for c in ver if "train" not in ov[c]["splits"]],
     "unmatched":         [c for c in ov if c not in ver]}
print({k: len(v) for k, v in G.items()}, "total", sum(len(v) for v in G.values()))
def score(n):
    fs = [f"{R}/{n}.json"] + ([f"{R}/{n}_s2.json"] if n in ("noloss", "full") else [])
    runs = [json.load(open(f))["per_case"] for f in fs]
    return {c: np.mean([np.mean([r[c][k]["dice"] for k in ("WT","TC","ET")]) for r in runs]) for c in runs[0]}
names = ["baseline", "continued", "noloss", "full", "unet"]
S = {n: score(n) for n in names}
rng = np.random.default_rng(0)
def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
print("\nmean Dice per group (95% CI), source:", R)
for n in names:
    row = [n.ljust(9)]
    for g, ids in G.items():
        x = np.array([S[n][c] for c in ids]); lo, hi = ci(x)
        row.append(f"{g}: {x.mean():.4f} [{lo:.4f},{hi:.4f}]")
    print(" | ".join(row))
unseen = G["matched_untrained"] + G["unmatched"]
print("\nU-Net minus each transformer (paired)")
for g, ids in {**G, "unseen": unseen}.items():
    for n in ("baseline", "continued", "noloss", "full"):
        d = np.array([S["unet"][c] - S[n][c] for c in ids]); lo, hi = ci(d)
        print(f"{g:18s} vs {n:9s} n={len(d)} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
json.dump(G, open("results/external_groups_verified.json", "w"), indent=1)
