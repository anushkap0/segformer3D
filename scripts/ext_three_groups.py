import json, numpy as np
ov = json.load(open("logs/ext/overlap.json"))
G = {"trained":           [c for c,v in ov.items() if "train" in v["splits"]],
     "matched_untrained": [c for c,v in ov.items() if v["matches"] and "train" not in v["splits"]],
     "unmatched":         [c for c,v in ov.items() if not v["matches"]]}
print({k: len(v) for k, v in G.items()}, "total", sum(len(v) for v in G.values()))
multi = sum(1 for v in ov.values() if len(v["matches"]) > 1)
mixed = sum(1 for v in ov.values() if len(v["splits"]) > 1)
leq   = [all(v["label_equal"]) for v in ov.values() if v["matches"]]
print("cases with >1 match:", multi, "| matches spanning several splits:", mixed,
      "| matched cases with identical label counts:", f"{sum(leq)}/{len(leq)}")
json.dump(G, open("results/external_groups.json", "w"), indent=1)

names = ["baseline","continued","noloss","full","unet"]
def score(n):
    fs = [f"results/external/{n}.json"] + ([f"results/external/{n}_s2.json"] if n in ("noloss","full") else [])
    runs = [json.load(open(f))["per_case"] for f in fs]
    return {c: np.mean([np.mean([r[c][k]["dice"] for k in ("WT","TC","ET")]) for r in runs]) for c in runs[0]}
S = {n: score(n) for n in names}
rng = np.random.default_rng(0)
def ci(x):
    b = [x[rng.integers(0,len(x),len(x))].mean() for _ in range(5000)]
    return np.percentile(b,2.5), np.percentile(b,97.5)
print("\nmean Dice per group (95% CI)")
for n in names:
    row = [n.ljust(9)]
    for g, ids in G.items():
        x = np.array([S[n][c] for c in ids]); lo, hi = ci(x)
        row.append(f"{g}: {x.mean():.4f} [{lo:.4f},{hi:.4f}]")
    print(" | ".join(row))
print("\nU-Net minus each transformer (paired)")
for g, ids in {**G, "unseen_175": G["matched_untrained"]+G["unmatched"]}.items():
    for n in ("baseline","continued","noloss","full"):
        d = np.array([S["unet"][c]-S[n][c] for c in ids]); lo, hi = ci(d)
        print(f"{g:18s} vs {n:9s} n={len(d)} {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
