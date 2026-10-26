import json, csv, numpy as np
R = "results/external_flip"; G = json.load(open("results/external_groups_verified.json"))
cand = {r["ext_case"]: float(r["flair_corr"]) for r in csv.DictReader(open("results/match_verification_flip.csv"))}
failed = {c for c, v in cand.items() if v <= 0.99}
print("candidate matches:", len(cand), "| failed verification:", len(failed), sorted(failed))
def avg(names):
    rr = [{c: np.mean([v[k]["dice"] for k in ("WT", "TC", "ET")]) for c, v in json.load(open(f"{R}/{n}.json"))["per_case"].items()} for n in names.split(",")]
    return {c: np.mean([r[c] for r in rr]) for c in rr[0]}
S = {"baseline": avg("baseline"), "continued": avg("continued"), "noloss": avg("noloss,noloss_s2"), "full": avg("full,full_s2"), "unet": avg("unet")}
sets = {"unseen, all (187)": G["matched_untrained"] + G["unmatched"],
        "unseen without the 12 failed": [c for c in G["matched_untrained"] + G["unmatched"] if c not in failed],
        "unmatched without the 12 failed": [c for c in G["unmatched"] if c not in failed]}
print({k: len(v) for k, v in sets.items()})
g = np.random.default_rng(0)
print("\nmean Dice per model")
for lab, ids in sets.items():
    print(f"  {lab:32s}", *[f"{m} {np.mean([S[m][c] for c in ids]):.4f}" for m in S])
print("\nU-Net minus each transformer (paired, 95% CI)")
for lab, ids in sets.items():
    for m in ("baseline", "continued", "noloss", "full"):
        d = np.array([S["unet"][c] - S[m][c] for c in ids]); bs = [d[g.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        print(f"  {lab:32s} n={len(d)} vs {m:9s} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  includes 0: {'yes' if lo <= 0 <= hi else 'no'}")
