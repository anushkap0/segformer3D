import json, csv, os, collections, numpy as np
R = "results"; OUT = f"{R}/final"; os.makedirs(OUT, exist_ok=True)
REG = ("WT", "TC", "ET"); T, E = "test", "external_flip"
RUNS = ["baseline", "continued", "noloss", "noloss_s2", "full", "full_s2", "unet"]
M = {"baseline": "baseline", "continued": "continued", "noloss": "noloss,noloss_s2", "full": "full,full_s2", "unet": "unet"}
PEN = 373.13   # BraTS HD95 penalty for a one-sided empty label; confirm in the challenge metrics code before citing

def load(s, n): return json.load(open(f"{R}/{s}/{n}.json"))["per_case"]
def cdice(pc): return {c: float(np.mean([v[k]["dice"] for k in REG])) for c, v in pc.items()}
def avg(s, names):
    rr = [cdice(load(s, n)) for n in names.split(",")]
    ids = set.intersection(*[set(r) for r in rr])
    return {c: float(np.mean([r[c] for r in rr])) for c in ids}
def paired(a, b, B=10000):
    ids = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in ids]); g = np.random.default_rng(0)
    bs = np.array([d[g.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- Items 1 and 2: paired CIs ----------
ci = [["comparison", "split", "n", "diff_A_minus_B", "ci_low", "ci_high", "ci_includes_0"]]
def add(label, s, A, B):
    n, m, lo, hi = paired(avg(s, A), avg(s, B)); flag = "yes" if lo <= 0 <= hi else "no"
    ci.append([label, s, n, f"{m:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", flag])
    print(f"{label:52s} {s:14s} n={n} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  includes 0: {flag}")
print("=== PAIRED CIs ===")
add("Boundary no-loss (2 seeds) vs Continued", T, "noloss,noloss_s2", "continued")
add("  seed 1 only: no-loss vs Continued", T, "noloss", "continued")
add("  seed 2 only: no-loss vs Continued", T, "noloss_s2", "continued")
add("Boundary + loss (2 seeds) vs Continued", T, "full,full_s2", "continued")
add("Boundary no-loss (2 seeds) vs Baseline", T, "noloss,noloss_s2", "baseline")
add("Boundary + loss vs no-loss (loss effect)", T, "full,full_s2", "noloss,noloss_s2")
add("U-Net vs Boundary + loss (2 seeds)", T, "unet", "full,full_s2")
add("U-Net vs Boundary no-loss (2 seeds)", T, "unet", "noloss,noloss_s2")
add("U-Net vs Continued", T, "unet", "continued")
add("U-Net vs Baseline", T, "unet", "baseline")
add("U-Net vs Baseline", E, "unet", "baseline")
add("U-Net vs Continued", E, "unet", "continued")
add("U-Net vs Boundary no-loss (2 seeds)", E, "unet", "noloss,noloss_s2")
add("U-Net vs Boundary + loss (2 seeds)", E, "unet", "full,full_s2")
csv.writer(open(f"{OUT}/ci_table.csv", "w", newline="")).writerows(ci)

# ---------- Item 3: memorisation check on the 369 verified matches ----------
print("\n=== EXTERNAL GROUPS ===")
ov = json.load(open("logs/ext/overlap.json"))
ver = {r["ext_case"]: r for r in csv.DictReader(open(f"{R}/match_verification_flip.csv")) if float(r["flair_corr"]) > 0.99}
G = collections.defaultdict(list); weak = 0
for c, r in ver.items():
    trained = "train" in ov[c]["splits"]
    cnt = int(r["label_counts_equal"]) == 1; sp = float(r["wt_dice"]) >= 0.999
    weak += int(cnt and not sp)
    G[("trained" if trained else "untrained") + "_" + ("identical" if cnt and sp else "reannotated")].append(c)
G["unmatched"] = [c for c in ov if c not in ver]
print("verified matches:", len(ver), "| groups:", {k: len(v) for k, v in G.items()}, "| total", sum(len(v) for v in G.values()))
print("trained total:", len(G["trained_identical"]) + len(G["trained_reannotated"]), "| untrained total:", len(G["untrained_identical"]) + len(G["untrained_reannotated"]),
      "| label counts equal but WT Dice < 0.999:", weak)
S = {m: avg(E, spec) for m, spec in M.items()}
def bci(x, B=5000):
    g = np.random.default_rng(0); b = [x[g.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)
grp = [["model", "group", "n", "mean_dice", "ci_low", "ci_high"]]
for m in M:
    row = [m.ljust(9)]
    for g_, ids in G.items():
        x = np.array([S[m][c] for c in ids]); lo, hi = bci(x)
        grp.append([m, g_, len(x), f"{x.mean():.4f}", f"{lo:.4f}", f"{hi:.4f}"]); row.append(f"{g_} n={len(x)} {x.mean():.4f}")
    print(" | ".join(row))
csv.writer(open(f"{OUT}/group_table.csv", "w", newline="")).writerows(grp)
def unp(x, y, B=10000):
    g = np.random.default_rng(0)
    b = [x[g.integers(0, len(x), len(x))].mean() - y[g.integers(0, len(y), len(y))].mean() for _ in range(B)]
    return x.mean() - y.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
UT = G["untrained_identical"] + G["untrained_reannotated"]
comps = [("trained_identical vs matched-untrained (all)", G["trained_identical"], UT),
         ("trained_reannotated vs matched-untrained (all)", G["trained_reannotated"], UT),
         ("trained_identical vs untrained_identical", G["trained_identical"], G["untrained_identical"]),
         ("trained_reannotated vs untrained_reannotated", G["trained_reannotated"], G["untrained_reannotated"])]
print("\nMemorisation comparisons (unpaired bootstrap, difference of means)")
mem = [["model", "comparison", "n_a", "n_b", "diff", "ci_low", "ci_high"]]
for m in M:
    for lab, a, b in comps:
        if len(a) >= 5 and len(b) >= 5:
            d, lo, hi = unp(np.array([S[m][c] for c in a]), np.array([S[m][c] for c in b]))
            mem.append([m, lab, len(a), len(b), f"{d:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}"])
            print(f"{m:9s} {lab:48s} {len(a)} vs {len(b)}  {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
csv.writer(open(f"{OUT}/memorisation_table.csv", "w", newline="")).writerows(mem)

# ---------- Item 4: BraTS convention for empty labels ----------
print("\n=== BRATS EMPTY-LABEL CONVENTION (penalty %.2f) ===" % PEN)
def empties(root, ids):
    o = {}
    for c in ids:
        y = np.asarray(np.load(f"{root}/{c}/label.npy", mmap_mode="r"))
        o[c] = {"WT": not (y > 0).any(), "TC": not np.isin(y, [1, 3]).any(), "ET": not (y == 3).any()}
    return o
EMP = {T: empties("data/processed/BraTS2021_FULL", list(load(T, "baseline"))),
       E: empties("data/processed/BraTS2017ext", list(load(E, "baseline")))}
def conv(pc, emp, tag):
    dice = {k: {} for k in REG}; hd = {k: {} for k in REG}; st = collections.Counter(); amb = []
    for c, v in pc.items():
        for k in REG:
            d, h = v[k]["dice"], v[k]["hd95"]
            if np.isfinite(h): dice[k][c] = d; hd[k][c] = h; continue
            if not emp[c][k]:
                dice[k][c] = 0.0; hd[k][c] = PEN; st["GT present, prediction empty"] += 1
            elif np.isfinite(d) and d >= 0.999:
                dice[k][c] = 1.0; hd[k][c] = 0.0; st["both empty"] += 1
            else:
                p = v[k].get("precision")
                if np.isfinite(d) and d < 1e-3:
                    dice[k][c] = 0.0; hd[k][c] = PEN; st["GT empty, prediction present"] += 1
                else:
                    st["AMBIGUOUS"] += 1; amb.append((tag, c, k, d, v[k].get("sensitivity"), p))
    return dice, hd, st, amb
def table(s):
    pcs = {n: load(s, n) for n in RUNS}; ids = list(pcs["baseline"]); res = {}; ambs = []
    for n in RUNS:
        dice, hd, st, amb = conv(pcs[n], EMP[s], n); res[n] = (dice, hd); ambs += amb
        print(f"  {s:14s} {n:10s}", dict(st))
    if ambs:
        w = csv.writer(open(f"{OUT}/ambiguous_{s}.csv", "w", newline="")); w.writerow(["run", "case", "region", "dice", "sensitivity", "precision"]); w.writerows(ambs)
        print(f"  WARNING: {len(ambs)} ambiguous cases in {OUT}/ambiguous_{s}.csv (paste the first lines)")
    fin = [c for c in ids if all(np.isfinite(pcs[n][c][k]["hd95"]) for n in RUNS for k in REG)]
    print(f"  exclusion sensitivity set: {len(fin)} of {len(ids)} cases")
    tab = {}
    for n in RUNS:
        dice, hd = res[n]
        dm = [np.mean(list(dice[k].values())) for k in REG]; hm = [np.mean(list(hd[k].values())) for k in REG]
        tab[(n, "brats")] = dm + [np.mean(dm)] + hm
        agg = json.load(open(f"{R}/{s}/{n}.json"))["aggregate"]; dm2 = [agg[k]["dice"] for k in REG]
        hm2 = [np.mean([pcs[n][c][k]["hd95"] for c in fin]) for k in REG]
        tab[(n, "exclusion")] = dm2 + [np.mean(dm2)] + hm2
    for cv in ("brats", "exclusion"):
        for name, (a, b) in {"noloss_mean2": ("noloss", "noloss_s2"), "full_mean2": ("full", "full_s2")}.items():
            tab[(name, cv)] = list(np.mean([tab[(a, cv)], tab[(b, cv)]], axis=0))
    w = csv.writer(open(f"{OUT}/main_table_{s}.csv", "w", newline=""))
    w.writerow(["model", "convention", "WT_dice", "TC_dice", "ET_dice", "mean_dice", "HD95_WT", "HD95_TC", "HD95_ET"])
    for (n, cv), v in tab.items():
        w.writerow([n, cv] + [f"{x:.4f}" for x in v]); print(f"  {s:14s} {n:13s} {cv:10s}", " ".join(f"{x:.4f}" for x in v))
    json.dump({n: float(tab[(n, "brats")][3]) for n in RUNS}, open(f"{OUT}/mean_dice_{s}.json", "w"))
table(T); table(E)
print("\nwritten to", OUT)
