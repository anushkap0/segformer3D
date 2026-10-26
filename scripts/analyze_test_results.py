import json
from pathlib import Path
import numpy as np

ROOT = Path("logs/test")
BOOTSTRAPS = 20000
SEED = 42

FILES = {
    "Baseline": "v2_phase1_test.json",
    "Boundary seed 1": "v2_phase2_test.json",
    "Boundary seed 2": "v2_phase2_s2_test.json",
    "No-loss seed 1": "v2_abl_noloss_test.json",
    "No-loss seed 2": "v2_abl_noloss_s2_test.json",
    "Continued": "v2_continued_test.json",
    "U-Net": "unet_test.json",
}

def load_model(filename):
    path = ROOT / filename
    if not path.exists():
        print(f"WARNING: Missing {path}")
        return None

    with path.open() as f:
        data = json.load(f)

    cases = data.get("per_case", {})
    if not cases:
        print(f"WARNING: No per_case entries in {path}")
        return None
    return cases

def macro_dice(case):
    regions = [case[r]["dice"] for r in ("WT", "TC", "ET")]
    return float(np.mean(regions))

def bootstrap_ci(differences, rng):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    samples = rng.integers(0, n, size=(BOOTSTRAPS, n))
    boot_means = differences[samples].mean(axis=1)
    return (
        differences.mean(),
        np.quantile(boot_means, 0.025),
        np.quantile(boot_means, 0.975),
    )

models = {
    name: load_model(filename)
    for name, filename in FILES.items()
}
models = {name: cases for name, cases in models.items()
          if cases is not None}

print("\n=== TEST RESULTS: MEAN DICE AND HD95 ===")
print("HD95 is in the distance unit used by your evaluation pipeline.")
print(f"{'Model':<22} {'N':>4} {'MacroDice':>10} "
      f"{'WT HD95':>10} {'TC HD95':>10} {'ET HD95':>10}")

for name, cases in models.items():
    dice_values = []
    hd_values = {r: [] for r in ("WT", "TC", "ET")}

    for case in cases.values():
        dice_values.append(macro_dice(case))
        for region in hd_values:
            value = case[region].get("hd95")
            if value is not None and np.isfinite(value):
                hd_values[region].append(float(value))

    means = [
        np.mean(hd_values[r]) if hd_values[r] else float("nan")
        for r in ("WT", "TC", "ET")
    ]
    print(f"{name:<22} {len(dice_values):>4} "
          f"{np.mean(dice_values):>10.5f} "
          f"{means[0]:>10.4f} {means[1]:>10.4f} {means[2]:>10.4f}")

print("\n=== PAIRED BOOTSTRAP 95% CIs: MACRO DICE DIFFERENCES ===")
comparisons = [
    ("Boundary seed 1", "No-loss seed 1"),
    ("Boundary seed 2", "No-loss seed 2"),
    ("No-loss seed 1", "Baseline"),
    ("No-loss seed 2", "Baseline"),
    ("Boundary seed 1", "Baseline"),
    ("U-Net", "Baseline"),
]

for model_a, model_b in comparisons:
    if model_a not in models or model_b not in models:
        print(f"SKIP: {model_a} vs {model_b} (file missing)")
        continue

    a, b = models[model_a], models[model_b]
    common = sorted(set(a) & set(b))
    if not common:
        print(f"SKIP: {model_a} vs {model_b} (no common case IDs)")
        continue

    differences = [
        macro_dice(a[c]) - macro_dice(b[c])
        for c in common
    ]
    rng = np.random.default_rng(SEED)
    mean, low, high = bootstrap_ci(differences, rng)

    print(
        f"{model_a} - {model_b}: n={len(common)}, "
        f"diff={mean:+.5f}, 95% CI [{low:+.5f}, {high:+.5f}]"
    )

print("\n=== PAIRED REGION-WISE HD95 DIFFERENCES ===")
print("Negative HD95 difference favors the first model.")

for model_a, model_b in [
    ("Boundary seed 1", "No-loss seed 1"),
    ("Boundary seed 2", "No-loss seed 2"),
]:
    if model_a not in models or model_b not in models:
        continue

    a, b = models[model_a], models[model_b]
    common = sorted(set(a) & set(b))

    for region in ("WT", "TC", "ET"):
        pairs = []
        for c in common:
            x = a[c][region].get("hd95")
            y = b[c][region].get("hd95")
            if x is None or y is None:
                continue
            if np.isfinite(x) and np.isfinite(y):
                pairs.append(float(x) - float(y))

        if pairs:
            rng = np.random.default_rng(SEED)
            mean, low, high = bootstrap_ci(pairs, rng)
            print(
                f"{model_a} - {model_b} | {region}: "
                f"n={len(pairs)}, diff={mean:+.4f}, "
                f"95% CI [{low:+.4f}, {high:+.4f}]"
            )
        else:
            print(f"{model_a} - {model_b} | {region}: no valid HD95 pairs")
