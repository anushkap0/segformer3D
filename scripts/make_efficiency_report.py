import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

with open("logs/efficiency.json") as f:
    efficiency = json.load(f)

# Display name, test-result JSON, efficiency entry
models = [
    ("Baseline", "v2_phase1_test.json", "v2_phase1_baseline"),
    ("Boundary seed 1", "v2_phase2_test.json", "v2_phase2_boundary"),
    ("Boundary seed 2", "v2_phase2_s2_test.json", "v2_phase2_boundary_s2"),
    ("No-loss seed 1", "v2_abl_noloss_test.json", "v2_abl_noloss"),
    ("No-loss seed 2", "v2_abl_noloss_s2_test.json", "v2_abl_noloss_s2"),
    ("Continued", "v2_continued_test.json", "v2_phase1_continued"),
    ("3D U-Net", "unet_test.json", "phase4_baselines"),
]

rows = []
for name, test_file, eff_key in models:
    test_path = Path("logs/test") / test_file
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test results: {test_path}")
    if eff_key not in efficiency:
        raise KeyError(f"Missing efficiency entry: {eff_key}")

    with open(test_path) as f:
        test = json.load(f)

    cases = test.get("per_case", {})
    if not cases:
        raise ValueError(f"No per_case data in {test_path}")

    dice = []
    for case in cases.values():
        vals = [case[r]["dice"] for r in ("WT", "TC", "ET")]
        dice.append(float(np.mean(vals)))

    e = efficiency[eff_key]
    required = ("params_M", "gmacs", "ms_median", "peak_mem_MB")
    missing = [k for k in required if k not in e]
    if missing:
        raise ValueError(f"{eff_key} missing efficiency fields: {missing}")

    rows.append({
        "model": name,
        "test_cases": len(dice),
        "macro_dice": float(np.mean(dice)),
        "params_M": e["params_M"],
        "gmacs": e["gmacs"],
        "latency_ms_median": e["ms_median"],
        "peak_memory_MB": e["peak_mem_MB"],
    })

csv_path = OUT / "efficiency_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Aggregate seeds for architecture-level plotting.
groups = {
    "Baseline": ["Baseline"],
    "Boundary": ["Boundary seed 1", "Boundary seed 2"],
    "Boundary (no loss)": ["No-loss seed 1", "No-loss seed 2"],
    "Continued": ["Continued"],
    "3D U-Net": ["3D U-Net"],
}

fig, ax = plt.subplots(figsize=(10, 6))
offsets = {
    "Baseline": (5, 8),
    "Boundary": (-8, 13),
    "Boundary (no loss)": (8, -16),
    "Continued": (5, -17),
    "3D U-Net": (-55, 8),
}

for label, members in groups.items():
    selected = [r for r in rows if r["model"] in members]
    x = float(np.mean([r["gmacs"] for r in selected]))
    scores = np.array([r["macro_dice"] for r in selected])
    y = float(scores.mean())
    # For two-seed groups, show seed-to-seed standard deviation.
    err = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0

    ax.errorbar(
        x, y, yerr=err, fmt="o", capsize=4,
        markersize=7, label=label
    )
    dx, dy = offsets[label]
    ax.annotate(
        f"{label}\n{y:.4f}",
        (x, y), xytext=(dx, dy), textcoords="offset points",
        fontsize=9
    )

ax.set_xscale("log")
ax.set_xlabel("GMACs per 128³ input (log scale)")
ax.set_ylabel("Mean per-case macro Dice (WT, TC, ET)")
ax.set_title("Segmentation Accuracy vs. Inference Compute")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plot_path = OUT / "dice_vs_gmacs.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n=== EFFICIENCY SUMMARY ===")
print(f"{'Model':<22} {'Dice':>8} {'GMACs':>10} "
      f"{'Params(M)':>10} {'Latency(ms)':>13} {'Memory(MB)':>12}")
for r in rows:
    print(f"{r['model']:<22} {r['macro_dice']:>8.5f} "
          f"{r['gmacs']:>10.3f} {r['params_M']:>10.3f} "
          f"{r['latency_ms_median']:>13.3f} "
          f"{r['peak_memory_MB']:>12.1f}")

print(f"\nCSV saved:  {csv_path}")
print(f"Plot saved: {plot_path}")
print("Plot error bars show seed-to-seed SD where two seeds are available.")
