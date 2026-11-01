import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
eff = json.load(open("logs/efficiency.json"))
runs = [("Baseline", "v2_phase1_baseline", "logs/v2_phase1_val.json"),
        ("Baseline +200 epochs", "v2_phase1_baseline", "logs/v2_continued_val.json"),
        ("Boundary, no loss", "v2_phase2_boundary", "logs/v2_abl_noloss_val.json"),
        ("Boundary", "v2_phase2_boundary", "logs/v2_phase2_val.json"),
        ("3D U-Net", "phase4_baselines", "logs/unet_val.json")]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for i, (label, key, path) in enumerate(runs):
    pc = json.load(open(path))["per_case"]
    v = np.array([np.mean([pc[c][r]["dice"] for r in ("WT", "TC", "ET")]) for c in pc])
    boot = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    x = eff[key]["gmacs"] * (1 + 0.04 * i)
    ax.errorbar(x, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", capsize=3, label=label)
ax.set_xscale("log"); ax.set_xlabel("GMACs per 128^3 patch (log scale)"); ax.set_ylabel("Mean Dice (validation, 95% CI)")
ax.legend(fontsize=7); fig.tight_layout()
os.makedirs("figures", exist_ok=True); fig.savefig("figures/dice_vs_gmacs.png", dpi=200)
print("saved figures/dice_vs_gmacs.png")
