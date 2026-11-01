import json, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
eff = json.load(open("logs/efficiency.json")); brats = "--brats" in sys.argv
def md(n):
    if brats: return json.load(open("results/final/mean_dice_test.json"))[n]
    a = json.load(open(f"results/test/{n}.json"))["aggregate"]; return float(np.mean([a[k]["dice"] for k in ("WT", "TC", "ET")]))
M = [("Baseline", ["baseline"], "v2_phase1_baseline", (14, -10), "#1f77b4", "left"),
     ("Continued", ["continued"], "v2_phase1_continued", (14, 8), "#ff7f0e", "left"),
     ("Boundary, no loss", ["noloss", "noloss_s2"], "v2_abl_noloss", (20, -22), "#2ca02c", "left"),
     ("Boundary + loss", ["full", "full_s2"], "v2_phase2_boundary", (20, 20), "#d62728", "left"),
     ("3D U-Net", ["unet"], "phase4_baselines", (-14, -22), "#9467bd", "right")]
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for name, runs, k, off, col, ha in M:
    v = [md(r) for r in runs]; m = float(np.mean(v)); x = eff[k]["gmacs"] * {"Boundary, no loss": 0.93, "Boundary + loss": 1.07}.get(name, 1.0)
    yerr = [[m - min(v)], [max(v) - m]] if len(v) > 1 else None
    ax.errorbar(x, m, yerr=yerr, fmt="o", color=col, capsize=4, markersize=7, lw=1.6)
    ax.annotate(name, (x, m), textcoords="offset points", xytext=off, ha=ha, fontsize=10, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.7))
ax.set_xscale("log"); ax.set_xlim(11, 700)
ax.set_xticks([15.5, 43, 481]); ax.set_xticklabels(["15.5", "43", "481"]); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("GMACs (128³ input, log scale)"); ax.set_ylabel("Test mean Dice"); ax.grid(alpha=.3)
lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.0006, hi); lo, hi = ax.get_ylim(); ax.text(0.02, 0.97, f"y-axis spans {hi - lo:.3f} Dice", transform=ax.transAxes, va="top", fontsize=8, color="0.35")
print("y-axis span:", round(hi - lo, 4))
plt.savefig("results/dice_vs_gmacs.png", dpi=300, bbox_inches="tight"); plt.savefig("results/dice_vs_gmacs.pdf", bbox_inches="tight")
