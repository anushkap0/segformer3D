import json, glob, numpy as np, nibabel as nib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
raw, proc = "/workspace/Brats2017", "data/processed/BraTS2017ext"
ids = ["BRATS_001", "BRATS_002", "BRATS_003"]
COL = {"edema": (0.15, 0.75, 0.25), "core": (0.15, 0.35, 0.95), "enh": (0.90, 0.15, 0.15)}
RAW = {1: "edema", 2: "core", 3: "enh"}      # MSD labels: 1 edema, 2 non-enhancing, 3 enhancing
INT = {1: "core", 2: "edema", 3: "enh"}      # internal labels after remapping
def overlay(lab, m):
    o = np.zeros(lab.shape + (4,))
    for v, n in m.items(): o[lab == v, :3] = COL[n]; o[lab == v, 3] = 0.5
    return o.transpose(1, 0, 2)
def chan(c, k):
    f = glob.glob(f"{raw}/imagesTr/{c}_{k:04d}.nii*")
    return nib.load(f[0]).get_fdata() if f else nib.load(glob.glob(f"{raw}/imagesTr/{c}.nii*")[0]).get_fdata()[..., k]
titles = ["Raw FLAIR", "Raw T1ce", "Processed FLAIR", "Processed T1ce"]
fig, ax = plt.subplots(3, 4, figsize=(12, 9.4))
for r, c in enumerate(ids):
    lab = np.rint(nib.load(glob.glob(f"{raw}/labelsTr/{c}.nii*")[0]).get_fdata()).astype(int)
    y = np.load(f"{proc}/{c}/label.npy"); x = np.load(f"{proc}/{c}/image.npy")
    zr = int(np.argmax((lab == 3).sum((0, 1)))); zp = int(np.argmax((y == 3).sum((0, 1))))
    panels = [(chan(c, 0)[..., zr], lab[..., zr], RAW), (chan(c, 2)[..., zr], lab[..., zr], RAW),
              (x[3][..., zp], y[..., zp], INT), (x[1][..., zp], y[..., zp], INT)]
    for j, (im, lb, m) in enumerate(panels):
        a = ax[r, j]; a.imshow(im.T, cmap="gray", origin="lower"); a.imshow(overlay(lb, m), origin="lower"); a.axis("off")
        if r == 0: a.set_title(titles[j], fontsize=11)
        if j == 0: a.text(-0.03, 0.5, c, transform=a.transAxes, rotation=90, va="center", ha="right", fontsize=11)
fig.legend(handles=[Patch(color=COL["edema"], label="Edema"), Patch(color=COL["core"], label="Necrotic / non-enhancing core"),
                    Patch(color=COL["enh"], label="Enhancing tumour")], loc="lower center", ncol=3, frameon=False, fontsize=11)
plt.tight_layout(rect=(0.02, 0.05, 1, 1))
plt.savefig("results/label_check.png", dpi=300, bbox_inches="tight"); plt.savefig("results/label_check.pdf", bbox_inches="tight")
