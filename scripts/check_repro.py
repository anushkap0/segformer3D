import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "data/processed/BraTS2017_repro"; m = json.load(open(D + "/split_manifest.json"))
print({k: len(v) for k, v in m.items()})
bad = [c for k in ("train", "val") for c in m[k] if not (os.path.exists(f"{D}/{c}/image.npy") and os.path.exists(f"{D}/{c}/label.npy"))]
print("missing:", len(bad), bad[:3])
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
for r, c in enumerate(m["val"][:2]):
    x = np.load(f"{D}/{c}/image.npy"); y = np.load(f"{D}/{c}/label.npy")
    print(c, x.shape, x.dtype, "range", round(float(x.min()), 3), round(float(x.max()), 3), "labels", np.unique(y).tolist(),
          "voxels 1/2/3:", [int((y == k).sum()) for k in (1, 2, 3)])
    z = int(np.argmax((y == 3).sum((1, 2))))
    for j, (im, t) in enumerate([(x[3][z], "FLAIR"), (x[1][z], "T1ce")]):
        ax[r, j].imshow(im, cmap="gray", origin="lower")
        ax[r, j].imshow(np.ma.masked_equal(y[z], 0), alpha=.45, cmap="jet", origin="lower")
        ax[r, j].set_title(f"{c} {t}"); ax[r, j].axis("off")
plt.savefig("results/repro_check.png", dpi=110, bbox_inches="tight")
print("val cases with no enhancing tumour:", sum(1 for c in m["val"] if not (np.load(f"{D}/{c}/label.npy") == 3).any()))
print("saved results/repro_check.png")
