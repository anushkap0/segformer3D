import os
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = "/workspace/Brats2017"
OUT = "/workspace/segformer3D/outputs/label_mapping/brats2017_mapping_check.png"

with open(os.path.join(ROOT, "dataset.json")) as f:
    meta = json.load(f)

print("Dataset:", meta["name"])
print("Modalities:", meta["modality"])
print("Labels:", meta["labels"])

cases = ["BRATS_001", "BRATS_002", "BRATS_003"]

# BraTS 2017 labels already use the internal convention.
label_map = {0: 0, 1: 1, 2: 2, 3: 3}

cmap = ListedColormap(["black", "#e6e6e6", "#2878b5", "#e53935"])
fig, axes = plt.subplots(len(cases), 3, figsize=(12, 12))

for row, case in enumerate(cases):
    # Channel 0 is FLAIR according to dataset.json.
    image_path = os.path.join(ROOT, "imagesTr", case + "_0000.nii.gz")
    label_path = os.path.join(ROOT, "labelsTr", case + ".nii.gz")

    if not os.path.isfile(image_path) or not os.path.isfile(label_path):
        raise FileNotFoundError(
            f"Missing image or label for {case}: {image_path}, {label_path}"
        )

    image = nib.load(image_path).get_fdata(dtype=np.float32)
    label = np.rint(nib.load(label_path).get_fdata()).astype(np.uint8)

    if image.shape != label.shape:
        raise ValueError(
            f"{case}: image shape {image.shape} != label shape {label.shape}"
        )

    # Select slice containing the largest number of labelled voxels.
    counts = np.count_nonzero(label > 0, axis=(0, 1))
    z = int(np.argmax(counts))

    mapped = np.zeros_like(label)
    for src, dst in label_map.items():
        mapped[label == src] = dst

    assert np.array_equal(label, mapped), f"Mapping mismatch in {case}"

    axes[row, 0].imshow(image[:, :, z].T, cmap="gray", origin="lower")
    axes[row, 0].set_title(f"{case}: FLAIR, slice {z}")

    axes[row, 1].imshow(
        label[:, :, z].T, cmap=cmap, vmin=0, vmax=3,
        origin="lower", interpolation="nearest"
    )
    axes[row, 1].set_title("Original labels")

    axes[row, 2].imshow(
        mapped[:, :, z].T, cmap=cmap, vmin=0, vmax=3,
        origin="lower", interpolation="nearest"
    )
    axes[row, 2].set_title("Remapped labels")

    for ax in axes[row]:
        ax.axis("off")

fig.suptitle(
    "BraTS 2017 label-mapping sanity check\n"
    "0=background, 1=edema, 2=non-enhancing, 3=enhancing",
    fontsize=13
)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)

print("PASS: all three cases preserved their labels.")
print("Saved figure:", OUT)
