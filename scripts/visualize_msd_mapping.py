import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = "/tmp/msd_inspect/Task01_BrainTumour"
OUT = "/workspace/segformer3D/outputs/label_mapping/msd_task01_mapping_check.png"

CASES = ["BRATS_457", "BRATS_306", "BRATS_206"]

# Internal convention expected by this project:
# 1 = NCR/NET, 2 = ED, 3 = ET
LABEL_MAP = {0: 0, 1: 2, 2: 1, 3: 3}

cmap = ListedColormap(["black", "#e6e6e6", "#2878b5", "#e53935"])
fig, axes = plt.subplots(len(CASES), 4, figsize=(16, 12))

for row, case in enumerate(CASES):
    image_path = os.path.join(ROOT, "imagesTr", case + ".nii.gz")
    label_path = os.path.join(ROOT, "labelsTr", case + ".nii.gz")

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    # Metadata says channel 0 is FLAIR.
    image = image_nii.get_fdata(dtype=np.float32)[..., 0]
    label = np.rint(label_nii.get_fdata()).astype(np.uint8)

    if image.shape != label.shape:
        raise ValueError(f"{case}: image/mask shape mismatch")

    mapped = np.zeros_like(label)
    for src, dst in LABEL_MAP.items():
        mapped[label == src] = dst

    if not np.array_equal(np.unique(label), np.array(sorted(LABEL_MAP))):
        raise ValueError(f"{case}: unexpected source label values")

    # Pick the axial slice with the most labelled voxels.
    z = int(np.argmax(np.count_nonzero(label > 0, axis=(0, 1))))

    original_slice = label[:, :, z].T
    mapped_slice = mapped[:, :, z].T

    axes[row, 0].imshow(image[:, :, z].T, cmap="gray", origin="lower")
    axes[row, 0].set_title(f"{case}: FLAIR, z={z}")

    axes[row, 1].imshow(
        original_slice, cmap=cmap, vmin=0, vmax=3,
        origin="lower", interpolation="nearest"
    )
    axes[row, 1].set_title("Original MSD labels")

    axes[row, 2].imshow(
        mapped_slice, cmap=cmap, vmin=0, vmax=3,
        origin="lower", interpolation="nearest"
    )
    axes[row, 2].set_title("Mapped labels")

    # WT = labels 1,2,3; TC = labels 2,3 in the mapped convention; ET = 3.
    regions = np.zeros_like(mapped_slice)
    regions[np.isin(mapped_slice, [1, 2, 3])] = 1
    regions[np.isin(mapped_slice, [1, 3])] = 2
    regions[mapped_slice == 3] = 3

    axes[row, 3].imshow(
        regions, cmap=cmap, vmin=0, vmax=3,
        origin="lower", interpolation="nearest"
    )
    axes[row, 3].set_title("WT / TC / ET regions")

    for ax in axes[row]:
        ax.axis("off")

fig.suptitle(
    "MSD Task01 BrainTumour: label mapping inspection\n"
    "Source: 1=edema, 2=non-enhancing, 3=enhancing | "
    "Internal: 1=NCR/NET, 2=ED, 3=ET",
    fontsize=13
)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)

print("PASS: generated visualization for", len(CASES), "cases")
print("Saved:", OUT)
print("Mapping:", LABEL_MAP)
