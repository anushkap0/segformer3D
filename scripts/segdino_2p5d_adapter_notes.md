# Adapting SegDINO into the Phase 4 comparison table

SegDINO (arXiv:2606.17972, code: https://github.com/script-Yang/SegDINO) is a **2D**
segmentation framework: a frozen DINOv3 backbone + a lightweight decoder (Token Pyramid
Adaptation + Scale-Aware Decoding). It was benchmarked on 2D medical datasets (TN3K,
Kvasir-SEG, ISIC) and 2D/video natural-image datasets (MSD-2D slices, VMD-D, ViSha), and
its own PanCT dataset. There is no 3D variant.

You have two honest options if you want it in your comparison table:

## Option A (recommended): keep it out of the 3D Dice/HD95 table entirely
Report SegDINO only in the optional Phase 4b 2D secondary experiment (ISIC or fundus,
per ODFormer's datasets), where it's evaluated on its native 2D task. Do not put its
numbers in the same table as SegFormer3D/U-Net3D/TokenSeg on BraTS -- the tasks aren't
comparable and a shared table implies they are.

## Option B: build a 2.5D adapter (more work, more defensible than nothing)
If you want a same-dataset comparison point, run SegDINO slice-wise on the axial slices
of your BraTS/MSD volumes and stack the 2D predictions back into a volume:

1. Clone the official repo: `git clone https://github.com/script-Yang/SegDINO.git`
   and its DINOv3 dependency: `git clone https://github.com/facebookresearch/dinov3.git`
2. Write a slicing dataloader that extracts axial slices from your preprocessed
   `image.npy` volumes (4 modalities -> you'll need to either train 4 separate 2D
   SegDINO instances, or concatenate modalities into extra input channels and adapt
   SegDINO's stem conv, since DINOv3 expects 3-channel RGB-like input by default).
3. Run per-slice inference, stack predictions along the slice axis back into a volume.
4. Evaluate with the SAME `src/utils/metrics.py` functions used for the 3D models, on
   the SAME voxel spacing, so HD95 numbers are computed identically.
5. Report inference latency and memory separately for 2.5D (N forward passes per
   volume) vs. true 3D (1 forward pass per volume) -- this is itself a relevant
   efficiency comparison point given the "lightweight" framing of your project.

This file intentionally does not include adapter code, since the concatenated-modality
question in step 2 is a real design decision you should make deliberately (e.g. training
4 independent single-modality SegDINO models and late-fusing logits is a cleaner,
more defensible choice than hacking a 4-channel stem onto a backbone pretrained on
3-channel natural images).
