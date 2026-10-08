# Lightweight Transformer-Based 3D Segmentation with Boundary Refinement

Reproduction of **SegFormer3D** (Perera et al., CVPR-W 2024, arXiv:2404.10156) plus a novel
**3D boundary-aware refinement module** inspired by the boundary branch in **PFF-Net**
(Liu et al., Frontiers in Computer Science 2025), adapted from 2D to 3D volumetric data.

This repo was scaffolded to run entirely outside this chat sandbox (no GPU / no network here).
Everything below is real, runnable code — you provide the GPU, data, and a `pip install`.

---

## 0. Important scope notes (read first)

- **SegFormer3D** — official code: https://github.com/OSUPCVLab/SegFormer3D
  Phase 1 is a *reproduction*, so first try to get the official repo running as a sanity check
  before trusting this reimplementation. The model in `src/models/segformer3d.py` here is a
  clean-room reimplementation from the paper description (overlap patch embed, efficient
  self-attention with sequence reduction, all-MLP decoder) — useful for the boundary-module
  surgery in Phase 2, but cross-check baseline numbers against the official repo.

- **PFF-Net**'s boundary branch uses 2D Sobel operators on a 2D CNN-Transformer fusion backbone.
  `src/models/boundary_refinement.py` implements a **3D Sobel/Scharr-based boundary branch**
  adapted for anisotropic voxel spacing (BraTS voxels are ~1mm isotropic after standard
  preprocessing, but don't assume this for other datasets — check header spacing).
  This 2D→3D port is itself a piece of the novelty; document it as such, don't just call it
  "adapted."

- **SegDINO** (arXiv:2606.17972 / code at https://github.com/script-Yang/SegDINO) is a **2D**
  method (TN3K, Kvasir-SEG, ISIC, MSD-2D slices, VMD-D, ViSha). It is NOT a 3D volumetric
  method. Use it either as (a) a slice-wise 2.5D baseline you adapt yourself, or (b) the
  backbone for the optional Phase 4b 2D secondary experiment (ISIC / fundus). Don't present it
  in the same 3D Dice/HD95 table as SegFormer3D/TokenSeg without flagging the adaptation.

- **TokenSeg** (arXiv:2601.04519) has **no public code** and its primary benchmark is a private
  960-case breast DCE-MRI dataset. Cite it for the domain-shift-gap framing in Phase 3's
  motivation; do not plan to reproduce its numbers directly. If you want a real numeric
  comparison, benchmark against its *cited* baselines (nnU-Net, Swin UNETR, TransUNet — all in
  their Table II) instead, since those you can actually run yourself.

- **ODFormer** — optional Phase 4b (2D fundus secondary experiment), Swin-based, code at
  https://mias.group/ODFormer.

---

## 1. Environment setup

```bash
conda create -n seg3d python=3.10 -y
conda activate seg3d
pip install -r requirements.txt
```

GPU: any CUDA 11.8+/12.x card with >=16GB VRAM comfortably fits SegFormer3D
(4.5M params) at BraTS patch size 128^3 with batch size 2-4. This repo does not require
multi-GPU, but `train.py` supports `torchrun` DDP if you have it.

---

## 2. Data

### BraTS (Phase 1 & 2 primary)
Register and download from https://www.synapse.org/brats (or the current year's BraTS
challenge portal — the exact host changes year to year, check before assuming Synapse).
Place raw data under `data/raw/BraTS/` following the official directory layout
(one folder per case with `*_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz`).

Then:
```bash
python scripts/prepare_brats.py --raw_dir data/raw/BraTS --out_dir data/processed/BraTS
```
This resamples to 1mm isotropic, z-score normalizes per-modality, crops to nonzero region,
and writes a `.json` split manifest (train/val/test) with an 80/10/10 case-level split.

### Medical Segmentation Decathlon (Phase 3, cross-dataset generalization)
Download Task01_BrainTumour (or Task09_Spleen for a bigger domain shift) from
http://medicaldecathlon.com/. Same prep script works:
```bash
python scripts/prepare_brats.py --raw_dir data/raw/MSD_Task01 --out_dir data/processed/MSD_Task01 --dataset msd
```

---

## 3. Phase 1 — Reproduce SegFormer3D baseline

```bash
python src/train.py --config configs/phase1_baseline.yaml
```

Trains plain SegFormer3D (encoder + all-MLP decoder, no boundary branch) on BraTS.
Logs Dice / HD95 per class (WT, TC, ET) to `logs/phase1/`.

Evaluate:
```bash
python src/evaluate.py --config configs/phase1_baseline.yaml --checkpoint checkpoints/phase1/best.pt
```

Target: match paper's reported BraTS mean Dice (paper reports competitive-with-SOTA results
at 4.5M params — verify exact numbers against Table in arXiv:2404.10156 v2, since v1→v2 revised
some values).

---

## 4. Phase 2 — Boundary-aware refinement (core novelty)

```bash
python src/train.py --config configs/phase2_boundary.yaml
```

This wires `BoundaryRefinementHead` (see `src/models/boundary_refinement.py`) onto the
SegFormer3D decoder as a second, auxiliary output supervised with a boundary-aware loss
(Dice + BCE on the region mask, plus a boundary-consistency term on the Sobel-derived edge
map). Ablation configs (`configs/phase2_ablation_*.yaml`) let you turn the boundary branch,
the dual-supervision, and the fusion gate on/off independently, mirroring PFF-Net's ablation
structure (their variants A/B/C).

---

## 5. Phase 3 — Cross-dataset generalization

```bash
python src/train.py --config configs/phase3_generalization.yaml
```

Trains on BraTS, evaluates zero-shot (no fine-tuning) on MSD Task01 (brain) to quantify the
domain-shift gap that TokenSeg's abstract flags as an open problem. Optionally fine-tune with
`--finetune` to report both zero-shot and fine-tuned numbers.

---

## 6. Phase 4 — Benchmarking

`configs/phase4_baselines.yaml` trains a plain 3D U-Net baseline (`src/models/unet3d.py`,
included) on identical data splits for a fair comparison table. For SegDINO, see the note
above — it needs a separate 2D/2.5D pipeline; a starter adapter is sketched in
`scripts/segdino_2p5d_adapter_notes.md` rather than fully implemented, since it's a
different data pipeline entirely.

---

## 7. Phase 5 — Uncertainty estimation

Two paths, in `src/models/uncertainty.py`:

- **MC-Dropout** (recommended starting point — no retraining required): reuses your
  Phase 2 checkpoint directly, since the decoder already has `nn.Dropout3d` before the
  classifier. Run:
  ```bash
  python src/uncertainty_eval.py --config configs/phase5_uncertainty.yaml \
      --checkpoint checkpoints/phase2/best.pt --num_samples 20 --split test
  ```
  This reports, per BraTS region, whether predictive variance is actually higher on
  voxels the model gets wrong than on voxels it gets right (`mean_var_separation` in
  `logs/phase5_uncertainty/summary.json`) — check this number before claiming the
  uncertainty map is clinically useful; a model can have good Dice and poorly-calibrated
  uncertainty at the same time. Note: decoder dropout is currently tuned at 0.1 for
  Phase 2 segmentation accuracy, not for uncertainty spread — consider a dedicated
  higher-dropout (0.2–0.3) training run if MC-Dropout samples come out too similar.

- **Evidential Deep Learning** (`EvidentialHead` + `EvidentialLoss`): single-pass
  uncertainty via Dirichlet concentration parameters. Only valid if you retrain with
  `mutually_exclusive: true` on the raw 4-way label set — it doesn't apply to BraTS's
  default overlapping WT/TC/ET sigmoid multi-label formulation. Not wired into
  `train.py` yet; the module is ready to import if you want to build that variant out.

## 8. Ablation sweep (Phase 2)

```bash
python scripts/run_ablations.py
```

Runs, in dependency order (phase1 first, since phase2_* configs warm-start from its
checkpoint): plain SegFormer3D → boundary branch with no supervision (architecture-only
ablation) → full model (boundary branch + dual supervision, weight=0.5) → boundary
weight doubled to 1.0. Trains any config whose checkpoint doesn't already exist,
evaluates all of them on the test split, and writes
`logs/ablation_sweep/ablation_comparison.{csv,md}` — the markdown table is meant to
drop straight into a thesis/paper writeup.

Options:
```bash
python scripts/run_ablations.py --eval_only          # re-eval existing checkpoints only
python scripts/run_ablations.py --configs phase1_baseline phase2_boundary   # subset
```

To additionally isolate the gated-fusion step's contribution (does the boundary branch
help even before the final re-fusion?), re-run `src/evaluate.py` on the Phase 2
checkpoint with `eval_output: region_logits` instead of `refined_logits` in a copy of
`configs/phase2_boundary.yaml` — no retraining needed, since both heads are produced
in the same forward pass.

---

## Repo layout

```
configs/                 # YAML experiment configs
src/models/segformer3d.py        # SegFormer3D reimplementation
src/models/boundary_refinement.py # 3D boundary branch + fusion (core novelty)
src/models/unet3d.py             # baseline for Phase 4
src/data/brats_dataset.py        # dataset + augmentation pipeline
src/losses/                      # Dice, boundary-aware composite loss
src/utils/metrics.py             # Dice, HD95, sensitivity/precision
src/train.py / src/evaluate.py
scripts/prepare_brats.py         # preprocessing / manifest generation
```
## References
[1] Perera, S. et al. SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation. arXiv:2404.10156, 2024. https://arxiv.org/abs/2404.10156
[2] Enhancing medical image segmentation via complementary CNN-transformer fusion and boundary perception (PFF-Net). Frontiers in Computer Science, 2025. https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1677905/full
[3] TokenSeg: Efficient 3D Medical Image Segmentation via Hierarchical Visual Token Compression. arXiv:2601.04519. https://arxiv.org/pdf/2601.04519
[4] SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation. arXiv:2606.17972. https://arxiv.org/pdf/2606.17972
[5] ODFormer: Semantic Fundus Image Segmentation Using Transformer for Optic Nerve Head Detection. arXiv:2405.09552. https://arxiv.org/pdf/2405.09552
