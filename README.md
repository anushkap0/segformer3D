# Boundary-Refined SegFormer3D for Brain Tumour Segmentation

A controlled evaluation of a lightweight 3D transformer (SegFormer3D, Perera et al., CVPR-W 2024, [arXiv:2404.10156](https://arxiv.org/abs/2404.10156)) with and without a boundary-refinement branch, compared with a 3D U-Net, on BraTS 2021 and on an external set (Medical Segmentation Decathlon Task01) whose overlap with BraTS 2021 was audited case by case.

This repository contains the training and evaluation code, all configurations, the analysis scripts, and the small result files (per-case metrics, tables and figures) behind the manuscript. Data, trained checkpoints and training logs are **not** included.

## Main findings

All numbers come from the frozen analysis in `results/` (paired bootstrap 95% CIs over cases, 10,000 resamples). Boundary models are the mean of two Phase 2 seeds.

| Model | Params (M) | GMACs | Test mean Dice (126 cases) |
|---|---|---|---|
| Baseline (SegFormer3D) | 4.53 | 15.5 | 0.8951 |
| Continued (extra-training control) | 4.53 | 15.5 | 0.8969 |
| Boundary, no loss | 5.36 | 43.0 | 0.8993 |
| Boundary + loss | 5.36 | 43.0 | 0.9011 |
| 3D U-Net | 22.58 | 481.2 | 0.9106 |

- The full boundary model beats its extra-training control (Continued) by +0.0042 mean Dice (95% CI +0.0020 to +0.0075).
- The boundary branch alone (+0.0024, CI −0.0008 to +0.0044) and the boundary loss alone (+0.0018, CI −0.0014 to +0.0062) have no detectable effect on the mean. Per case, the branch gives a small consistent shift (median +0.0028; surface Dice +0.0048, CI +0.0010 to +0.0077); the loss gives none.
- The 3D U-Net has the highest accuracy at 11–31× the GMACs. On the 187 external cases unseen in training it leads every transformer variant by +0.011 to +0.015 Dice, with every CI excluding zero. The lead is unchanged after removing 12 uncertain matches and the matches from the BraTS 2021 validation split.
- A reproduction of SegFormer3D on BraTS 2017 with the authors' preprocessing, split and recipe gives a mean Dice of 0.827–0.831 against the published 0.821.

Full tables are in `results/final/` and in the manuscript.

## Repository layout

```
configs/      YAML experiment configs
              v2_*.yaml            final models (Baseline, Continued, Boundary no loss / with loss, seeds 1 and 2)
              ext_*.yaml           external-set evaluation of the same models
              phase4_baselines.yaml / ext_unet.yaml   3D U-Net
              repro2017*.yaml      BraTS 2017 reproduction
              phase1_*, phase2_*, phase3_*, phase5_*  earlier development configs
src/          train.py, evaluate.py, train_repro.py, models/, data/, losses/, utils/
scripts/      preprocessing, evaluation drivers, overlap audit, statistics, figures
results/      per-case JSONs (test, external_flip), tables (results/final), figures
logs/         efficiency.json (params, GMACs, latency, memory), logs/ext/overlap.json
```

## Setup

```bash
conda create -n seg3d python=3.10 -y && conda activate seg3d
pip install -r requirements.txt        # also needs monai, nibabel, scipy, matplotlib, pyyaml
```

Experiments were run on a single 80 GB NVIDIA GPU (DGX node), PyTorch 2.4 (NVIDIA container 24.06), with mixed precision.

The `v2_*` models use the official SegFormer3D code. Clone <https://github.com/OSUPCVLab/SegFormer3D> and set `model.official_repo_dir` in the config (the configs default to `/workspace/official_src_clone`).

## Data

**BraTS 2021** (1,251 cases). Download from the BraTS 2021 challenge, place it under `data/raw/BraTS2021_FULL/`, then:

```bash
python scripts/prepare_brats.py --raw_dir data/raw/BraTS2021_FULL --out_dir data/processed/BraTS2021_FULL --legacy_names
```

This resamples to 1 mm, z-scores each modality over the brain, crops to the non-zero region and writes `split_manifest.json` (case-level 80/10/10 split, seed 42: 1,000 / 125 / 126). Use `--legacy_names` for the `*_t1ce.nii.gz` file naming.

**External set** (MSD Task01 BrainTumour, 484 cases). Download from <http://medicaldecathlon.com/>, then:

```bash
python scripts/prepare_msd.py --raw_dir /path/to/Task01_BrainTumour --out_dir data/processed/BraTS2017ext
```

The MSD copy is rotated 180° in-plane relative to BraTS 2021. `scripts/flip_ext.py` writes a corrected copy (`BraTS2017ext_flip`), which is the one used for all reported external results. Labels are remapped from MSD to the internal convention (1→2, 2→1, 3→3); `scripts/fig_label_check.py` draws the check.

## Training

```bash
# Phase 1: Baseline from scratch (200 epochs)
python src/train.py --config configs/v2_phase1_baseline.yaml

# Phase 2: initialised from the Phase 1 checkpoint (200 epochs each)
python src/train.py --config configs/v2_phase1_continued.yaml      # Continued
python src/train.py --config configs/v2_abl_noloss.yaml            # Boundary, no loss (seed 1)
python src/train.py --config configs/v2_abl_noloss_s2.yaml         # Boundary, no loss (seed 2)
python src/train.py --config configs/v2_phase2_boundary.yaml       # Boundary + loss (seed 1)
python src/train.py --config configs/v2_phase2_boundary_s2.yaml    # Boundary + loss (seed 2)

# 3D U-Net from scratch
python src/train.py --config configs/phase4_baselines.yaml
```

Checkpoints go to `checkpoints/<name>/best.pt` (selected on the validation split). Seed 1 runs were not seeded; seed 2 sets `train.seed: 2`. Both seeds share one Phase 1 checkpoint, so seed 2 varies only Phase 2.

## Evaluation

```bash
# test split, plain inference (no test-time augmentation)
python src/evaluate.py --config configs/v2_phase1_baseline.yaml \
    --checkpoint checkpoints/v2_phase1/best.pt --split test --output_json results/test/baseline.json

# all models, test split and external set
bash scripts/eval_all.sh
bash scripts/eval_ext_flip.sh        # external set, orientation corrected
```

Inference uses sliding windows (128³, 50% overlap, Gaussian weighting). The JSONs hold per-case Dice, HD95, sensitivity, precision and surface Dice for WT, TC and ET.

## Analysis

```bash
python scripts/check_overlap.py              # fingerprint match of external cases to BraTS 2021 -> logs/ext/overlap.json
python scripts/verify_matches_flip.py        # confirm matches by FLAIR correlation after alignment
python scripts/ext_groups_v2.py results/external_flip   # trained / matched-not-trained / unmatched groups
python scripts/final_analysis.py             # paired CIs, groups, memorisation, HD95 with the BraTS empty-label rule
python scripts/sensitivity_failed12.py       # zero-shot result without 12 uncertain matches
python scripts/sensitivity_valsplit.py       # ... and without matches in the BraTS 2021 validation split
python scripts/extra_analysis.py             # surface Dice, median differences, Wilcoxon tests, distributions
python scripts/benchmark_models.py v2_phase1_baseline v2_phase2_boundary phase4_baselines --time   # params, GMACs, latency, memory
python scripts/fig_dice_vs_gmacs.py          # Dice vs GMACs figure
```

HD95 follows the BraTS 2021 rule: both masks empty gives Dice 1 and HD95 0; exactly one empty gives Dice 0 and HD95 373.13 (the diagonal of a 240×240×155 volume). A version with those cases excluded is reported as a sensitivity check.

## BraTS 2017 reproduction

A check that the model and recipe reproduce the published SegFormer3D result. It uses the authors' preprocessing (RAS orientation, per-slice min-max scaling, fixed 128³ crop), their 411/73 split and their recipe (800 epochs, lr 1e-4, weight decay 0.01, batch size 2, cosine schedule with restarts).

```bash
# 1. run the authors' preprocessing (from https://github.com/OSUPCVLab/SegFormer3D)
# 2. convert its output to this repo's format and use the authors' split
python scripts/convert_authors_2017.py
python scripts/check_repro.py
# 3. train and compare
python src/train_repro.py --config configs/repro2017.yaml
python src/evaluate.py --config configs/repro2017.yaml --checkpoint checkpoints/repro2017/best.pt --split test --output_json results/repro2017_best.json
python scripts/repro_compare.py results/repro2017_best.json
```

`src/train_repro.py` is `train.py` with a warm-restart cosine schedule added. The run differs from the authors' in input size (whole 128³ volumes, not 96³ crops), augmentation, the loss implementation and an approximated restart schedule; it validates the model and recipe, not the BraTS 2021 preprocessing.

## Limitations

- Seed 1 runs were not seeded; Baseline, Continued and the U-Net are single runs; seed spread is as large as the boundary effects.
- Confidence intervals resample cases only and are not corrected for multiple comparisons.
- Timings come from a shared GPU.
- Some external cases could be related scans of patients in BraTS 2021; excluding the uncertain ones did not change the conclusions.
- No qualitative overlays of predictions are included.

## Background and references

The boundary branch is inspired by the boundary branch of PFF-Net (Frontiers in Computer Science, 2025) and is adapted from 2D to 3D volumes. `scripts/segdino_2p5d_adapter_notes.md` holds notes on a 2.5D SegDINO adapter, which was not used in the final experiments.

- Perera S, Navard P, Yilmaz A. SegFormer3D: an efficient transformer for 3D medical image segmentation. CVPR Workshops 2024. arXiv:2404.10156.
- Baid U, et al. The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv:2107.02314.
- Antonelli M, et al. The Medical Segmentation Decathlon. Nat Commun 2022;13:4128.
- Hers B, et al. SegFormer3D: improving the robustness of deep learning model-based image segmentation in ultrasound volumes of the pediatric hip. Ultrasound Med Biol 2025;51:751–8.
