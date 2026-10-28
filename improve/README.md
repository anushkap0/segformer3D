# Inference-time improvement pipeline (26–30 Oct)

Goal: improve BR-SegFormer3D with no retraining, and test fairly whether a small
SegFormer3D ensemble matches or beats the 3D U-Net. Three levers, applied **identically
to every model, including the U-Net**:

1. **Flip test-time augmentation**: average of the 8 flip combinations of the three axes.
2. **Seed ensembles**: probability average of checkpoints you already have
   (`full` = 2 BR-SegFormer3D seeds, 86 GMACs; `seg4` = + 2 no-loss seeds, 172 GMACs;
   both far below the U-Net's 481 GMACs).
3. **Validation-tuned post-processing**: per-region thresholds, removal of small WT
   components, the BraTS rule that relabels a tiny predicted ET as tumour core (this
   removes the false-positive ET that the BraTS HD95 rule penalises with 373.13 mm), and
   optionally ET within TC within WT.

Everything after inference runs on saved probabilities, so tuning and statistics are fast
and need no GPU.

## Files

| File | Step | Needs GPU |
|---|---|---|
| `infer_probs.py` | sliding-window inference, saves plain and 8-flip TTA probabilities (uint8 npz, resumable) | yes |
| `tune_val.py` | chooses post-processing on the **validation split only** | no |
| `evaluate_pp.py` | per-case Dice, HD95 (BraTS rule, medpy definition), surface Dice 1 mm, false-positive and missed-region counts | no |
| `stats.py` | paired bootstrap CI, median, Wilcoxon, win rate, non-inferiority and superiority tests | no |
| `postprocess.py`, `common.py` | shared code | no |
| `run_pipeline.sh` | runs all steps; edit the CONFIG block | |
| `PROTOCOL.md` | analysis plan to fill in and commit **before** testing | |

## Before running

* Put the `improve/` folder at the repository root on the server.
* In `run_pipeline.sh`, set the config and checkpoint paths of the seven runs, the
  external data directory and the two case lists (`groups/external_all_484.txt`,
  `groups/zero_shot_187.txt`, from `results/external_groups_verified.json`).
* `infer_probs.py` builds models with `src.train.build_model(cfg)` and loads
  `model_state_dict`, as `src/evaluate.py` does. If the server builds the official
  SegFormer3D through `src/models/official_wrappers.py`, change only `load_model()`.
* Disk: about 2–5 MB per case per model (uint8, compressed), roughly 20 GB in total.

## Schedule

| Day | Step | Command |
|---|---|---|
| 27 Oct | Inference of all 7 runs on val (125), test (126) and external (484) | `run_pipeline.sh infer` (GPU, resumable; run models in parallel if several GPUs are free) |
| 27 Oct | **Sanity check**: plain pass + threshold 0.5 must reproduce the frozen main table to 4 decimals in Dice | `run_pipeline.sh sanity` |
| 28 Oct | Tune on validation; fill in and commit `PROTOCOL.md` and `pp/*.json` | `run_pipeline.sh tune` |
| 28 Oct | Evaluate test and external **once** | `run_pipeline.sh test` |
| 28 Oct | Statistics | `run_pipeline.sh stats` |
| 29 Oct | Update tables and text; new snapshot and hash | |
| 30 Oct | Submit | |

Inference cost: TTA needs 8 passes per case. At the measured window latencies (8.8 ms for
BR-SegFormer3D, 22.5 ms for the U-Net) inference itself is fast; data loading and the
sliding-window overhead dominate. If time is short, use `--no_tta` for the external set first.

## Rules that keep the result publishable

* **Same treatment for the U-Net.** It gets TTA and its own validation-tuned
  post-processing. A comparison where only SegFormer3D is post-processed will be rejected.
* **No test-set tuning.** `tune_val.py` refuses any split except validation. Run the test
  step once, after committing the protocol.
* **Report everything** listed in `PROTOCOL.md`, including configurations that do not
  improve, and label these analyses as post hoc relative to the frozen snapshot.
* **What can be claimed:**
  * Superiority over the U-Net only if the lower CI bound of (candidate − U-Net) is above 0
    on the test split.
  * Non-inferiority if it is above −0.01. Combined with roughly 3× fewer GMACs, that is a
    publishable efficiency claim.
  * Otherwise, report that the U-Net remains more accurate.

## Issues found in the public repository (check whether the server code has them)

1. `BoundaryBranch3D.forward` builds the Sobel edge prior from `softmax(region_logits)`,
   but the three regions are overlapping sigmoid channels. `sigmoid` is the consistent
   choice. Changing it needs retraining, so leave it for the revision and describe the
   implemented version in the paper.
2. `train.py::validate` evaluates a **random crop** per validation case (the dataset's
   `_random_crop` is used with `train=False`), so checkpoint selection is noisy. If the
   server code still does this, full-volume sliding-window validation is the cheapest
   retraining-free fix for future runs. Do not change the frozen checkpoints now.
3. `hd95_score` returns NaN for empty masks; the frozen results used the BraTS rule, as
   this pipeline does.
