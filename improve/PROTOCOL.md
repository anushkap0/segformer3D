# Pre-specified analysis plan for the inference-time improvements

Fill in the date and commit this file together with `pp/*.json` **before** running
`run_pipeline.sh test`. Do not change anything below after the test split has been
evaluated; report every configuration listed here, whatever the result.

Date fixed: 2026-10-28  Commit: the git commit containing this file and pp/*.json

## Configurations (all models receive identical treatment)

| Name | Models | Inference | Post-processing |
|---|---|---|---|
| unet | 3D U-Net | plain / 8-flip TTA | tuned on validation for the U-Net |
| full | BR-SegFormer3D seeds 1 + 2 (probability average) | plain / 8-flip TTA | tuned on validation for this ensemble |
| seg4 | BR-SegFormer3D s1, s2 + Boundary, no loss s1, s2 | plain / 8-flip TTA | tuned on validation for this ensemble |

Compute per 128³ window: unet 481.2 GMACs; full 2 x 43.0 = 86.0; seg4 4 x 43.0 = 172.0
(TTA multiplies all three by 8).

## Tuning

Post-processing parameters (per-region thresholds, minimum WT component size, minimum ET
volume, ET within TC within WT) are chosen on the 125 validation cases only, by
`tune_val.py`, separately for each configuration and inference mode. The test split and
the external set are not used for any choice.

## Primary comparison

Mean Dice on the 126 test cases, candidate minus U-Net, both with TTA and their own tuned
post-processing; paired bootstrap 95% CI (10,000 resamples of cases, seed 0).

* Non-inferiority margin: **0.01** mean Dice. The candidate is non-inferior if the lower
  bound of the CI is above -0.01.
* Superiority: lower bound above 0.

## Secondary comparisons

The same contrast on the 175 zero-shot external cases (72 matched but not trained + 103 unmatched); per-region Dice; HD95 (BraTS rule)
with the number of penalised cases; surface Dice at 1 mm; the plain-inference versions;
each model with versus without TTA and post-processing (to show what each step adds).

## Reporting

All of the above are reported in the paper or its supplement, including configurations
that do not improve. These analyses are post hoc relative to the frozen snapshot
(sha256 4bc61574...3f282) and are labelled as such.

## Notes fixed before testing (28 Oct)

* The 0.01 Dice margin is the smallest difference we regard as practically relevant. It was
  fixed after the frozen single-model test gap to the U-Net (0.0108) was already known, so it
  was not chosen blind; the conclusion must not depend on it, and the plain CI is reported too.
* Post-processing parameters are in pp/*.json (tuned on the 125 validation cases only).
  On validation the tuned post-processing added only 0.002 to 0.003 mean Dice, and the
  U-Net remained about 0.02 ahead of both ensembles.
* HD95 and surface Dice come from improve/common.py for all models and are not mixed with
  the earlier frozen-table values (different empty-mask rule and surface definition).
* The two BR-SegFormer3D seeds share one Phase 1 checkpoint, so ensemble diversity is limited;
  the U-Net is a single run.
