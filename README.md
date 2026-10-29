# SegFormer3D: Boundary-Aware 3D Medical Image Segmentation

Research repository for evaluating SegFormer3D-style transformer models against a 3D U-Net baseline for volumetric brain tumor segmentation, with a focus on computational efficiency, boundary-aware learning, and external generalization.

> **Research status:** Experimental evaluation and statistical analysis. The reported results do not establish non-inferiority of the transformer models to 3D U-Net at a Dice margin of 0.01.

## Table of Contents

- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Experimental Evaluation](#experimental-evaluation)
- [Main Test Results](#main-test-results)
- [External Generalization](#external-generalization)
- [Statistical Analysis](#statistical-analysis)
- [Inference Settings](#inference-settings)
- [Computational Efficiency](#computational-efficiency)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)

## Overview

Three-dimensional medical image segmentation is computationally demanding, particularly when models must preserve spatial detail and accurately delineate small or irregular anatomical structures.

This project evaluates transformer-based volumetric segmentation against a 3D U-Net baseline. The experiments investigate:

1. Segmentation performance on held-out test cases.
2. The effect of boundary-aware model configurations.
3. Generalization to external cases.
4. Potential overlap between external evaluation cases and training data.
5. Statistical uncertainty in model comparisons.
6. The effect of inference settings, including test-time augmentation (TTA).

The goal is to provide a reproducible, statistically informed comparison rather than relying on Dice scores alone.

## Research Objectives

- Evaluate transformer-based 3D segmentation models against a conventional 3D U-Net.
- Investigate whether boundary-oriented architectural changes improve segmentation quality.
- Assess generalization on external cases, separating previously encountered cases from zero-shot cases.
- Compare plain inference and test-time augmentation.
- Quantify paired performance differences and uncertainty using confidence intervals.
- Examine whether computational savings justify any reduction in segmentation performance.

## Experimental Evaluation

### Main test set

The main comparison uses 126 test cases.

The supplied statistical reports compare two transformer configurations against the 3D U-Net baseline:

- `full`: the full transformer configuration.
- `seg4`: an alternative transformer configuration using four segmentation stages, as identified by the experiment labels.
- `unet`: the 3D U-Net reference model.

The exact architectural differences should be confirmed from the corresponding configuration files before interpreting `full` and `seg4` as specific architectural ablations.

### External evaluation

External results are divided into groups according to dataset matching and training overlap.

| Evaluation group | Cases | Interpretation |
|---|---:|---|
| Unmatched external cases | 103 | Cases without a detected dataset match |
| Matched but not trained | 72 | Matched cases excluded from training |
| Verified zero-shot evaluation | 175 | Combined zero-shot group in the verified report |
| Verified unmatched group | 115 | Unmatched group in the verified report |
| Verified held-out group | 72 | Matched cases excluded from training |
| Verified trained-overlap group | 297 | Cases identified as overlapping with training data |

The groups above come from different stages or definitions of external verification. They must not be added together as though they were mutually exclusive groups from one partition.

The final manuscript should document the matching procedure, dataset identifiers, exclusions, and exact case-level membership for each group.

## Main Test Results

The following table summarizes the supplied paired comparison statistics for mean Dice.

**Metric:** Transformer Dice minus 3D U-Net Dice. Negative values favor 3D U-Net.

| Model | Inference | N | Mean difference | 95% CI |
|---|---|---:|---:|---|
| Full | Plain | 126 | -0.0109 | [-0.0196, -0.0008] |
| Full | TTA | 126 | -0.0091 | [-0.0184, +0.0026] |
| Seg4 | Plain | 126 | -0.0074 | [-0.0171, +0.0046] |
| Seg4 | TTA | 126 | -0.0129 | [-0.0226, -0.0023] |

### Interpretation

- The full model has a negative mean Dice difference under both inference settings.
- For full-model TTA and seg4 plain inference, the reported 95% confidence intervals include zero.
- For full-model plain inference and seg4 TTA, the reported confidence intervals exclude zero in the negative direction.
- The reported Wilcoxon signed-rank tests indicate negative median differences for all four comparisons.

A confidence interval that includes zero does not establish equivalence. Likewise, a confidence interval excluding zero does not by itself establish clinical significance.

### Non-inferiority analysis

The predefined non-inferiority margin is 0.01 Dice.

For non-inferiority, the lower confidence bound for the transformer-minus-U-Net difference must be greater than -0.01.

None of the four main-test comparisons satisfies this criterion:

| Model | Inference | Lower 95% CI bound | Non-inferiority |
|---|---|---:|---|
| Full | Plain | -0.0196 | Not established |
| Full | TTA | -0.0184 | Not established |
| Seg4 | Plain | -0.0171 | Not established |
| Seg4 | TTA | -0.0226 | Not established |

These results do not demonstrate that the transformer models are non-inferior to 3D U-Net under the selected margin.

## External Generalization

### Verified zero-shot group: 175 cases

| Model | Inference | Mean Dice difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0149 | [-0.0209, -0.0089] |
| Full | TTA | -0.0143 | [-0.0206, -0.0082] |
| Seg4 | Plain | -0.0136 | [-0.0195, -0.0077] |
| Seg4 | TTA | -0.0139 | [-0.0202, -0.0078] |

All four reported confidence intervals are below zero. On this zero-shot group, the transformer models have lower mean Dice than 3D U-Net.

This supports the conclusion that the transformer configurations did not match the U-Net baseline on the evaluated zero-shot cases.

### Verified trained-overlap group: 297 cases

| Model | Inference | Mean Dice difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0025 | [-0.0075, +0.0031] |
| Full | TTA | -0.0036 | [-0.0083, +0.0013] |
| Seg4 | Plain | -0.0018 | [-0.0068, +0.0037] |
| Seg4 | TTA | -0.0029 | [-0.0075, +0.0020] |

All four confidence intervals include zero. The results do not show a statistically detectable mean difference for this group under the reported analysis.

This group should not be interpreted as an independent external generalization benchmark if the evaluated cases were used for training. It is useful for investigating the effect of dataset overlap and potential memorization.

### Verified unmatched group: 115 cases

| Model | Inference | Mean Dice difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0167 | [-0.0259, -0.0082] |
| Full | TTA | -0.0142 | [-0.0234, -0.0053] |
| Seg4 | Plain | -0.0149 | [-0.0239, -0.0064] |
| Seg4 | TTA | -0.0141 | [-0.0234, -0.0052] |

The confidence intervals are below zero for all four comparisons, indicating lower mean Dice for the transformer configurations in this group.

### Interpretation of external results

The difference between the zero-shot and trained-overlap groups suggests that overlap status is important when interpreting external evaluation.

However, these comparisons alone cannot establish that overlap caused the observed performance differences. Differences in case characteristics, acquisition protocols, data quality, and other dataset properties may also contribute.

## Statistical Analysis

The supplied result files report the following analyses:

- Paired mean Dice differences between each transformer model and 3D U-Net.
- 95% confidence intervals for the mean difference.
- Median differences.
- Wilcoxon signed-rank tests.
- Non-inferiority assessment using a margin of 0.01 Dice.
- Comparisons under plain inference and test-time augmentation.

### Statistical interpretation

The primary comparison should be defined before final reporting, including:

1. The primary model and inference configuration.
2. The primary metric and aggregation method.
3. The paired statistical procedure.
4. The confidence interval method.
5. The non-inferiority margin and its justification.
6. The treatment of multiple comparisons.

Because multiple models, inference settings, and external groups are evaluated, the report should state whether multiplicity correction was used. The supplied text files do not establish that correction was applied.

Confidence intervals and p-values should be interpreted alongside effect sizes and case-level variability.

## Inference Settings

Two inference settings are included:

- **Plain inference:** Standard model inference without test-time augmentation.
- **TTA:** Test-time augmentation, which combines predictions obtained from transformed inputs according to the implemented evaluation procedure.

Both settings are reported for the main test and verified external groups.

For a fair comparison, the U-Net and transformer predictions must use consistent preprocessing, label mapping, post-processing, and evaluation conventions. The supplied main-test report specifies that both models are post-processed.

The exact transformations, inverse transformations, prediction aggregation, and post-processing operations should be documented from the evaluation code.

## Computational Efficiency

Computational efficiency is an important part of the research question. The project has previously evaluated segmentation quality, multiply-accumulate operations (GMACs), parameter count, inference latency, and memory usage.

The repository contains or has referenced an efficiency summary and a Dice-versus-GMACs plot. Their latest values should be checked before publishing a numerical efficiency comparison.

Recommended reporting metrics:

| Metric | Purpose |
|---|---|
| Mean Dice | Segmentation overlap |
| HD95 | Boundary-distance quality |
| GMACs | Approximate computational workload |
| Parameter count | Model size |
| Inference latency | Runtime under a documented setup |
| Peak memory | Runtime memory requirements |

Efficiency measurements should specify input dimensions, hardware, precision, batch size, warm-up iterations, timing procedure, and whether preprocessing and post-processing are included.

A lower GMAC count does not automatically imply lower end-to-end latency or better clinical utility.

## Reproducibility

Run the following commands from the repository root to inspect the currently saved statistical summaries:

```bash
cd /workspace/segformer3D

echo "===== MAIN RESULTS ====="
cat results_v2/stats_main.txt

echo "===== EXTERNAL RESULTS ====="
cat results_v2/stats_ext_groups.txt

echo "===== VERIFIED EXTERNAL RESULTS ====="
cat results_v2/stats_ext_verified_groups.txt
```

These commands read existing result files; they do not train models or rerun evaluation.

### Recommended artifacts to preserve

- Experiment configuration files.
- Model checkpoints and checkpoint identifiers.
- Dataset split manifests.
- Case-level predictions and metrics.
- External overlap and matching reports.
- Label-mapping verification artifacts.
- Statistical analysis scripts.
- Environment and dependency versions.
- Efficiency measurements.
- Final tables and figures.

For reproducibility, record the Git commit, configuration, checkpoint, dataset version, seed, and evaluation command associated with each reported result.

### Reproducing the statistical analysis

Use the project's actual analysis script and documented arguments. The exact command is intentionally not specified here because the supplied result output does not identify the script or its command-line interface.

Do not retrain models simply to reproduce a table if the required predictions and saved outputs already exist.

## Limitations

1. **Non-inferiority not established:** The main-test confidence intervals do not satisfy the predefined 0.01 Dice non-inferiority criterion.
2. **Zero-shot performance gap:** All reported zero-shot mean-difference confidence intervals favor 3D U-Net.
3. **Potential dataset overlap:** External results must be interpreted in light of matching and training exposure.
4. **Group-definition differences:** The original and verified external reports use different group sizes. The exact case membership needs to be documented.
5. **Metric coverage:** The supplied summaries focus on mean Dice. They do not provide enough information to report final HD95 results.
6. **Multiple comparisons:** The available summaries do not establish whether adjustments for multiple testing were performed.
7. **Generalizability:** Results from these evaluated cases cannot establish performance on every institution, scanner, or clinical population.
8. **Efficiency reporting:** Numerical efficiency claims should be taken from the finalized benchmark artifacts, not inferred from segmentation statistics.

## Conclusion

The supplied results show that the evaluated transformer configurations have lower mean Dice than 3D U-Net on the main test and zero-shot external groups. The main-test comparisons do not establish non-inferiority at a 0.01 Dice margin.

In the verified trained-overlap group, the reported mean differences are smaller and the confidence intervals include zero. This highlights the importance of separating genuine zero-shot evaluation from cases that overlap with training data.

The appropriate research conclusion is therefore not that the transformer models outperform U-Net, but that their segmentation performance, external generalization, computational requirements, and sensitivity to dataset overlap require joint evaluation.

Further reporting should prioritize transparent case-level overlap verification, finalized HD95 results, reproducible efficiency measurements, and a clearly specified statistical protocol.
