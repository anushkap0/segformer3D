# SegFormer3D: Boundary-Aware Brain Tumor Segmentation and External Generalization

[![Paper](https://img.shields.io/badge/Reference-CVPRW%202024-blue)](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Perera_SegFormer3D_An_Efficient_Transformer_for_3D_Medical_Image_Segmentation_CVPRW_2024_paper.html)
[![Task](https://img.shields.io/badge/Task-3D%20Medical%20Segmentation-purple)](https://www.med.upenn.edu/cbica/brats2021/)
[![Status](https://img.shields.io/badge/Status-Research%20Evaluation-orange)](#limitations)

A research project investigating transformer-based 3D medical image segmentation, boundary-aware architectural modifications, computational efficiency, and external generalization in brain tumor segmentation.

This repository evaluates SegFormer3D-style transformer configurations against a 3D U-Net baseline using paired test-set comparisons and external dataset overlap analysis.

**Current finding:** The evaluated transformer configurations do not establish non-inferiority to 3D U-Net at the predefined mean Dice margin of 0.01. Performance differences are more pronounced in the reported zero-shot external group than in the group identified as overlapping with training data.

> This README describes experimental results supplied by the project. It is not a claim of independent reproduction of the original SegFormer3D paper.

## Contents

- [Background](#background)
- [Research Questions](#research-questions)
- [Models and Experimental Design](#models-and-experimental-design)
- [Main Test Results](#main-test-results)
- [External Generalization](#external-generalization)
- [Statistical Methods](#statistical-methods)
- [Computational Efficiency](#computational-efficiency)
- [Datasets and Label Mapping](#datasets-and-label-mapping)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [References](#references)

## Background

Three-dimensional medical image segmentation requires models to identify anatomical structures from volumetric imaging data. Brain tumor segmentation is particularly challenging because tumor subregions vary in size, shape, contrast, and appearance across MRI modalities.

Convolutional neural networks, including 3D U-Net, use local spatial operations to learn hierarchical features. Transformer-based segmentation models offer an alternative approach for representing relationships across spatial locations.

SegFormer3D, introduced by Perera, Navard, and Yilmaz, uses a hierarchical transformer encoder and an MLP-based decoder for efficient volumetric segmentation [1].

This project investigates whether transformer-based segmentation configurations can offer a useful balance between segmentation performance, computational cost, and generalization to previously unseen external cases.

The work also examines whether apparent external performance is affected by overlap between evaluation data and the model's training data.

## Research Questions

1. How do the evaluated transformer configurations compare with 3D U-Net on the held-out test set?
2. Do the transformer configurations satisfy a predefined non-inferiority margin of 0.01 mean Dice?
3. How does performance change under test-time augmentation (TTA)?
4. How do results differ between zero-shot external cases and cases identified as overlapping with training data?
5. Can computational efficiency be evaluated alongside segmentation quality?
6. How should dataset matching and overlap verification influence interpretation of external evaluation results?

## Models and Experimental Design

The supplied result files compare the following experiment labels:

| Label | Description |
|---|---|
| `full` | Full transformer configuration |
| `seg4` | Alternative transformer configuration identified by the `seg4` experiment label |
| `unet` | 3D U-Net comparison baseline |

The precise architectural differences between `full` and `seg4` should be verified against the committed configuration files before publication.

### Inference configurations

- **Plain:** Standard inference.
- **TTA:** Test-time augmentation using the transformations implemented by the evaluation pipeline.

The supplied main-test report specifies post-processing for both compared models.

A fair comparison requires consistent preprocessing, label mapping, evaluation conventions, and post-processing. The implementation details should be documented alongside the experiment configurations.

## Main Test Results

The main comparison uses 126 paired test cases. The following values are transcribed from `results_v2/stats_main.txt`.

The effect size is defined as:

\[
\Delta = \mathrm{Dice}_{\mathrm{transformer}}
-\mathrm{Dice}_{\mathrm{U\text{-}Net}}
\]

Negative differences favor 3D U-Net.

| Transformer | Inference | Cases | Mean difference | 95% CI |
|---|---|---:|---:|---|
| Full | Plain | 126 | -0.0109 | [-0.0196, -0.0008] |
| Full | TTA | 126 | -0.0091 | [-0.0184, +0.0026] |
| Seg4 | Plain | 126 | -0.0074 | [-0.0171, +0.0046] |
| Seg4 | TTA | 126 | -0.0129 | [-0.0226, -0.0023] |

### Interpretation

- All four mean differences are negative.
- The confidence intervals for full/plain and seg4/TTA exclude zero in the negative direction.
- The confidence intervals for full/TTA and seg4/plain include zero.
- The reported Wilcoxon signed-rank tests indicate negative median differences for all four comparisons.

The supplied results therefore do not show a consistent mean Dice advantage for the transformer configurations over 3D U-Net.

A confidence interval containing zero does not establish equivalence. A statistically detectable difference also does not, by itself, establish clinical importance.

### Non-inferiority analysis

The predefined non-inferiority margin is 0.01 mean Dice.

For transformer-minus-U-Net differences, the lower confidence bound must be greater than -0.01 to satisfy this criterion.

| Transformer | Inference | Lower 95% CI | Conclusion |
|---|---|---:|---|
| Full | Plain | -0.0196 | Non-inferiority not established |
| Full | TTA | -0.0184 | Non-inferiority not established |
| Seg4 | Plain | -0.0171 | Non-inferiority not established |
| Seg4 | TTA | -0.0226 | Non-inferiority not established |

None of the four comparisons meets the stated criterion.

The margin should be justified in the research protocol, and the confidence-interval procedure should be documented in the final methods section.

## External Generalization

External comparisons are reported in:

- `results_v2/stats_ext_groups.txt`
- `results_v2/stats_ext_verified_groups.txt`

These files contain groups with different definitions and sample sizes. Their results must be interpreted according to each group's actual case membership.

### Verified zero-shot group: 175 cases

| Transformer | Inference | Mean difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0149 | [-0.0209, -0.0089] |
| Full | TTA | -0.0143 | [-0.0206, -0.0082] |
| Seg4 | Plain | -0.0136 | [-0.0195, -0.0077] |
| Seg4 | TTA | -0.0139 | [-0.0202, -0.0078] |

All four confidence intervals are below zero.

On this reported zero-shot group, the transformer configurations have lower mean Dice than the 3D U-Net baseline.

### Verified trained-overlap group: 297 cases

| Transformer | Inference | Mean difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0025 | [-0.0075, +0.0031] |
| Full | TTA | -0.0036 | [-0.0083, +0.0013] |
| Seg4 | Plain | -0.0018 | [-0.0068, +0.0037] |
| Seg4 | TTA | -0.0029 | [-0.0075, +0.0020] |

All four confidence intervals include zero. The analysis does not detect a mean difference for these comparisons at the conventional 5% significance level, based on the reported intervals.

This group is not an independent zero-shot generalization benchmark if the cases were used during training. It is useful for investigating the influence of training overlap.

### Verified unmatched group: 115 cases

| Transformer | Inference | Mean difference | 95% CI |
|---|---|---:|---|
| Full | Plain | -0.0167 | [-0.0259, -0.0082] |
| Full | TTA | -0.0142 | [-0.0234, -0.0053] |
| Seg4 | Plain | -0.0149 | [-0.0239, -0.0064] |
| Seg4 | TTA | -0.0141 | [-0.0234, -0.0052] |

All four confidence intervals are below zero, indicating lower mean Dice for the transformer configurations on this reported group.

### External group definitions

The supplied reports also contain the following earlier or alternative group definitions:

| Group | Cases |
|---|---:|
| Original unmatched group | 103 |
| Original matched-but-not-trained group | 72 |
| Verified unmatched group | 115 |
| Verified held-out group | 72 |
| Verified trained-overlap group | 297 |
| Verified zero-shot group | 175 |

The 103-case and 115-case unmatched groups must not be treated as identical without checking their case manifests. Similarly, the 175-case zero-shot group and the separate verified group counts require explicit documentation of inclusion and exclusion rules.

### Interpretation

The reported mean differences are closer to zero in the trained-overlap group than in the zero-shot group. This makes dataset overlap an important factor in interpreting the external evaluation.

However, the group comparisons alone do not establish that overlap caused the observed performance gap. Differences in acquisition protocols, image quality, population, annotations, and other case characteristics may also contribute.

Case-level overlap manifests and the exact matching procedure should accompany any publication based on these results.

## Statistical Methods

The supplied analysis reports:

- Paired mean Dice differences.
- 95% confidence intervals.
- Median differences.
- Wilcoxon signed-rank tests.
- A non-inferiority assessment with a 0.01 Dice margin.
- Separate comparisons for plain inference and TTA.

### Recommended reporting standards

For reproducibility, the final methods section should specify:

1. The primary outcome and model comparison.
2. Whether Dice is averaged across cases, tumor regions, or both.
3. The confidence-interval estimation procedure and resampling unit.
4. The number of bootstrap resamples and random seed, if bootstrap intervals were used.
5. The paired statistical test and handling of ties or missing values.
6. The prespecified non-inferiority margin and its rationale.
7. Whether multiple-comparison adjustments were performed.
8. The precise definition of each external evaluation group.

The supplied text files do not independently establish all these implementation details. They should be checked against the analysis scripts before being described as finalized methods.

## Computational Efficiency

Segmentation accuracy should be assessed alongside model complexity and inference cost.

The project's earlier efficiency evaluation included measurements at an input volume of \(128^3\), including GMACs, parameter count, latency, and peak memory. Final numerical claims should be taken from the current benchmark artifacts and verified against the exact model checkpoints.

Recommended reporting metrics:

| Metric | Interpretation |
|---|---|
| Mean Dice | Volumetric overlap quality |
| HD95 | Boundary-distance error |
| GMACs | Approximate arithmetic workload |
| Parameter count | Model size |
| Inference latency | Runtime under a documented setup |
| Peak memory | Runtime memory requirement |

For a fair benchmark, document the hardware, software environment, input shape, precision, batch size, warm-up iterations, timing procedure, and whether preprocessing and post-processing are included.

A reduction in GMACs does not automatically imply lower end-to-end latency or better clinical performance.

No efficiency ranking is asserted here because the current result summaries supplied for this README do not contain the finalized efficiency table.

## Datasets and Label Mapping

The project's broader experimental work concerns multimodal MRI brain tumor segmentation and BraTS-style tumor subregions.

BraTS provides a standardized benchmark for evaluating brain tumor segmentation methods [2, 3].

Typical BraTS MRI inputs include:

- T1-weighted MRI.
- Contrast-enhanced T1-weighted MRI.
- T2-weighted MRI.
- T2-FLAIR MRI.

For the BraTS label convention used in the project's preprocessing context:

| Label | Region |
|---:|---|
| 0 | Background |
| 1 | Necrotic and non-enhancing tumor core (NCR/NET) |
| 2 | Peritumoral edema (ED) |
| 3 | Enhancing tumor (ET) |

**Important:** Label conventions can differ between raw datasets and preprocessing pipelines. Confirm the mapping in the actual loader and saved label arrays before applying this table to any specific experiment.

Dataset preparation should preserve the mapping between MRI modalities, ground-truth labels, case identifiers, and train/validation/test splits.

### External overlap verification

External evaluation should distinguish:

- Cases identified as matching training data.
- Matched cases explicitly excluded from training.
- Cases with no detected match.
- Cases whose matching status remains unresolved.

The matching method, identifiers, exclusion rules, and case counts should be reported. A missing match is not proof that two scans originate from different subjects, and a matching identifier alone is not always proof of image-level duplication.

## Reproducibility

### Inspect the current statistical summaries

Run these commands from the repository root:

```bash
cd /workspace/segformer3D

echo "===== MAIN RESULTS ====="
cat results_v2/stats_main.txt

echo "===== EXTERNAL RESULTS ====="
cat results_v2/stats_ext_groups.txt

echo "===== VERIFIED EXTERNAL RESULTS ====="
cat results_v2/stats_ext_verified_groups.txt
```

These commands only read existing results. They do not train models or rerun evaluation.

### Verify the repository state

```bash
git status --short
git log -1 --oneline
git diff --check
```

### Recommended artifacts

Preserve the following alongside the final paper or report:

- Model configuration files.
- Checkpoint filenames and checksums.
- Dataset version and split manifests.
- Case-level predictions and metrics.
- External overlap and matching reports.
- Label-mapping verification figures.
- Statistical analysis scripts and arguments.
- Environment and dependency versions.
- Final efficiency tables and plots.
- Final main and external results tables.

Record the Git commit, checkpoint, configuration, dataset split, random seed, and evaluation settings for every reported result.

Do not retrain a model merely to update the README when the required predictions and evaluation outputs already exist.

## Limitations

1. **Non-inferiority:** The reported main-test comparisons do not establish non-inferiority at the 0.01 Dice margin.
2. **Zero-shot generalization:** All reported verified zero-shot confidence intervals favor 3D U-Net in mean Dice.
3. **Dataset overlap:** Results from cases identified as overlapping with training data should not be interpreted as independent zero-shot generalization.
4. **Group membership:** The external reports contain differing unmatched-group sizes. Case manifests must resolve the differences.
5. **Metric coverage:** The supplied statistical summaries do not include final HD95 results.
6. **Multiple comparisons:** The available summaries do not establish whether multiplicity correction was applied.
7. **Clinical interpretation:** Statistical differences in Dice do not automatically establish clinical significance.
8. **Reproducibility:** Final publication requires the exact analysis code, configurations, dataset splits, and evaluation environment.

## Conclusion

The current results show that the evaluated transformer configurations have lower mean Dice than 3D U-Net on the main test and reported zero-shot external groups. Non-inferiority at a margin of 0.01 has not been established.

For the verified trained-overlap group, the reported mean differences are smaller and the confidence intervals include zero. This reinforces the need to distinguish training-overlap analyses from genuine zero-shot evaluation.

The central contribution of this investigation is a structured comparison of segmentation quality, statistical uncertainty, external dataset overlap, and computational efficiency. Final conclusions should be based on verified case membership, complete boundary metrics, and a reproducible statistical protocol.

## References

1. **Perera, S., Navard, P., & Yilmaz, A. (2024).** SegFormer3D: An Efficient Transformer for 3D Medical Image Segmentation. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, 4981–4988.  
   - Paper: https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Perera_SegFormer3D_An_Efficient_Transformer_for_3D_Medical_Image_Segmentation_CVPRW_2024_paper.html  
   - Official project page: https://osupcvlab.github.io/SegFormer3D/  
   - Official code: https://github.com/OSUPCVLab/SegFormer3D  
   - DOI: https://doi.org/10.1109/CVPRW63382.2024.00503

2. **Baid, U., et al. (2021).** The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification.  
   - BraTS 2021 challenge information: https://www.med.upenn.edu/cbica/brats2021/  
   - Dataset and challenge details should be cited using the exact publication corresponding to the dataset release used in the experiments.

3. **Menze, B. H., et al. (2015).** The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024.  
   - DOI: https://doi.org/10.1109/TMI.2014.2377694

4. **Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016).** 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 424–432.  
   - DOI: https://doi.org/10.1007/978-3-319-46723-8_49

5. **Reinke, A., et al. (2021).** Understanding metric-related pitfalls in image analysis validation. *Nature Methods*, 18, 1207–1212.  
   - DOI: https://doi.org/10.1038/s41592-021-01260-5

6. **Müller, D., Soto-Rey, I., & Kramer, F. (2022).** Towards a guideline for evaluation metrics in medical image segmentation. *BMC Research Notes*, 15, 210.  
   - Article: https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/

7. **Dice, L. R. (1945).** Measures of the Amount of Ecologic Association Between Species. *Ecology*, 26(3), 297–302.  
   - DOI: https://doi.org/10.2307/1932409

8. **Wilcoxon, F. (1945).** Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83.  
   - DOI: https://doi.org/10.2307/3001968

## Citation

If you use this repository in a report, thesis, or publication, cite the original SegFormer3D paper and the exact dataset and evaluation references relevant to your experiments.

The repository's own experimental results should be cited using the project version, Git commit, and associated result artifacts once these have been finalized.

