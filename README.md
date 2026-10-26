# SegFormer3D: 3D Brain Tumor Segmentation

A research project exploring transformer-based 3D medical image segmentation, boundary-aware training objectives, computational efficiency, and generalization to external datasets.

## Overview

This project investigates 3D brain tumor segmentation using a SegFormer3D-style architecture. The experiments compare baseline segmentation, boundary-aware training, ablations, continued training, and a 3D U-Net reference model.

The research focuses on three questions:

- Does boundary-aware training improve tumor segmentation?
- How do segmentation performance and computational cost compare across models?
- How well do trained models generalize to external data, and how can dataset overlap affect the interpretation of results?

**Project status:** Experimental implementation and evaluation analysis completed for the reported runs. Results should be interpreted in the context of the dataset, split, and evaluation protocol used.

## Research Objectives

1. Evaluate 3D brain tumor segmentation using multimodal MRI.
2. Compare baseline and boundary-aware training experiments.
3. Investigate the effect of removing the boundary-related loss.
4. Compare segmentation quality with computational efficiency.
5. Analyze external-dataset performance and possible overlap with training data.
6. Report limitations and controlled negative results transparently.

## Key Experiments

| Experiment | Purpose |
|---|---|
| Baseline | Reference SegFormer3D-style model |
| Continued training | Evaluate additional training from an existing baseline checkpoint |
| Boundary-aware, Seed 1 | Evaluate the boundary-aware architecture and training setup |
| Boundary-aware, Seed 2 | Examine variation across a second seed |
| No-boundary-loss, Seed 1 | Ablation to investigate the contribution of boundary loss |
| No-boundary-loss, Seed 2 | Replicate the ablation with a second seed |
| 3D U-Net | Convolutional reference model for comparison |

The precise architecture, configuration, dataset split, and checkpoint associated with each result should be taken from the corresponding configuration and experiment log.

## Reported Test Results

The following values are from the recorded test results for the SegFormer3D experiments.

| Model / experiment | Mean Dice | WT Dice | TC Dice | ET Dice |
|---|---:|---:|---:|---:|
| Baseline | 0.8951 | 0.9347 | 0.9052 | 0.8454 |
| Continued training | 0.8969 | 0.9362 | 0.9074 | 0.8472 |
| Boundary-aware, Seed 1 | 0.8999 | 0.9394 | 0.9107 | 0.8494 |
| Boundary-aware, Seed 2 | 0.9023 | 0.9397 | 0.9091 | 0.8582 |
| No-boundary-loss, Seed 1 | 0.9000 | 0.9390 | 0.9097 | 0.8512 |

**Metric definitions**

- **Mean Dice:** Average Dice score across the evaluated tumor regions, according to the evaluation script.
- **WT:** Whole Tumor.
- **TC:** Tumor Core.
- **ET:** Enhancing Tumor.

Dice scores range from 0 to 1, with higher values indicating greater overlap between predicted and reference segmentation masks. Confirm the aggregation method and region definitions in the evaluation code before making comparisons with external publications.

These are recorded experiment results, not a guarantee of performance on other datasets or hardware.

## Computational Efficiency

The project includes an efficiency analysis using 128 × 128 × 128 input volumes.

Recorded measurements include:

- Multiply-accumulate operations (GMACs).
- Number of model parameters.
- Inference latency.
- Peak memory usage.
- Dice score versus computational cost.

The comparison includes transformer-based experiments and a 3D U-Net reference model. Hardware, precision, batch size, warm-up, synchronization, and measurement procedure can affect efficiency results; consult the experiment scripts and generated reports for the exact setup.

Relevant generated artifacts include:

- `outputs/efficiency_summary.csv`
- `outputs/dice_vs_gmacs.png`

Use the CSV as the source of numerical efficiency measurements and the plot for visual comparison.

## External Generalization and Dataset Overlap

External evaluation is included to investigate generalization beyond the primary test set.

The recorded overlap analysis identifies these groups:

| Group | Cases |
|---|---:|
| Cases identified as overlapping with training data | 309 |
| Matched cases not identified as training cases | 72 |
| Unmatched cases | 103 |
| Total cases analyzed | 484 |

These groups reflect the recorded matching analysis and should not automatically be interpreted as independent, leakage-free evaluation groups. Matching criteria, label consistency, and case-level results must be checked before drawing conclusions.

The project also includes analysis of dataset label mappings and per-case metrics. External results should be reported separately from primary test results, with attention to possible training-data overlap and differences in dataset composition.

## Repository Structure

The main directories used by the project include:

```text
segformer3D/
├── configs/       # Experiment configurations
├── scripts/       # Data preparation and analysis utilities
├── src/           # Model and project source code
├── figures/       # Research figures
├── outputs/       # Generated plots and analysis artifacts
├── results/       # Experiment result files
└── README.md      # Project documentation
```

Additional directories and files may be present. Refer to the repository for the complete implementation and experiment artifacts.

## Environment and Dependencies

The recorded research environment used PyTorch 2.4.0a0+f70bd71a48.nv24.06 on a GPU-enabled server.

The exact CUDA version, dependency versions, and hardware configuration should be taken from the environment used for the corresponding experiment. Do not assume that the recorded environment is identical to a local workstation.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anushkap0/segformer3D.git
cd segformer3D
```

### 2. Inspect experiment configurations

Review the YAML files under `configs/` to identify the model configuration, dataset paths, input dimensions, and evaluation settings.

```bash
find configs -maxdepth 2 -type f
```

### 3. Set up the environment

Use the dependency instructions or environment files included in the repository, if available. Install a PyTorch build compatible with the target machine's CUDA environment when using a GPU.

Do not assume that installing a different PyTorch version will reproduce the original results exactly.

### 4. Prepare the dataset

Use the existing preparation scripts under `scripts/` and follow their documented arguments. Dataset files must be obtained from their original sources and placed in the paths expected by the selected configuration.

Medical imaging datasets may have separate access terms and usage restrictions.

### 5. Evaluate an existing checkpoint

To evaluate an existing model, locate the evaluation script and checkpoint path referenced by the experiment configuration and logs. Use the script's actual supported command-line arguments.

This repository contains experiment analysis artifacts; the availability of every dataset, checkpoint, and dependency required to reproduce each run should be verified before attempting evaluation.

**Evaluation does not require retraining** when the required checkpoint and prepared dataset are already available.

## Results and Reproducibility

For reliable comparisons:

- Use the same evaluation split when comparing models.
- Report the tumor regions and Dice aggregation method.
- Include both seeds where available.
- Report HD95 alongside Dice when computed and validated.
- Separate primary test results from external evaluation.
- Document dataset overlap and label-mapping checks.
- Report inference cost with the input dimensions and measurement conditions.
- Preserve experiment configurations and logs associated with each checkpoint.

## Limitations

- Performance can vary with random seed, dataset composition, and preprocessing.
- External dataset overlap can complicate claims about generalization.
- Efficiency numbers depend on hardware and measurement conditions.
- A change in Dice score alone does not establish statistical significance.
- Results from one dataset do not establish clinical readiness.

This project is intended for research and educational evaluation, not direct clinical diagnosis or treatment.

## Future Work

Potential follow-up work includes stronger statistical analysis, additional independent external validation, systematic reproducibility checks, and further investigation of boundary-aware objectives.

## Acknowledgements

This work builds on research into transformer-based 3D medical image segmentation and publicly available brain tumor segmentation datasets. Please consult the relevant papers, dataset documentation, and upstream implementation licenses when using or extending this work.

## License

Check the repository for a `LICENSE` file before redistributing or reusing this project. No license is implied by this README.


