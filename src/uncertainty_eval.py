"""Phase 5 (MC-Dropout path): runs stochastic multi-sample inference on an existing
Phase 2 checkpoint -- no retraining required -- and reports both segmentation
accuracy and how well predictive uncertainty tracks actual error, which is the
number that actually supports a "clinically useful uncertainty" claim (a model can
have high mean Dice and still produce uncertainty maps that are uncorrelated with
where it's wrong -- that would undercut the framing, so check this rather than
assuming it).

Usage:
    python src/uncertainty_eval.py --config configs/phase2_boundary.yaml \\
        --checkpoint checkpoints/phase2/best.pt --num_samples 20 --split test \\
        --output_dir logs/phase5_uncertainty

Outputs per case:
  - mean_prob, predictive_variance saved as .npy (for qualitative figures)
  - a per-case CSV row with: dice, mean predictive variance inside the predicted
    lesion, mean predictive variance outside it, and an uncertainty-error
    correlation summary (see `compute_uncertainty_calibration`)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import labels_to_brats_regions, load_split_manifest
from src.evaluate import load_full_volume
from src.models.uncertainty import enable_mc_dropout
from src.train import build_model
from src.utils.metrics import compute_region_metrics


def compute_uncertainty_calibration(
    pred_binary: np.ndarray, target_binary: np.ndarray, variance_map: np.ndarray
) -> dict:
    """A simple, transparent calibration check: is mean predictive variance higher
    on voxels where the model is WRONG (false positive or false negative) than on
    voxels where it's RIGHT? This should hold if the uncertainty estimate is doing
    its job; if it doesn't hold, say so plainly rather than reporting only the
    favorable numbers.
    """
    error_mask = pred_binary != target_binary
    correct_mask = ~error_mask
    mean_var_error = float(variance_map[error_mask].mean()) if error_mask.sum() > 0 else float("nan")
    mean_var_correct = float(variance_map[correct_mask].mean()) if correct_mask.sum() > 0 else float("nan")
    return {
        "mean_variance_on_errors": mean_var_error,
        "mean_variance_on_correct": mean_var_correct,
        "separation": mean_var_error - mean_var_correct if not (np.isnan(mean_var_error) or np.isnan(mean_var_correct)) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", default="logs/phase5_uncertainty")
    parser.add_argument("--save_maps", action="store_true", help="save per-case .npy uncertainty maps (can be large)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_boundary_model = cfg["model"]["type"] == "segformer3d_boundary"
    eval_output = cfg.get("eval", {}).get("eval_output", "logits")
    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    mutually_exclusive = cfg["data"].get("mutually_exclusive", False)
    patch_size = tuple(cfg["data"]["patch_size"])

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    manifest = load_split_manifest(cfg["data"]["manifest"])
    case_ids = manifest[args.split]
    data_dir = cfg["data"]["data_dir"]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_maps:
        os.makedirs(os.path.join(args.output_dir, "maps"), exist_ok=True)

    csv_rows = []
    region_names = ("WT", "TC", "ET")

    for case_id in case_ids:
        image, label, spacing = load_full_volume(data_dir, case_id)
        image_t = torch.from_numpy(image).float().unsqueeze(0).to(device)
        target = torch.from_numpy(labels_to_brats_regions(label)).float()

        # MC-Dropout via repeated sliding-window passes. This is more expensive
        # than single-pass inference by a factor of num_samples -- for large
        # volumes / many test cases, consider running on a representative subset
        # first (e.g. 10-20 cases) rather than the full test split.
        enable_mc_dropout(model)
        samples = []
        for _ in range(args.num_samples):
            def predictor(x, _model=model):
                outputs = _model(x)
                logits = outputs[eval_output] if is_boundary_model else outputs
                return torch.sigmoid(logits) if not mutually_exclusive else torch.softmax(logits, dim=1)

            with torch.no_grad():
                probs = sliding_window_inference(
                    image_t, roi_size=patch_size, sw_batch_size=1,
                    predictor=predictor, overlap=0.5, mode="gaussian",
                )
            samples.append(probs[0].cpu())

        stacked = torch.stack(samples, dim=0)  # (N, C, D, H, W)
        mean_prob = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)

        pred_binary = (mean_prob > threshold).numpy()
        target_binary = target.numpy().astype(bool)

        metrics = compute_region_metrics(
            torch.log(mean_prob.clamp(min=1e-7) / (1 - mean_prob.clamp(max=1 - 1e-7))),  # back to logit space for the metric fn's sigmoid
            target, threshold=threshold, voxel_spacing=tuple(spacing),
            mutually_exclusive=mutually_exclusive,
        )

        row = {"case_id": case_id}
        for i, region in enumerate(region_names):
            row[f"{region}_dice"] = metrics[region]["dice"]
            calib = compute_uncertainty_calibration(
                pred_binary[i], target_binary[i], variance[i].numpy()
            )
            row[f"{region}_var_on_errors"] = calib["mean_variance_on_errors"]
            row[f"{region}_var_on_correct"] = calib["mean_variance_on_correct"]
            row[f"{region}_var_separation"] = calib["separation"]
        csv_rows.append(row)

        if args.save_maps:
            np.save(os.path.join(args.output_dir, "maps", f"{case_id}_mean_prob.npy"), mean_prob.numpy())
            np.save(os.path.join(args.output_dir, "maps", f"{case_id}_variance.npy"), variance.numpy())

        print(f"{case_id}: " + " ".join(f"{region}_dice={row[f'{region}_dice']:.4f}" for region in region_names)
              + " | " + " ".join(f"{region}_var_sep={row[f'{region}_var_separation']:.5f}" for region in region_names))

    csv_path = os.path.join(args.output_dir, "uncertainty_calibration.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # summary: mean separation across cases per region -- positive means variance
    # is meaningfully higher on error voxels than correct voxels (good calibration
    # signal); near-zero or negative means the uncertainty map isn't tracking error
    # well and that should be reported as a limitation, not glossed over.
    summary = {}
    for region in region_names:
        seps = [r[f"{region}_var_separation"] for r in csv_rows if not np.isnan(r[f"{region}_var_separation"])]
        summary[region] = {
            "mean_dice": float(np.mean([r[f"{region}_dice"] for r in csv_rows])),
            "mean_var_separation": float(np.mean(seps)) if seps else float("nan"),
        }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Uncertainty calibration summary ===")
    for region, s in summary.items():
        print(f"{region}: mean_dice={s['mean_dice']:.4f} mean_var_separation={s['mean_var_separation']:.6f} "
              f"({'uncertainty tracks error' if s['mean_var_separation'] > 0 else 'WEAK/NO calibration signal -- report this'})")
    print(f"\nWrote {csv_path} and summary.json to {args.output_dir}")


if __name__ == "__main__":
    main()
