"""Full-volume evaluation with sliding-window inference (MONAI's
`sliding_window_inference`), since training uses random patches but real evaluation
must run on the whole (variable-size) volume for honest Dice/HD95 numbers -- patch-only
eval would inflate scores by discarding hard, low-signal regions outside the sampled
patch. Usage:

    python src/evaluate.py --config configs/phase1_baseline.yaml --checkpoint checkpoints/phase1/best.pt
    python src/evaluate.py --config configs/phase2_boundary.yaml --checkpoint checkpoints/phase2/best.pt --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import labels_to_brats_regions, load_split_manifest
from src.train import build_model
from src.utils.metrics import compute_region_metrics, aggregate_metrics


def load_full_volume(data_dir: str, case_id: str):
    case_dir = os.path.join(data_dir, case_id)
    image = np.load(os.path.join(case_dir, "image.npy"))
    label = np.load(os.path.join(case_dir, "label.npy"))
    with open(os.path.join(case_dir, "spacing.json")) as f:
        spacing = json.load(f)["spacing"]
    return image, label, spacing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_boundary_model = cfg["model"]["type"] == "segformer3d_boundary"
    eval_output = cfg.get("eval", {}).get("eval_output", "logits")
    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    mutually_exclusive = cfg["data"].get("mutually_exclusive", False)
    sw_overlap = cfg.get("eval", {}).get("sw_overlap", 0.5)
    patch_size = tuple(cfg["data"]["patch_size"])

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"reported best_mean_dice={ckpt.get('best_mean_dice', float('nan')):.4f}")

    manifest = load_split_manifest(cfg["data"]["manifest"])
    case_ids = manifest[args.split]
    data_dir = cfg["data"]["data_dir"]

    def predictor(x):
        outputs = model(x)
        if is_boundary_model:
            return outputs[eval_output]
        return outputs

    per_case_results = []
    per_case_ids = []
    for case_id in case_ids:
        image, label, spacing = load_full_volume(data_dir, case_id)
        image_t = torch.from_numpy(image).float().unsqueeze(0).to(device)  # (1, 4, D, H, W)

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = sliding_window_inference(
                image_t, roi_size=patch_size, sw_batch_size=1,
                predictor=predictor, overlap=sw_overlap, mode="gaussian",
            )
        logits = logits[0].float().cpu()

        target = torch.from_numpy(labels_to_brats_regions(label)).float()
        metrics = compute_region_metrics(
            logits, target, threshold=threshold, voxel_spacing=tuple(spacing),
            mutually_exclusive=mutually_exclusive,
        )
        per_case_results.append(metrics)
        per_case_ids.append(case_id)
        mean_dice = sum(metrics[r]["dice"] for r in metrics) / len(metrics)
        print(f"{case_id}: mean_dice={mean_dice:.4f} "
              + " ".join(f"{r}={metrics[r]['dice']:.4f}" for r in metrics))

    agg = aggregate_metrics(per_case_results)
    print("\n=== Aggregate results ===")
    for region, m in agg.items():
        print(f"{region}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {"aggregate": agg, "per_case": dict(zip(per_case_ids, per_case_results))},
                f, indent=2,
            )
        print(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    main()
