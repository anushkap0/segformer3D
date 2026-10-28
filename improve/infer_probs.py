"""Step 1 (GPU): run sliding-window inference once per model and save probabilities.

For every case it stores two maps in <out_dir>/<case_id>.npz:
  p_plain : identity pass only (reproduces the frozen evaluation)
  p_tta   : mean of the 8 flip combinations over the three spatial axes
Both are sigmoid probabilities of the WT, TC and ET channels.

Example
  python improve/infer_probs.py --config configs/phase2_boundary.yaml \
      --checkpoint checkpoints/phase2_s1/best.pt --manifest data/processed/BraTS/split_manifest.json \
      --split val --data_dir data/processed/BraTS --out_dir preds/full_s1/val

Model loading uses src.train.build_model(cfg) and the checkpoint key
"model_state_dict", exactly like src/evaluate.py. If the server code builds models
differently (e.g. src/models/official_wrappers.py), edit `load_model` only.
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference

from common import case_list, load_case, save_probs

FLIPS = [tuple(a for a, on in zip((2, 3, 4), mask) if on) for mask in itertools.product([0, 1], repeat=3)]


def load_model(cfg: dict, checkpoint: str, device):
    from src.train import build_model
    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--split")
    ap.add_argument("--cases_file", help="text file with one case id per line (e.g. the external set)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval_output", default=None, help="boundary models: refined_logits (default from config)")
    ap.add_argument("--sw_overlap", type=float, default=None)
    ap.add_argument("--no_tta", action="store_true", help="only the identity pass (faster)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    out_key = args.eval_output or cfg.get("eval", {}).get("eval_output", "refined_logits")
    roi = tuple(cfg["data"]["patch_size"])
    overlap = args.sw_overlap if args.sw_overlap is not None else cfg.get("eval", {}).get("sw_overlap", 0.5)

    def predictor(x):
        out = model(x)
        return out[out_key] if isinstance(out, dict) else out

    def infer(x):
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = sliding_window_inference(x, roi_size=roi, sw_batch_size=1, predictor=predictor,
                                              overlap=overlap, mode="gaussian")
        return torch.sigmoid(logits.float())

    os.makedirs(args.out_dir, exist_ok=True)
    cases = case_list(args.manifest, args.split, args.cases_file)
    t0 = time.time()
    for n, cid in enumerate(cases, 1):
        target = os.path.join(args.out_dir, f"{cid}.npz")
        if os.path.exists(target):
            continue  # resumable
        image, _, _ = load_case(args.data_dir, cid)
        x = torch.from_numpy(image).float().unsqueeze(0).to(device)
        plain = infer(x)
        probs = {"p_plain": plain[0].cpu().numpy()}
        if not args.no_tta:
            acc = plain.clone()
            for axes in FLIPS[1:]:
                acc += torch.flip(infer(torch.flip(x, axes)), axes)
            probs["p_tta"] = (acc / len(FLIPS))[0].cpu().numpy()
        save_probs(target, probs)
        print(f"[{n}/{len(cases)}] {cid}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
