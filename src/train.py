"""Training entrypoint. Usage:

    python src/train.py --config configs/phase1_baseline.yaml
    python src/train.py --config configs/phase2_boundary.yaml
    python src/train.py --config configs/phase4_baselines.yaml

Reads a YAML config (see configs/), builds the model/data/loss/optimizer accordingly,
and trains with a foreground-oversampling patch-based pipeline + cosine LR schedule +
AMP mixed precision. Saves the best (by val WT+TC+ET mean Dice) and last checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.brats_dataset import BraTSDataset, load_split_manifest
from src.losses.losses import DiceBCELoss, DualSupervisionCompositeLoss
from src.models.segformer3d import SegFormer3D
from src.models.boundary_refinement import SegFormer3DWithBoundary
from src.models.unet3d import UNet3D
from src.utils.metrics import compute_region_metrics, aggregate_metrics


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    if m["type"] == "segformer3d":
        model = SegFormer3D(
            in_channels=m["in_channels"], num_classes=m["num_classes"],
            embed_dims=m["embed_dims"], num_heads=m["num_heads"],
            sr_ratios=m["sr_ratios"], depths=m["depths"],
            decoder_dim=m["decoder_dim"], drop_path_rate=m["drop_path_rate"],
        )
    elif m["type"] == "segformer3d_boundary":
        base = SegFormer3D(
            in_channels=m["in_channels"], num_classes=m["num_classes"],
            embed_dims=m["embed_dims"], num_heads=m["num_heads"],
            sr_ratios=m["sr_ratios"], depths=m["depths"],
            decoder_dim=m["decoder_dim"], drop_path_rate=m["drop_path_rate"],
        )
        init_ckpt = m.get("init_from_phase1_checkpoint")
        if init_ckpt and os.path.exists(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu")
            base.load_state_dict(state["model_state_dict"], strict=True)
            print(f"Warm-started region branch from {init_ckpt}")
        model = SegFormer3DWithBoundary(
            base, decoder_dim=m["decoder_dim"], num_classes=m["num_classes"],
            voxel_spacing=tuple(m.get("voxel_spacing", (1.0, 1.0, 1.0))),
        )
    elif m["type"] == "unet3d":
        model = UNet3D(
            in_channels=m["in_channels"], num_classes=m["num_classes"],
            base_channels=m["base_channels"], depth=m["depth"],
        )
    else:
        raise ValueError(f"Unknown model type: {m['type']}")
    return model


def build_loss(cfg: dict):
    loss_cfg = cfg["train"]["loss"]
    me = cfg["data"].get("mutually_exclusive", False)
    if loss_cfg["type"] == "dice_bce":
        return DiceBCELoss(me, loss_cfg["dice_weight"], loss_cfg["bce_weight"]), False
    elif loss_cfg["type"] == "dual_supervision_composite":
        return DualSupervisionCompositeLoss(
            loss_cfg["region_weight"], loss_cfg["refined_weight"],
            loss_cfg["boundary_weight"], me,
        ), True
    else:
        raise ValueError(f"Unknown loss type: {loss_cfg['type']}")


def build_dataloaders(cfg: dict, return_boundary_target: bool):
    manifest = load_split_manifest(cfg["data"]["manifest"])
    train_ds = BraTSDataset(
        cfg["data"]["data_dir"], manifest["train"],
        patch_size=tuple(cfg["data"]["patch_size"]), train=True,
        return_boundary_target=return_boundary_target,
    )
    val_ds = BraTSDataset(
        cfg["data"]["data_dir"], manifest["val"],
        patch_size=tuple(cfg["data"]["patch_size"]), train=False,
        return_boundary_target=return_boundary_target,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, cfg["train"]["num_workers"] // 2), pin_memory=True,
    )
    return train_loader, val_loader


def get_logits_for_eval(outputs, is_boundary_model: bool, eval_output: str):
    if is_boundary_model:
        return outputs[eval_output]
    return outputs


def train_one_epoch(model, loader, optimizer, loss_fn, is_boundary_model, device, scaler, grad_clip_norm):
    model.train()
    total_loss = 0.0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(scaler is not None)):
            outputs = model(image)
            if is_boundary_model:
                boundary_target = batch["boundary_target"].to(device, non_blocking=True)
                loss_dict = loss_fn(outputs, target, boundary_target)
                loss = loss_dict["total"]
            else:
                loss = loss_fn(outputs, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, is_boundary_model, eval_output, device, mutually_exclusive, threshold):
    model.eval()
    per_case_results = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"][0]  # drop batch dim, val loader has batch_size=1
        spacing = tuple(batch["spacing"][0].tolist())

        outputs = model(image)
        logits = get_logits_for_eval(outputs, is_boundary_model, eval_output)[0].cpu()

        metrics = compute_region_metrics(
            logits, target, threshold=threshold, voxel_spacing=spacing,
            mutually_exclusive=mutually_exclusive,
        )
        per_case_results.append(metrics)
    return aggregate_metrics(per_case_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device found, training on CPU will be extremely slow "
              "for 3D volumes -- this is fine for a shape/smoke test, not for real training.")

    model = build_model(cfg).to(device)
    is_boundary_model = cfg["model"]["type"] == "segformer3d_boundary"
    loss_fn, _ = build_loss(cfg)
    train_loader, val_loader = build_dataloaders(cfg, return_boundary_target=is_boundary_model)

    print(f"Model: {cfg['model']['type']}, params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Train cases: {len(train_loader.dataset)}, Val cases: {len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    epochs = cfg["train"]["epochs"]
    warmup_epochs = cfg["train"].get("warmup_epochs", 0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if (cfg["train"].get("amp", True) and device.type == "cuda") else None

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["train"]["log_dir"])

    start_epoch = 0
    best_mean_dice = -1.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_mean_dice = ckpt.get("best_mean_dice", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    eval_output = cfg.get("eval", {}).get("eval_output", "logits")
    mutually_exclusive = cfg["data"].get("mutually_exclusive", False)
    threshold = cfg.get("eval", {}).get("threshold", 0.5)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, is_boundary_model, device, scaler,
            cfg["train"].get("grad_clip_norm", 1.0),
        )
        scheduler.step()
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        log_line = f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} time={time.time()-t0:.1f}s"

        if (epoch + 1) % cfg["train"].get("val_interval", 5) == 0 or epoch == epochs - 1:
            val_metrics = validate(
                model, val_loader, is_boundary_model, eval_output, device,
                mutually_exclusive, threshold,
            )
            region_names = list(val_metrics.keys())
            mean_dice = sum(val_metrics[r]["dice"] for r in region_names) / len(region_names)
            for r in region_names:
                for m_name, m_val in val_metrics[r].items():
                    writer.add_scalar(f"val/{r}_{m_name}", m_val, epoch)
            log_line += f" val_mean_dice={mean_dice:.4f} " + " ".join(
                f"{r}_dice={val_metrics[r]['dice']:.4f}" for r in region_names
            )

            ckpt = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_dice": best_mean_dice, "config": cfg,
            }
            torch.save(ckpt, os.path.join(cfg["train"]["checkpoint_dir"], "last.pt"))
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                ckpt["best_mean_dice"] = best_mean_dice
                torch.save(ckpt, os.path.join(cfg["train"]["checkpoint_dir"], "best.pt"))
                log_line += " [new best]"

        print(log_line)

    writer.close()
    print(f"Training complete. Best val mean Dice: {best_mean_dice:.4f}")


if __name__ == "__main__":
    main()
