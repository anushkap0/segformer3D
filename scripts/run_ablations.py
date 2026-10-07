"""Runs the full Phase 1 -> Phase 2 ablation sweep and aggregates results into a
single comparison table (CSV + Markdown), mirroring PFF-Net's ablation table
structure (their variants A/B/C isolating boundary branch presence, boundary
supervision, and dual-supervision weighting).

Sweep order matters: phase1_baseline must run first (and complete) because every
phase2_* config warm-starts from `checkpoints/phase1/best.pt`. This script enforces
that ordering and skips training for any config whose checkpoint already exists,
so you can re-run this after a partial/interrupted sweep without redoing finished
runs.

Usage:
    python scripts/run_ablations.py                       # full sweep: train + eval
    python scripts/run_ablations.py --eval_only            # skip training, just re-eval
                                                             # existing checkpoints
    python scripts/run_ablations.py --configs phase1_baseline phase2_boundary
                                                             # run a subset

This calls src/train.py and src/evaluate.py as subprocesses (rather than importing
their internals directly) so each run gets a clean process -- avoids any state
leaking between runs (CUDA memory fragmentation, RNG state, etc.) that can happen
when training multiple models back-to-back in one Python process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (config_stem, checkpoint_dir, human-readable label) in dependency order.
# phase1_baseline MUST come first since the phase2_* configs load its checkpoint.
DEFAULT_SWEEP = [
    ("phase1_baseline", "checkpoints/phase1", "SegFormer3D (no boundary branch)"),
    ("phase2_ablation_no_boundary_loss", "checkpoints/phase2_ablation_no_boundary_loss",
     "+ boundary branch, boundary_weight=0 (architecture only, no supervision)"),
    ("phase2_boundary", "checkpoints/phase2",
     "+ boundary branch + dual supervision, boundary_weight=0.5 (full model)"),
    ("phase2_ablation_boundary_weight_high", "checkpoints/phase2_ablation_boundary_weight_high",
     "+ boundary branch + dual supervision, boundary_weight=1.0"),
]


def run_train(config_name: str) -> None:
    config_path = os.path.join(REPO_ROOT, "configs", f"{config_name}.yaml")
    cmd = [sys.executable, os.path.join(REPO_ROOT, "src", "train.py"), "--config", config_path]
    print(f"\n{'='*80}\nTRAINING: {config_name}\n{'='*80}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_eval(config_name: str, checkpoint: str, output_json: str, split: str = "test") -> dict:
    config_path = os.path.join(REPO_ROOT, "configs", f"{config_name}.yaml")
    cmd = [
        sys.executable, os.path.join(REPO_ROOT, "src", "evaluate.py"),
        "--config", config_path, "--checkpoint", checkpoint,
        "--split", split, "--output_json", output_json,
    ]
    print(f"\n{'-'*80}\nEVALUATING: {config_name} ({checkpoint})\n{'-'*80}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    with open(output_json) as f:
        return json.load(f)["aggregate"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                         help="subset of config stems to run, e.g. phase1_baseline phase2_boundary")
    parser.add_argument("--eval_only", action="store_true",
                         help="skip training, only (re-)evaluate existing checkpoints")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--results_dir", default="logs/ablation_sweep")
    args = parser.parse_args()

    sweep = DEFAULT_SWEEP
    if args.configs:
        sweep = [s for s in DEFAULT_SWEEP if s[0] in args.configs]
        missing = set(args.configs) - {s[0] for s in sweep}
        if missing:
            print(f"WARNING: unknown config stems requested, ignoring: {missing}")

    os.makedirs(args.results_dir, exist_ok=True)
    rows = []

    for config_name, ckpt_dir, label in sweep:
        best_ckpt = os.path.join(REPO_ROOT, ckpt_dir, "best.pt")

        if not args.eval_only:
            if os.path.exists(best_ckpt):
                print(f"Checkpoint already exists for {config_name} ({best_ckpt}), skipping training. "
                      f"Delete it manually to force a re-run.")
            else:
                run_train(config_name)

        if not os.path.exists(best_ckpt):
            print(f"SKIPPING EVAL for {config_name}: no checkpoint found at {best_ckpt} "
                  f"(training may have failed or --eval_only was set before any training ran).")
            continue

        output_json = os.path.join(args.results_dir, f"{config_name}_{args.split}_results.json")
        agg = run_eval(config_name, best_ckpt, output_json, args.split)

        row = {"config": config_name, "label": label}
        for region, metrics in agg.items():
            for m_name, m_val in metrics.items():
                row[f"{region}_{m_name}"] = round(m_val, 4)
        mean_dice = sum(agg[r]["dice"] for r in agg) / len(agg)
        mean_hd95 = sum(agg[r]["hd95"] for r in agg) / len(agg)
        row["mean_dice"] = round(mean_dice, 4)
        row["mean_hd95"] = round(mean_hd95, 4)
        rows.append(row)

    if not rows:
        print("No results collected -- nothing to write.")
        return

    # CSV
    csv_path = os.path.join(args.results_dir, "ablation_comparison.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")

    # Markdown table (drop-in for a paper/thesis writeup)
    md_path = os.path.join(args.results_dir, "ablation_comparison.md")
    with open(md_path, "w") as f:
        f.write("# Ablation comparison\n\n")
        f.write("| Config | Mean Dice | Mean HD95 | WT Dice | TC Dice | ET Dice |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['label']} | {r['mean_dice']} | {r['mean_hd95']} | "
                f"{r.get('WT_dice', '-')} | {r.get('TC_dice', '-')} | {r.get('ET_dice', '-')} |\n"
            )
    print(f"Wrote {md_path}")

    print("\n=== Summary ===")
    for r in rows:
        print(f"{r['label']}: mean_dice={r['mean_dice']} mean_hd95={r['mean_hd95']}")


if __name__ == "__main__":
    main()
