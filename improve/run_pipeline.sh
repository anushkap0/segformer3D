#!/usr/bin/env bash
# Inference-time improvement pipeline: flip-TTA + seed ensemble + validation-tuned
# post-processing, applied identically to SegFormer3D variants and the 3D U-Net.
# Edit the paths in the CONFIG block, then run step by step:
#   bash improve/run_pipeline.sh infer     # GPU, ~1 day for all models (resumable)
#   bash improve/run_pipeline.sh sanity    # must reproduce the frozen numbers
#   bash improve/run_pipeline.sh tune      # validation only
#   (commit pp/*.json and PROTOCOL.md BEFORE the next step)
#   bash improve/run_pipeline.sh test      # test split and external set, run ONCE
#   bash improve/run_pipeline.sh stats
set -euo pipefail
cd "$(dirname "$0")/.."

# ------------------------------------------------------------------ CONFIG (edit)
DATA=data/processed/BraTS2021_FULL
MANIFEST=$DATA/split_manifest.json
EXT_DATA=data/processed/BraTS2017ext
EXT_ALL=groups/external_all_484.txt
EXT_ZS=groups/zero_shot_175.txt
declare -A CFG=( [base]=configs/v2_phase1_baseline.yaml [cont]=configs/v2_phase1_continued.yaml
                 [nol_s1]=configs/v2_abl_noloss.yaml [nol_s2]=configs/v2_abl_noloss_s2.yaml
                 [full_s1]=configs/v2_phase2_boundary.yaml [full_s2]=configs/v2_phase2_boundary_s2.yaml
                 [unet]=configs/phase4_baselines.yaml )
declare -A CKPT=( [base]=checkpoints/v2_phase1/best.pt [cont]=checkpoints/v2_continued/best.pt
                  [nol_s1]=checkpoints/v2_abl_noloss/best.pt [nol_s2]=checkpoints/v2_abl_noloss_s2/best.pt
                  [full_s1]=checkpoints/v2_phase2/best.pt [full_s2]=checkpoints/v2_phase2_s2/best.pt
                  [unet]=checkpoints/phase4_unet3d/best.pt )
MARGIN=0.01
declare -A ENS=( [unet]="unet" [full]="full_s1 full_s2" [seg4]="full_s1 full_s2 nol_s1 nol_s2" )
# ---------------------------------------------------------------------------------

PY="python improve"
step=${1:-}

case "$step" in
infer)
  for m in ${2:-${!CFG[@]}}; do
    for sp in val test; do
      $PY/infer_probs.py --config ${CFG[$m]} --checkpoint ${CKPT[$m]} --data_dir $DATA \
          --manifest $MANIFEST --split $sp --out_dir preds/$m/$sp
    done
    $PY/infer_probs.py --config ${CFG[$m]} --checkpoint ${CKPT[$m]} --data_dir $EXT_DATA \
        --cases_file $EXT_ALL --out_dir preds/$m/ext
  done ;;
sanity)
  # plain pass + default threshold must match results/main_table.csv (Dice to 4 decimals)
  for m in base cont full_s1 unet; do
    $PY/evaluate_pp.py --pred_dirs preds/$m/test --key p_plain --pp none --data_dir $DATA \
        --manifest $MANIFEST --split test --out_csv results_v2/sanity_$m.csv
  done ;;
tune)
  mkdir -p pp
  for name in "${!ENS[@]}"; do
    dirs=""; for m in ${ENS[$name]}; do dirs="$dirs preds/$m/val"; done
    for key in p_plain p_tta; do
      $PY/tune_val.py --pred_dirs $dirs --key $key --data_dir $DATA --manifest $MANIFEST \
          --split val --out pp/${name}_${key}.json
    done
  done ;;
test)
  for name in "${!ENS[@]}"; do
    for key in p_plain p_tta; do
      dt=""; de=""; for m in ${ENS[$name]}; do dt="$dt preds/$m/test"; de="$de preds/$m/ext"; done
      $PY/evaluate_pp.py --pred_dirs $dt --key $key --pp pp/${name}_${key}.json --data_dir $DATA \
          --manifest $MANIFEST --split test --out_csv results_v2/test_${name}_${key}.csv
      $PY/evaluate_pp.py --pred_dirs $de --key $key --pp pp/${name}_${key}.json --data_dir $EXT_DATA \
          --cases_file $EXT_ALL --out_csv results_v2/ext_${name}_${key}.csv
    done
  done ;;
stats)
  for cand in full seg4; do
    for key in p_plain p_tta; do
      $PY/stats.py --a results_v2/test_${cand}_${key}.csv --b results_v2/test_unet_${key}.csv \
          --margin $MARGIN --label "TEST  $cand vs unet ($key, both post-processed)"
      $PY/stats.py --a results_v2/ext_${cand}_${key}.csv --b results_v2/ext_unet_${key}.csv \
          --subset $EXT_ZS --margin $MARGIN --label "ZERO-SHOT  $cand vs unet ($key)"
    done
  done ;;
*) echo "usage: $0 {infer|sanity|tune|test|stats}"; exit 1 ;;
esac
