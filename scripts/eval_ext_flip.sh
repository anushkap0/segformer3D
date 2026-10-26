#!/bin/bash
cd /workspace/segformer3D; mkdir -p results/external_flip
while read name extcfg ckpt; do
  python src/evaluate.py --config configs/$extcfg.yaml --checkpoint checkpoints/$ckpt/best.pt --split test \
    --data_dir data/processed/BraTS2017ext_flip --manifest data/processed/BraTS2017ext_flip/split_manifest.json \
    --output_json results/external_flip/$name.json > results/external_flip/$name.log 2>&1
done <<'LIST'
baseline  ext_v2_phase1         v2_phase1
continued ext_v2_continued      v2_continued
noloss    ext_v2_abl_noloss     v2_abl_noloss
noloss_s2 ext_v2_abl_noloss_s2  v2_abl_noloss_s2
full      ext_v2_phase2         v2_phase2
full_s2   ext_v2_phase2_s2      v2_phase2_s2
unet      ext_unet              phase4_unet3d
LIST
