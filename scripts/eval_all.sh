#!/bin/bash
cd /workspace/segformer3D; mkdir -p results/test results/external
while read name cfg extcfg ckpt; do
  python src/evaluate.py --config configs/$cfg.yaml    --checkpoint checkpoints/$ckpt/best.pt --split test --output_json results/test/$name.json     > results/test/$name.log 2>&1
  python src/evaluate.py --config configs/$extcfg.yaml --checkpoint checkpoints/$ckpt/best.pt --split test --output_json results/external/$name.json > results/external/$name.log 2>&1
done <<'LIST'
baseline  v2_phase1_baseline    ext_v2_phase1         v2_phase1
continued v2_phase1_continued   ext_v2_continued      v2_continued
noloss    v2_abl_noloss         ext_v2_abl_noloss     v2_abl_noloss
full      v2_phase2_boundary    ext_v2_phase2         v2_phase2
unet      phase4_baselines      ext_unet              phase4_unet3d
noloss_s2 v2_abl_noloss_s2      ext_v2_abl_noloss_s2  v2_abl_noloss_s2
full_s2   v2_phase2_boundary_s2 ext_v2_phase2_s2      v2_phase2_s2
LIST
