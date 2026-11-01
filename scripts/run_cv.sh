#!/bin/bash
cd /workspace/segformer3D
mkdir -p configs/cv
for K in 0 1 2 3 4; do
  mkdir -p logs/cv/fold_${K} checkpoints/cv/fold_${K}
  P1=configs/cv/phase1_fold${K}.yaml
  P2=configs/cv/phase2_fold${K}.yaml
  sed -e "s#split_manifest.json#fold_${K}_manifest.json#" \
      -e "s#checkpoint_dir: checkpoints/phase1#checkpoint_dir: checkpoints/cv/fold_${K}/phase1#" \
      -e "s#log_dir: logs/phase1#log_dir: logs/cv/fold_${K}/phase1#" \
      configs/phase1_baseline.yaml > $P1
  sed -e "s#split_manifest.json#fold_${K}_manifest.json#" \
      -e "s#init_from_phase1_checkpoint: checkpoints/phase1/best.pt#init_from_phase1_checkpoint: checkpoints/cv/fold_${K}/phase1/best.pt#" \
      -e "s#checkpoint_dir: checkpoints/phase2#checkpoint_dir: checkpoints/cv/fold_${K}/phase2#" \
      -e "s#log_dir: logs/phase2#log_dir: logs/cv/fold_${K}/phase2#" \
      configs/phase2_boundary.yaml > $P2
  [ -f checkpoints/cv/fold_${K}/phase1.DONE ] || { python src/train.py --config $P1 > logs/cv/fold_${K}/phase1_stdout.log 2>&1 && touch checkpoints/cv/fold_${K}/phase1.DONE; }
  [ -f checkpoints/cv/fold_${K}/phase2.DONE ] || { python src/train.py --config $P2 > logs/cv/fold_${K}/phase2_stdout.log 2>&1 && touch checkpoints/cv/fold_${K}/phase2.DONE; }
  [ -f logs/cv/fold_${K}/phase1_test.json ] || python src/evaluate.py --config $P1 --checkpoint checkpoints/cv/fold_${K}/phase1/best.pt --split test --output_json logs/cv/fold_${K}/phase1_test.json > logs/cv/fold_${K}/phase1_eval.log 2>&1
  [ -f logs/cv/fold_${K}/phase2_test.json ] || python src/evaluate.py --config $P2 --checkpoint checkpoints/cv/fold_${K}/phase2/best.pt --split test --output_json logs/cv/fold_${K}/phase2_test.json > logs/cv/fold_${K}/phase2_eval.log 2>&1
done
echo "CV finished"
