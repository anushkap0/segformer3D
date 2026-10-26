#!/bin/bash
# usage: bash scripts/run_resumable.sh <config_name> <log_name> [max_restarts]
cd "$(dirname "$0")/.." || exit 1
cfg=$1; log=logs/$2.log; max=${3:-6}
ckdir=$(python -c "import yaml; print(yaml.safe_load(open('configs/$cfg.yaml'))['train']['checkpoint_dir'])")
[ -f "$log" ] || : > "$log"
for i in $(seq 0 "$max"); do
  resume=""
  [ -f "$ckdir/last.pt" ] && resume="--resume $ckdir/last.pt"
  python src/train.py --config configs/$cfg.yaml $resume >> "$log" 2>&1
  if tail -n 5 "$log" | grep -q "Training complete"; then exit 0; fi
  echo "[watchdog] run died (attempt $((i+1))), restarting in 30 s" >> "$log"
  sleep 30
done
echo "[watchdog] giving up after $max restarts" >> "$log"
