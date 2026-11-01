#!/bin/bash
# usage: bash scripts/run_test_once.sh <name> <config> <checkpoint_dir>
cd "$(dirname "$0")/.." || exit 1
n=$1; cfg=$2; ck=$3
mkdir -p logs/test
if [ -e logs/test/${n}_test.json ] || grep -q "^$n " logs/test/TEST_RUNS.txt 2>/dev/null; then
  echo "REFUSED: $n is already recorded as evaluated on test (logs/test/TEST_RUNS.txt)"; exit 1
fi
echo "$n $(date -u +%FT%TZ) $(sha256sum checkpoints/$ck/best.pt | cut -c1-16)" >> logs/test/TEST_RUNS.txt
python src/evaluate.py --config configs/$cfg.yaml --checkpoint checkpoints/$ck/best.pt --split test --output_json logs/test/${n}_test.json > logs/test/${n}_test.log 2>&1
