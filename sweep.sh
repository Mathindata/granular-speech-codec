#!/usr/bin/env bash
# Rate sweep: 3 quantizers x 4 codebook sizes = 12 runs.
# Usage: bash sweep.sh /path/to/train-clean-100 /path/to/dev-clean STEPS
# Each run writes runs/<quantizer>_<K>/eval.jsonl; then run plot_results.py.
set -euo pipefail
TRAIN=${1:?train dir}
EVAL=${2:?eval dir}
STEPS=${3:-40000}
for K in 16 64 256 1024; do
  for Q in fsq vq fuzzy; do
    python train.py --data_dir "$TRAIN" --eval_dir "$EVAL" --quantizer "$Q" \
      --codebook_size "$K" --steps "$STEPS" --w_content 1 --w_speaker 1 \
      --out "runs/${Q}_${K}"
  done
done
python plot_results.py --runs runs --out results.png
