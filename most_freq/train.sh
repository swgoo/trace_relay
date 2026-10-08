#!/usr/bin/env bash
# Paper protocol: paper/table/protocol.tex and paper/data/most_freq.json.
# Default: the complete sequential sweep. DRY_RUN=1 prints commands only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/paper-reproduction/most-freq}"
read -r -a centers <<< "${CENTERS:-${CENTER_WIDTH:-64 32 16}}"
read -r -a seeds <<< "${SEEDS:-${SEED:-42 43 44}}"
read -r -a models <<< "${MODELS:-${MODEL_FAMILY:-trace_relay}}"
sweep_options=(--execute)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  sweep_options=()
fi
tracking=(--wandb-mode "${WANDB_MODE:-online}" --wandb-project "${WANDB_PROJECT:-trace relay most freq}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  tracking+=(--wandb-entity "$WANDB_ENTITY")
fi

exec "$PYTHON" -u -m most_freq.sweep \
  --python "$PYTHON" --output-root "$OUTPUT_ROOT" --device "${DEVICE:-cuda}" \
  --centers "${centers[@]}" --seeds "${seeds[@]}" --models "${models[@]}" \
  "${sweep_options[@]}" -- \
  --precision "${PRECISION:-bf16}" --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --hidden-size 64 --intermediate-size 256 --heads 4 \
  --left-windows 15 15 15 --right-windows 7 7 7 --relay-strides 8 8 8 --skip-pairs \
  --vocab-size 5 --sampler separated_counts --min-count-gap 4 \
  --steps "${STEPS:-20000}" --batch-size 64 \
  --train-min-length 128 --train-max-length 256 \
  --validation-lengths 128 192 256 --validation-examples 512 --validation-seed 20261007 \
  --eval-lengths 128 192 256 512 1024 --eval-examples 1024 --eval-seed 20261006 \
  --eval-batch-size 64 --eval-chunk-size 0 --eval-every 250 --log-every 50 \
  --learning-rate 3e-4 --warmup-steps 100 --weight-decay 0.01 --grad-clip 1.0 --threads 1 \
  --early-stop-accuracy "${EARLY_STOP_ACCURACY:-0.99}" --early-stop-passes "${EARLY_STOP_PASSES:-2}" \
  "${tracking[@]}" "$@"
