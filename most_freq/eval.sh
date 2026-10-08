#!/usr/bin/env bash
# Evaluate the paper sweep's selected best checkpoints on the fixed ID/OOD panel.
# Defaults to bundled paper checkpoints; OUTPUT_ROOT selects another sweep.
# CHECKPOINT selects one checkpoint instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-most_freq/ckpt}"
EVAL_ROOT="${EVAL_ROOT:-outputs/paper-eval/most-freq}"
read -r -a centers <<< "${CENTERS:-${CENTER_WIDTH:-64 32 16}}"
read -r -a seeds <<< "${SEEDS:-${SEED:-42 43 44}}"
read -r -a models <<< "${MODELS:-${MODEL_FAMILY:-trace_relay}}"
read -r -a lengths <<< "${EVAL_LENGTHS:-128 192 256 512 1024}"
checkpoints=()
outputs=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  checkpoints+=("$CHECKPOINT")
  outputs+=("$EVAL_ROOT/$(basename "$(dirname "$CHECKPOINT")")")
else
  for seed in "${seeds[@]}"; do
    for center in "${centers[@]}"; do
      for model in "${models[@]}"; do
        name="most-freq-${model}-center${center}-seed${seed}"
        checkpoints+=("$OUTPUT_ROOT/$name/best.pt")
        outputs+=("$EVAL_ROOT/$name")
      done
    done
  done
fi

# Check every destination before starting any evaluation; never overwrite results.
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for index in "${!checkpoints[@]}"; do
    if [[ ! -f "${checkpoints[$index]}" ]]; then
      printf 'Missing checkpoint: %s\n' "${checkpoints[$index]}" >&2
      exit 1
    fi
    if [[ -e "${outputs[$index]}" ]]; then
      printf 'Refusing existing evaluation output: %s\n' "${outputs[$index]}" >&2
      exit 1
    fi
  done
fi

for index in "${!checkpoints[@]}"; do
  command=("$PYTHON" -u -m most_freq.evaluate
    --checkpoint "${checkpoints[$index]}" --output "${outputs[$index]}"
    --lengths "${lengths[@]}" --examples-per-length "${EXAMPLES_PER_LENGTH:-1024}"
    --batch-size "${EVAL_BATCH_SIZE:-64}" --seed "${EVAL_SEED:-20261006}"
    --device "${DEVICE:-cuda}" --precision "${PRECISION:-bf16}"
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --chunk-size "${EVAL_CHUNK_SIZE:-0}" --threads 1
    --sampler separated_counts --min-count-gap 4 --suffix-lengths 16 32 --example-limit 8
    "$@")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
done
