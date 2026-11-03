#!/usr/bin/env bash
# Paper configurations only; print commands unless --execute is supplied.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  cat <<'EOF'
Usage: bash modular_addition/paper_sweeps.sh GROUP [--dry-run|--execute]
  main        28 runs: baseline, upper state/delta/input-SG; seeds 42--48
  placement   30 additional runs: lower/middle state and delta placement/scale
  all-layer    9 runs: C (decay), A (0.05), B (0.02); seeds 42--44
  scaffold     6 additional runs: upper decay and lower input-SG; seeds 42--44
Upper fixed/input-SG diagnostics for seeds 42--44 are already included in main.
Runtime options: PYTHON, OUTPUT_ROOT, PARALLEL, DEVICE, PRECISION,
ATTN_IMPLEMENTATION, WANDB_MODE, WANDB_PROJECT, WANDB_ENTITY.
EOF
}
group="${1:-main}"
mode="${2:---dry-run}"
if [[ "$group" == "--help" || "$group" == "-h" ]]; then usage; exit 0; fi
if (( $# > 2 )); then usage >&2; exit 2; fi
case "$group" in main|placement|all-layer|scaffold) ;; *) usage >&2; exit 2 ;; esac
case "$mode" in
  --dry-run) execution=(--dry-run) ;;
  --execute) execution=(--execute) ;;
  *) usage >&2; exit 2 ;;
esac

paper_python="${PYTHON:-python3}"
paper_output="${OUTPUT_ROOT:-outputs/modular-addition-paper}"
tracking=(--wandb-mode "${WANDB_MODE:-disabled}" --wandb-project "${WANDB_PROJECT:-trace_relay_modular_addition}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then tracking+=(--wandb-entity "$WANDB_ENTITY"); fi
common=(
  --modulus 5 --sequence-length 40 --train-min-length 1
  --layers 3 --hidden-size 128 --intermediate-size 512 --heads 4
  --write-windows 5 --steps 40000 --batch-size 64
  --precision "${PRECISION:-bf16}" --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
  --learning-rate 0.0003 --weight-decay 0.01 --grad-clip 1
  --warmup-steps 100 --lr-scheduler cosine --lr-schedule-steps 40000 --min-lr-ratio 0.1
  --validation-examples 2000 --validation-seed 20261023
  --probe-examples 512 --eval-every 250 --length-eval-every 1000
  --eval-examples 1024 --eval-seed 20261024 --eval-batch-size 64 --threads 1
)
diagnostics=(--delta-diagnostics-every 2000 --delta-diagnostics-lengths 40 80 --save-diagnostic-checkpoints)

run() {
  local condition="$1" seed_list="$2"
  shift 2
  local -a paper_seeds evaluation
  read -r -a paper_seeds <<< "$seed_list"
  case "$condition" in
    state_*) evaluation=(--eval-lengths 20 40 80 160 320 --paired-validation-length 80) ;;
    all_*) evaluation=(--eval-lengths 20 40 80 96 128 160 320 640 --paired-validation-length 160) ;;
    *) evaluation=(--eval-lengths 20 40 80 96 128 160 320 --paired-validation-length 128) ;;
  esac
  "$paper_python" -u -m modular_addition.sweep \
    --python "$paper_python" --matched-initialization \
    --output-root "$paper_output/$condition" --seeds "${paper_seeds[@]}" \
    --parallel "${PARALLEL:-3}" --device "${DEVICE:-cuda}" \
    "${execution[@]}" "${tracking[@]}" -- \
    "${common[@]}" "${evaluation[@]}" "$@"
}
case "$group" in
  main)
    run baseline "42 43 44 45 46 47 48" --read-windows 5 --delta-windows 0
    run state_upper "42 43 44 45 46 47 48" --read-windows 5 5 40 --delta-windows 0
    # Historical diagnostic repeats reuse these same training seeds.
    run delta_upper_0p05 "42 43 44" --read-windows 5 --delta-windows 0 0 40 \
      --phase-correction-scale 0.05 --final-delta-ablation "${diagnostics[@]}"
    run delta_upper_0p05 "45 46 47 48" --read-windows 5 --delta-windows 0 0 40 \
      --phase-correction-scale 0.05 --final-delta-ablation
    run upper_input_sg "42 43 44" --read-windows 5 --delta-windows 0 0 40 \
      --phase-correction-scale 0.05 --delta-detach-inputs --final-delta-ablation "${diagnostics[@]}"
    run upper_input_sg "45 46 47 48" --read-windows 5 --delta-windows 0 0 40 \
      --phase-correction-scale 0.05 --delta-detach-inputs --final-delta-ablation
    ;;
  placement)
    run state_lower "42 43 44" --read-windows 40 5 5 --delta-windows 0
    run state_middle "42 43 44" --read-windows 5 40 5 --delta-windows 0
    for scale in 0.2 0.1 0.05; do
      label="${scale/./p}"
      run "delta_lower_$label" "42 43 44" --read-windows 5 --delta-windows 40 0 0 \
        --phase-correction-scale "$scale" --final-delta-ablation
      run "delta_middle_$label" "42 43 44" --read-windows 5 --delta-windows 0 40 0 \
        --phase-correction-scale "$scale" --final-delta-ablation
      # Upper 0.05 is already in main; placement tables reuse seeds 42--44.
      if [[ "$scale" != "0.05" ]]; then
        run "delta_upper_$label" "42 43 44" --read-windows 5 --delta-windows 0 0 40 \
          --phase-correction-scale "$scale" --final-delta-ablation
      fi
    done
    ;;
  all-layer)
    run all_C "42 43 44" --read-windows 5 --delta-windows 40 40 40 \
      --phase-correction-scale 0.05 --phase-correction-schedule linear_decay \
      --phase-correction-decay-start 24000 --phase-correction-decay-end 36000 --final-delta-ablation
    run all_A "42 43 44" --read-windows 5 --delta-windows 40 40 40 \
      --phase-correction-scale 0.05 --final-delta-ablation
    run all_B "42 43 44" --read-windows 5 --delta-windows 40 40 40 \
      --phase-correction-scale 0.02 --final-delta-ablation
    ;;
  scaffold)
    run upper_decay "42 43 44" --read-windows 5 --delta-windows 0 0 40 \
      --phase-correction-scale 0.05 --phase-correction-schedule linear_decay \
      --phase-correction-decay-start 24000 --phase-correction-decay-end 36000 \
      --final-delta-ablation "${diagnostics[@]}"
    run lower_input_sg "42 43 44" --read-windows 5 --delta-windows 40 0 0 \
      --phase-correction-scale 0.05 --delta-detach-inputs --final-delta-ablation "${diagnostics[@]}"
    ;;
esac
