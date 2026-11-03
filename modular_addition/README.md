# Modulo-5 paper sweeps

Commands for **Attending to Memory Writes Reduces Seed Variability in Recurrent
Length Generalization**. Run from the repository root with PyTorch, Transformers,
and FlashAttention-2. The paper uses CUDA/BF16. W&B is optional and disabled by default.

## Commands

Preview commands without starting training:

```sh
bash modular_addition/paper_sweeps.sh main
bash modular_addition/paper_sweeps.sh placement
bash modular_addition/paper_sweeps.sh all-layer
bash modular_addition/paper_sweeps.sh scaffold
```

Start a selected sweep by adding `--execute`:

```sh
bash modular_addition/paper_sweeps.sh main --execute
bash modular_addition/paper_sweeps.sh placement --execute
bash modular_addition/paper_sweeps.sh all-layer --execute
bash modular_addition/paper_sweeps.sh scaffold --execute
```

| Group | Conditions | Seeds | New training runs |
|---|---|---|---:|
| `main` | Baseline, upper state, upper delta (0.05), upper input-SG (0.05) | 42–48 | 28 |
| `placement` | Lower/middle state; lower/middle/upper delta at 0.2, 0.1, 0.05 | 42–44 | 30; upper delta 0.05 is reused from `main` |
| `all-layer` | C: 0.05 → 0; A: fixed 0.05; B: fixed 0.02, in C → A → B order | 42–44 | 9 |
| `scaffold` | Upper delta decay and lower input-SG | 42–44 | 6 |

These groups cover 73 distinct condition/seed runs. Supplementary comparisons
reuse seeds 42–44 from `main`. Instrumented upper-fixed/input-SG controls are
included in `main` for those three seeds; historical diagnostic repeats are
not counted twice.

All runs use modulo 5, IID symbols, uniformly sampled training lengths 1–40,
three layers, hidden width 128, SwiGLU width 512, four heads, write window 5,
40,000 updates, batch size 64, and the recorded AdamW/cosine schedule.
Read windows are 5 except for the selected state-attention layer (40).
Delta attention uses window 40 at selected layers.
Decay holds 0.05 through update 24,000 and reaches zero at update 36,000.

The matched initializer copies shared baseline weights, initializes each delta
module from a separate seeded stream, and restores training RNG state.
The ordinary trainer does not apply this matched initialization.

## Evaluation

The commands save LAST (40,000 updates) and BEST (minimum ordinary length-40
validation cross-entropy). Final evaluations use seed 20261024 and 1,024 examples
per length. Every group includes lengths 20, 40, 80, 160, and 320; delta sweeps
also retain 96/128, and all-layer sweeps also 640.
Seed 20261023 supplies 2,000 fixed length-40 validation examples.

Delta runs also evaluate the same checkpoints with correction disabled.
Training-path controls record length-40/80 probes at initialization, 250, 500,
1,000, and every 2,000 updates through 40,000, preserving RNG state and saving
diagnostic checkpoints. Ordinary validation logs retain every 250-update
observation so both 95% and 99% first-crossing times can be computed.
Paired-input diagnostics are auxiliary measurements, not paper outcome metrics.

## Runtime options

```sh
WANDB_MODE=online WANDB_ENTITY=your-team \
  bash modular_addition/paper_sweeps.sh main --execute

OUTPUT_ROOT=outputs/paper-reproduction PARALLEL=1 \
  bash modular_addition/paper_sweeps.sh placement --execute
```

Other environment options: `PYTHON`, `DEVICE`, `PRECISION`,
`ATTN_IMPLEMENTATION`, and `WANDB_PROJECT`.
Outputs: `outputs/modular-addition-paper/CONDITION/modular-addition-seedSEED/`.
Existing outputs/logs are refused. Use the same output root across groups.
`modular_addition/train.sh` delegates to this same launcher.
