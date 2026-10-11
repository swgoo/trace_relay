# TraceRelay

This is the repository for [TraceRelay: Attention-Aligned Recurrence over Rolling Traces](https://arxiv.org/abs/2610.11743).
The paper's experiments and checkpoints are available in [v0.0.1](https://github.com/swgoo/trace_relay/tree/v0.0.1).

Transformers-compatible causal language model with local Left attention and
delayed Right updates accumulated by a parallel recurrent phase scan.
U-Net stages share one dimension for hidden representations, attention, and
phase state (`hidden_sizes` / `depths`).

## BabyLM 2026 Strict

As a language modeling feasibility test, the 102.13M-parameter byte model
achieved approximately **65% BLiMP accuracy** (65.60% macro) after
**three epochs on BabyLM 2026 Strict**.

See [babylm/README.md](babylm/README.md) for data preparation, the 102.13M-parameter
16-layer byte model, training commands, and BLiMP evaluation.

```bash
python -m pip install -e '.[babylm,test]'
bash babylm/prepare.sh
bash babylm/train.sh --output outputs/babylm-unet16-seed42
bash babylm/eval.sh --checkpoint outputs/babylm-unet16-seed42/best \
  --output outputs/babylm-unet16-seed42/blimp
```

The training default is CUDA / BF16 / FlashAttention-2. Install the `flash` extra
for that backend. CPU and eager attention are also supported.

```bash
python -m pytest -q
```
