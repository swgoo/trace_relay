# Most-Freq (five symbols) paper checkpoints

These 9 selected `best.pt` checkpoints cover centers 64/32/16,
training seeds 42/43/44, and phase inheritance on.
They are the selected models recorded in `paper/data/most_freq.json`.

Weights remain FP32 with exactly the original tensor values. Configuration and
parameter names use the current `left`/`right` terminology. Optimizer and random
generator states are omitted; these files are for evaluation and cannot resume
the original training run.

Each run is stored as `<run-id>/best.pt`. `manifest.json` records file sizes,
SHA-256 hashes, original source checkpoint hashes, seeds, centers, selected steps,
and parameter counts. The checkpoint retains its experiment protocol with the
config names updated, W&B account identifiers removed, and local output paths
made repository-relative, plus export provenance. Total checkpoint size:
**9.62 MiB**.

From the repository root, evaluate the complete published sweep:

```bash
bash most_freq/eval.sh
```

The default evaluation uses CUDA BF16, FlashAttention-2, seed 20261006,
lengths `128 192 256 512 1024`, and 1,024 examples per length.
Results go under `outputs/paper-eval/`; existing result directories are refused.

For a smaller implementation check:

```bash
EXAMPLES_PER_LENGTH=128 EVAL_ROOT=outputs/most_freq-ckpt-check bash most_freq/eval.sh
```

`SEEDS`, `CENTERS`, and `MODELS` select subsets. `CHECKPOINT` selects one file;
`OUTPUT_ROOT` selects another checkpoint sweep. `DRY_RUN=1` prints commands only.
For CPU execution, set `DEVICE=cpu PRECISION=fp32 ATTN_IMPLEMENTATION=eager`.
