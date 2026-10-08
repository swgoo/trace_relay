# TraceRelay

A minimal PyTorch decoder with **Right attention → delayed phase scan → residual
merge → Left attention**, followed by a gated FFN. Persistent phases are updated
only by delayed Right updates, with eviction-aligned defaults. It uses
Hugging Face `PreTrainedConfig`, `PreTrainedModel`, and `GenerationMixin`.

## Install

Use Python 3.10+ with a suitable CPU or CUDA PyTorch installation:

```bash
python -m pip install -e .
```

Dependencies are PyTorch >= 2.6 and Transformers >= 5.14.1, < 6.
The backbone supports direct forward passes, cached continuation, generation,
and save/load through the public Transformers API shown below.

## FlashAttention-2

Install a CUDA build of PyTorch first, then install the optional backend in a
supported Linux/CUDA environment:

```bash
python -m pip install 'flash-attn>=2.6.0,<3' --no-build-isolation
```

Select the backend with the standard Transformers argument:

```python
import torch
from trace_relay import TraceRelayConfig, TraceRelayForCausalLM

# New models default to RoPE, which both eager and FlashAttention support.
config = TraceRelayConfig(attn_implementation="flash_attention_2", dtype="bfloat16")
model = TraceRelayForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)

# Or load a checkpoint trained with RoPE:
model = TraceRelayForCausalLM.from_pretrained(
    "checkpoint", attn_implementation="flash_attention_2", dtype=torch.bfloat16,
).to("cuda")

# Switch an existing RoPE model after moving it to CUDA / FP16 or BF16:
model.set_attn_implementation("flash_attention_2")
```

The backend calls `flash_attn_varlen_func` with native local windows for both
Left and Right attention. It packs contiguous unpadded spans and adjusts the kernel's
bottom-right Q/K alignment to preserve each delayed Right query's original center.
For an alignment shift `s`, the equivalent kernel window is `(left+s, right-s)`.
Rows with different shifts are grouped into separate calls. K/V are **not
expanded once per query**, and no dummy queries are needed during decoding.
RoPE always uses the original physical token positions.

Requirements: a FlashAttention-2-supported CUDA GPU, FP16/BF16 Q/K/V (model dtype
or CUDA autocast), and even per-head widths no larger than 256. Dropout follows
the model's train/eval mode. Full forward, backward, cached continuation, and
greedy/sampling generation use the same local-window semantics. Left and right
padding, including fully padded rows, are supported. Internal mask holes require
`attn_implementation="eager"` because compressing them would change physical
time and the model's windows. Unsupported devices, dtypes, position encodings,
and attention backends raise errors; there is no silent eager fallback.

**Checkpoint compatibility:** this ordering changes Right's query projection from
trace width to hidden size, introduces `left_input_norm`, and changes cache
semantics. New HF configs record `layer_order="right_scan_left"`; loading an
older config without that marker is rejected. Pre-reorder checkpoints require
their original implementation. Use revision
`df2099f013a32e55ac772a85e0e90bdd92318785` for the delayed-update baseline. No weight
conversion is implied, even when hidden size equals trace width.

New configs use RoPE. The reordered architecture also supports learned relative
bias with the eager backend. A new-format config missing `position_encoding`
retains the legacy relative-bias default; this does not migrate old architectures.

See the upstream [FlashAttention installation and API](https://github.com/Dao-AILab/flash-attention#installation-and-features)
for compatible hardware/builds. This release implements `eager` and
`flash_attention_2`; FA3/FA4, generic SDPA, and Hub kernel substitution are not
enabled.

## Mechanism

The ordering uses delayed Right updates, with no immediate phase update.
Let `x_t` be raw lower-layer hidden, `r_t = input_norm(x_t)`,
`D` the trace width, `L` the Left window, `R` the Right window, and `W` the stride.
All attention windows include their center; `L` and `R` count neighbors only.

1. **Right writes from raw input:**
   `u_i = Right(Q=r_i, K/V=r[i:i+R])` and
   `delta_i = scale * tanh(right_phase(u_i))`.
   This path has no dependency on Left attention or this layer's phase state.
2. **Delayed delivery and carry:**

   ```text
   phi_t = wrap(phi_(t-W) + delta_(t-W))
   valid_t = valid_(t-W) OR valid_raw_center_(t-W)
   phi_t = 0 and valid_t = false for t < 0
   wrap(a) = a mod (2*pi)
   ```

   Negative or masked centers deliver zero increments. Right queries are evaluated
   directly at arrival time `t`, centered at `i=t-W`. Since **W > R**, their entire
   window ends at `t-W+R < t`. No future output or pending Right update is flushed early.
   The two scans use vectorized prefix sums along W temporal residue classes.
3. **Expose persistent memory before Left:**
   `z_t = valid_t * [cos(phi_t), sin(phi_t)]`,
   `m_t = phase_output(z_t)`, and **`memory_t = x_t + dropout(m_t)`**.
   The projection has no bias, so invalid memory contributes exactly zero, with
   no cosine-at-zero artifact. Raw Right output is never directly residual-added.
4. **Left retrieves from the merged representation:**
   `v_t = left_input_norm(memory_t)` and
   `a_t = Left(Q=v_t, K/V=v[t-L:t])`.
   Q, K and V all include the recurrent readout. The modules are
   `left_attention` and `left_output`, with the configured `left_window`.
5. **Attention and FFN residuals:**
   `y_t = memory_t + dropout(left_output(a_t))`, followed by
   `out_t = y_t + dropout(down_proj(silu(gate_proj(post_norm(y_t))) *
   up_proj(post_norm(y_t))))`. The output is masked at padded positions.

With the defaults **W=L+1 and R=L**, a raw token's first memory arrives immediately
after it leaves the direct Left window. A complete Left window has W positions and
contains exactly one latest representative of each temporal lane. Lanes are not
learned semantic addresses; Left attention selects by content over their combined
local and recurrent representations. During warm-up, `memory_t=x_t` and the layer
acts as ordinary local attention with a FFN.

Right and Left execute serially relative to each other, while positions within
each attention remain parallel. No Python token loop or nonlinear same-layer
recurrent transition is introduced. Phase accumulation/modulo uses FP64 on
CPU/CUDA (FP32 on MPS), and stores FP32 phases. Sin/cos is continuous across wrap.
A raw token can affect the ordinary local path immediately when observed; its
**Right-derived recurrent contribution** cannot affect any output before delivery.

See [flow.md](flow.md) for the flowchart and cache paths.

## Per-layer configuration

Each of the following accepts an integer shared by all layers or a list with
exactly `num_hidden_layers` entries. Lists are normalized and serialized in
`config.json`.
The public window keys are `left_window` and `right_window`. Previous window
keys and parameter names are not supported; checkpoints must use the current names.

| Field | Meaning | Default |
|---|---|---|
| `trace_width` | Attention projection width and number of phase coordinates | 64 |
| `left_window` | Previous neighbors of memory-augmented Left attention | 32 |
| `right_window` | Raw right neighbors around the Right-attention center | `None`: copy Left window per layer |
| `relay_stride` | Future shift and recurrence interval W | `None`: Left window + 1 per layer |

`trace_width` must be divisible by `num_attention_heads`. The residual stream
uses a shared `hidden_size`; trace widths may differ between layers.
For eviction alignment, use `relay_stride = left_window + 1` and
`right_window <= left_window`; the defaults use equality for the
two windows. Explicit settings need only satisfy `relay_stride >
right_window`, so other causal combinations remain available.
`phase_update_scale`
defaults to pi. RoPE requires even `trace_width / num_attention_heads`. There is
no absolute position embedding or maximum position table; physical token
positions determine the relative attention geometry.

## Causal LM training and forward

```python
import torch
from trace_relay import TraceRelayConfig, TraceRelayForCausalLM

config = TraceRelayConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    trace_width=[16, 32, 64],
    left_window=[7, 15, 31],
    right_window=[7, 15, 31],
    relay_stride=[8, 16, 32],
)
model = TraceRelayForCausalLM(config)
input_ids = torch.randint(3, config.vocab_size, (2, 64))
output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
assert output.logits.shape == (2, 64, 256)
output.loss.backward()
```

`labels` are shifted internally for next-token cross-entropy. Use `-100` for
ignored targets. A 2D zero/one `attention_mask` excludes padding from attention,
Left attention, output, and loss. Loss requires both the predicting and target positions
to be unpadded. A sequence with no valid next-token targets returns a finite zero
loss. Standard `inputs_embeds`, `output_hidden_states`, `return_dict=False`,
embedding accessors, tied embeddings, and `logits_to_keep=N` at inference are
supported. `position_ids` are accepted for API compatibility, but do not override
the relative physical-time positions.

`TraceRelayModel` exposes the backbone as `last_hidden_state` with shape
`[batch, sequence, hidden_size]`. Outputs use the usual Transformers model-output
classes. No tokenizer or pretrained causal-LM checkpoint is bundled; configure vocabulary
size and special-token IDs to match your tokenizer.

## Cached continuation and generation

```python
model.eval()
with torch.no_grad():
    first = model(input_ids[:, :19], use_cache=True)
    second = model(
        input_ids[:, 19:],
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    continued_logits = torch.cat([first.logits, second.logits], dim=1)

    generated = model.generate(
        input_ids[:1, :8], max_new_tokens=12, do_sample=False,
        eos_token_id=None,
    )
```

`TraceRelayCache` stores these tensors per layer, with `K=max(L,W)`:

| Tensor | Meaning | Maximum shape per batch item |
|---|---|---|
| `raw_hidden` | `input_norm(lower_hidden)`, used only by Right | `[K,H]` |
| `memory_hidden` | `left_input_norm(lower_hidden + dropout(m))`, used by Left Q/K/V | `[K,H]` |
| `mask` | Shared mask for both histories, with identical physical positions | `[K]` |
| `phases` | Chronological tail of wrapped persistent phases | `[W,D]` |
| `phase_valid` | Whether a Right update has ever initialized each lane | `[W]` |

The histories store normalized attention inputs, not projected K/V or final layer
outputs. Both use the same bounded K to keep a single shared mask and source
start. Clones avoid retaining full prefill
allocations under `no_grad` while preserving the training graph. Storage is
constant in sequence length; chunk boundaries need not align with any stride.

There is no manual commit or flush. Masks may cover either the new chunk or the
entire prefix; cached prefix masks cannot be edited retroactively.

The cache is mutable and belongs to one sequence batch and one model's weights.
Create a fresh cache for a new sequence, after changing weights, or after moving
the model to a different dtype/device. Use `cache.detach()` between chunks for
truncated backpropagation; without it, the autograd graph spans chunks. Resetting
the cache clears all sequence state. For training with separate labeled chunks,
loss covers adjacent tokens **within** each call; include the boundary target in
your training loop if you need loss across chunk boundaries.

Padding occupies positions on the physical token clock. Contiguous left/right
padding is supported; internal mask holes are skipped as content but still count
toward windows and stride. Use left padding for batched `generate()`. Greedy
decoding and sampling work with or without caching; a supplied generation cache
requires the full prefix plus at least one unprocessed token as input. The
returned generation cache follows HF convention: the last sampled token may not
have been processed yet. Omit `cache_implementation`; generic KV caches, beam
search, speculative rollback, and static/compiled caches are not supported.

## Save, load, and AutoClass

```python
from transformers import AutoConfig, AutoModelForCausalLM
import trace_relay  # registers local AutoConfig / AutoModel / AutoModelForCausalLM

model.save_pretrained("checkpoint")
restored = trace_relay.TraceRelayForCausalLM.from_pretrained("checkpoint")
auto_model = AutoModelForCausalLM.from_pretrained("checkpoint")

# save_pretrained also copies the custom model/config source. In another
# environment without this package, load a checkpoint whose code you trust:
portable = AutoModelForCausalLM.from_pretrained(
    "checkpoint", trust_remote_code=True,
)
```

The AutoClass integration follows the official
[custom-model API](https://huggingface.co/docs/transformers/custom_models).
The source files also support direct imports from the repository using
`configuration_trace_relay` and `modeling_trace_relay`.

## Reproduce paper experiments

Each task directory contains executable `train.sh` and `eval.sh` launchers matching
the completed protocols in `paper/table/protocol.tex` and `paper/data/*.json`.
All use three seeds (42/43/44), centers 64/32/16, CUDA BF16, FlashAttention-2,
20,000-update caps, and stopping after two consecutive mean ID scores reach 99%.
Equal Repeats runs 18 inheritance/no-inheritance comparisons; Dyck and Most-Freq
each run nine experiments with inheritance. Most-Freq uses five symbols,
separated counts with gap four, and source lengths 128--256.

```bash
bash equal_repeats/train.sh
bash dyck/train.sh
bash most_freq/train.sh

bash equal_repeats/eval.sh
bash dyck/eval.sh
bash most_freq/eval.sh
```

Training runs sequentially and writes each selected checkpoint's final evaluation.
Fresh outputs default to `outputs/paper-reproduction/{equal-repeats,dyck,most-freq}`.
Each task's `ckpt/` bundles the paper-selected `best.pt` files with current
`left`/`right` parameter and config names: 18 Equal Repeats, nine Dyck, and nine
Most-Freq checkpoints. FP32 tensor values are preserved; optimizer and RNG state
are omitted. W&B account identifiers are removed, and local output paths are
repository-relative. These exports are for evaluation, not training resumption.
Each `ckpt/manifest.json` records source/export hashes, seeds, widths, selected
steps, parameter counts, and file sizes.

By default, `eval.sh` reads these bundled checkpoints using the paper's per-task
lengths, sample counts, and final seed 20261006, writing separate results under
`outputs/paper-eval/{equal-repeats,dyck,most-freq}`. Set `OUTPUT_ROOT` to evaluate
a newly trained sweep instead.
Existing run/log/evaluation directories are never overwritten.

`DRY_RUN=1` prints commands without executing jobs. `SEEDS`, `CENTERS`, and
`MODELS` accept space-separated subsets; `SEED`, `CENTER_WIDTH`, and
`MODEL_FAMILY` remain single-setting aliases. `OUTPUT_ROOT` selects a different
sweep directory for either launcher. For example:

```bash
DRY_RUN=1 bash most_freq/train.sh
SEED=42 CENTER_WIDTH=16 MODEL_FAMILY=trace_relay bash equal_repeats/train.sh
OUTPUT_ROOT=outputs/paper-reproduction/equal-repeats \
  EVAL_ROOT=outputs/equal-repeats-retrained-eval bash equal_repeats/eval.sh
CHECKPOINT=dyck/ckpt/dyck-trace_relay-center64-seed42/best.pt \
  EVAL_ROOT=outputs/dyck-single-recheck bash dyck/eval.sh
```

W&B defaults to each task's original project; set `WANDB_MODE=disabled` for local
logging only, or `WANDB_ENTITY` to choose an account. `PYTHON` selects an interpreter.
Extra trainer/evaluator options can be passed as CLI arguments; such overrides
change the reproduced protocol and are recorded in the run metadata.

## Equal Repeats: compact run-length relationships

`equal_repeats/task.py` follows DeepMind's **historical Equal Repeats** definition.
The reference is [`tasks/ndcf/equal_repeats.py` at commit `46bcae0`](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/blob/46bcae06945726f34e8e4560ab698fa2f2d2c889/tasks/ndcf/equal_repeats.py).
It retains this task, which was removed from upstream, rather than substituting a
task from the current upstream list. For input `0^l 1^m 0^n`, it predicts
`l=m → 0`, `m=n → 1`, or `l=n → 2`. When all three lengths are equal, the label
is always class 0. Examples:
`01000 → 0`, `0001100 → 1`, `011110 → 2`, `000111000 → 0`.

Binary symbols 0/1 are **true two-dimensional one-hot vectors**, `[1,0]` / `[0,1]`,
passed through a bias-free `Linear(2,H)` projection into the existing TraceRelay
backbone. The hidden state at the final valid token is passed to `Linear(H,3)`,
with one cross-entropy loss per sequence. Run lengths, boundaries, class identity,
counters, and intermediate answers are not supplied as model inputs. Positional
encoding and the TraceRelay recurrence use the existing implementation.
`trace_relay`, `no_carry`, and `swa` use the shared configuration helper in
`experiment_utils.py`; cached chunk evaluation and right padding are also supported.

**The original edge cases are preserved.** The sampled `a` ranges from
`1..floor(T/2)`, inclusive, so the endpoint for even lengths includes a run of
length zero. Sequences such as `0011 → 0`, `1100 → 1`, and `0000 → 2` may therefore
have fewer than three nonempty runs. Every generated sequence is checked against
the reference construction, including this endpoint. Nominal group sizes are
`floor(B/3), floor(B/3), B-2*floor(B/3)`, after which equal-all examples are
relabeled as class 0. The final class counts are not artificially rebalanced.
When `B % 3 == 2`, nominal class 2 has two more examples than the other groups;
at `T=3`, all labels are 0. Evaluation also records actual class counts and a
majority baseline. Uniform random chance is **1/3**. Classes 0/1 are paired
reversals, as in the original; when group sizes match, all three groups use the
same sampled `a` values. Bit-for-bit equivalence between PyTorch and JAX RNGs
is not required.

Each training step samples one integer `T ~ Uniform[train_min, train_max]` and
generates an online batch whose examples all have that length. There is no
curriculum or dataset epoch. Defaults are H=64, FFN=256, heads=4, 3 layers,
trace widths `[64,center,64]`, Left `[15,15,15]`, Right `[7,7,7]`, stride
`[8,8,8]`, and no skips. Training lengths are 32..256, and ID validation lengths
are 32/64/128/256. Checkpoints are selected by mean validation accuracy, with
lower mean CE as the tie-break. OOD lengths are not used for checkpoint selection.
Final evaluation uses one fixed best checkpoint to evaluate each of
32/64/128/256/512/1024/2048/4096 separately.

**Early stopping:** the launcher and sweep stop training after **two consecutive
ID validation checks reach at least 99% mean accuracy**, then evaluate the fixed
best checkpoint. Validation runs every 250 steps by default; a score below 99%
resets the streak to zero. The criterion uses the mean across 32/64/128/256,
rather than the minimum accuracy across lengths. OOD results do not affect stopping.
Control this with `--early-stop-accuracy 0.99 --early-stop-passes 2`, or use
`--disable-early-stop` for a fixed-step experiment. Early stopping is disabled
by default when calling the Python trainer directly. If the threshold is not
met, training continues for up to 20,000 steps. Results record `stop_reason`,
`early_stopped`, and the streak. Checkpoints also store the streak; resuming
with the same stopping conditions restores it, while changing them resets it.

**Current default geometry:** Left `[15,15,15]`, Right `[7,7,7]`, stride `[8,8,8]`.
With stride 8, the first delayed delivery occurs at the ninth token, and the
second accumulation onto a previously valid phase begins at the seventeenth
token. Each lane accumulates three Right updates in a 32-token input and 31 in
a 256-token input. Local attention helps learning on short inputs, while longer
training inputs exercise repeated recurrent carry. This exposure is recorded
in the protocol's `carry_exposure` field.

The earlier wide-window experiment used Left `[127,512,127]`, Right 127, and
stride 128. Its second accumulation began at the 257th token, so training up to
256 tokens did not exercise multi-hop carry. The wide-window and current
small-window results are separate experiments with different geometries.

```bash
# Single run (CUDA BF16, FlashAttention-2, 20k steps, W&B online).
tmux new-session -d -s equal-repeats 'SEED=42 CENTER_WIDTH=64 MODEL_FAMILY=trace_relay bash equal_repeats/train.sh'
tmux attach -t equal-repeats

# 18 runs = centers 64/32/16 × carry/no-carry × seeds 42/43/44.
# Prints commands by default; --execute runs them sequentially.
.venv/bin/python -m equal_repeats.sweep
tmux new-session -d -s equal-repeats-sweep \
  '.venv/bin/python -u -m equal_repeats.sweep --execute'

# Optionally wait for an existing training PID to exit before starting.
.venv/bin/python -m equal_repeats.sweep --execute --wait-pid 12345

# ID/OOD evaluation of one checkpoint: seed+length data are independent of batch partitioning.
.venv/bin/python -m equal_repeats.evaluate \
  --checkpoint outputs/equal-repeats-trace_relay-center64-seed42/best.pt \
  --output outputs/equal-repeats-trace_relay-center64-seed42/recheck \
  --lengths 32 64 128 256 512 1024 2048 4096 \
  --examples-per-length 10000 --batch-size 128 --seed 20261006 --device cuda

# Optional matplotlib plots; training/evaluation do not import matplotlib.
.venv/bin/python -m pip install matplotlib
.venv/bin/python -m equal_repeats.plot \
  outputs/equal-repeats-*-center*-seed*/final_eval/results.json \
  --output outputs/equal-repeats-figures
```

The launcher's W&B project is `TraceRelayEqualRepeats`. Configure experiments
with `CENTER_WIDTH`, `SEED`, `MODEL_FAMILY`, `OUTPUT_ROOT`, `STEPS`, `WANDB_MODE`,
`WANDB_PROJECT`, and additional CLI options. The sweep also supports a local-only
baseline through `--models swa`. The 18-run sweep places carry/no-carry runs with
the same width and seed next to each other and refuses to overwrite outputs or logs.

Each experiment saves `protocol.json`, `metrics.jsonl`, `best.pt`, `last.pt`,
and `results.json`. The checkpoint format is **`trace-relay-equal-repeats-v1`**.
It stores the model/task/version, length protocol, seed, commit and source hashes,
step and best validation metrics, plus optimizer, online generator, and RNG state.
`--resume last.pt --steps 80000` restores weights and training state into a fresh
output directory and continues to a total of 80k steps. On interruption, the
last completed validation checkpoint is preserved. The validation seed is
20261007 and the final evaluation seed is 20261006, separate from training seeds.

Final evaluation saves per-length accuracy, CE, count, chance, class counts,
confusion matrices, and generated-example hashes in `final_eval/results.json`
and `metrics.csv`. It does not combine lengths into a global test accuracy.
Plotting produces accuracy-versus-length PNG/PDF/SVG files, `summary.csv` with
seed means and sample SD, and `carry_gap.csv` with corresponding plots of the
**carry − no-carry** difference for matched training/evaluation seeds. Gap
calculations verify matching evaluation examples and memory geometry, without
enforcing an expected width ordering. Negative or non-monotonic values are
reported as observed.

Fixed-length training is also supported with
`--train-min-length 256 --train-max-length 256 --validation-lengths 256`.
Run a small implementation smoke test as follows:

```bash
.venv/bin/python -m equal_repeats.train --output outputs/equal-repeats-smoke \
  --device cpu --hidden-size 32 --intermediate-size 64 --heads 2 \
  --trace-widths 32 16 32 --left-windows 15 31 15 --right-windows 7 --relay-strides 8 \
  --train-min-length 16 --train-max-length 64 --validation-lengths 16 32 64 \
  --steps 500 --batch-size 32 --validation-examples 256 \
  --eval-lengths 16 32 64 128 --eval-examples 512 --final-evaluation
```

This task tests whether a long binary input can be compressed into a small
sufficient state, assessing retention of task-relevant state rather than exact
token-level recall.

## Dyck-(k,m): causal closing-bracket prediction

`dyck/task.py`, `dyck/model.py`, `dyck/train.py`, and `dyck/evaluate.py` test
**ordered, stack-like working memory** using the existing TraceRelay core.
Inputs are valid bracket sequences with `k` bracket types and nesting depth at
most `m`. The default difficulty is **k=8, m=10**, with random chance close-type
accuracy of **1/k=12.5%**.

The background references are the bounded hierarchical language in
[Hewitt et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.156/), their
[RNN construction](https://github.com/john-hewitt/dyckkm-constructions), and
[Yao et al., ACL 2021 / the Princeton Dyck benchmark](https://github.com/princeton-nlp/dyck-transformer).
Hewitt's finite-precision theory motivates the `m log k` scaling of required
memory, but TraceRelay center width is not interpreted as a literal number of
memory bits. This implementation is a **separate diagnostic that predicts only
the next closing bracket's type**. It should not be compared as a reproduction
of existing full-language-model scores or PDFA corpus distributions.

The input vocabulary contains exactly `2k` symbols: `0..k-1` are openings and
`k..2k-1` are closings. True one-hot inputs `[B,T,2k]` pass through a bias-free
linear projection, and each token's backbone hidden state is classified by
`Linear(H,k)`. **Logits at position t predict the close type at t+1.** Opening
targets and padding are excluded with `-100`; CE is averaged only over valid
close targets. Target shifting and causality are tested to ensure predictions
cannot read the closing bracket being predicted. Stack contents, depth,
matching distance, and the next open/close event are not supplied as model inputs.

Sampler semantics are **`dyck-km-fixed-length-close-v1`**. Each step samples one
even length uniformly, then chooses open/close events in proportion to valid
completion counts from float64 row-normalized DP. The stack starts and ends
empty, and every prefix depth lies in `0..m`. Opening types are sampled uniformly;
a close pops the stack top. Samples are not required to reach depth m.
DP costs `O(T*m)`, and generation uses an `O(B*T)` stack sampler that processes
the batch in parallel. The core model's sequence-parallel prefill and recurrence
are unchanged. A separate strict parser rereads the token sequence to verify
labels, distances, and depths independently of the generator.

The current geometry is H=64, FFN=256, heads=4, 3 layers, widths `[64,center,64]`,
**Left `[15,15,15]`, Right `[7,7,7]`, stride `[8,8,8]`, and no skips**.
Training uses even lengths **32..256**, ID validation uses **32/64/128/256**,
and OOD evaluation uses **512/1024/2048/4096**. The same `k,m` are retained
throughout length extrapolation. Checkpoints are selected by mean ID close
accuracy across lengths, with lower mean close CE as the tie-break. The launcher
and sweep stop after **two consecutive checks reach at least 99%**, with
validation every 250 steps by default and a maximum of 20,000 steps. Early
stopping must be specified explicitly when calling the Python trainer directly.
OOD results are not used for training, checkpoint selection, early stopping,
or learning-rate control.

`trace_relay`, `no_carry`, and `swa` share the existing model-family configuration
helper. **The current run policy trains only the carry variant.** The default
sweep has **9 runs**: centers 64/32/16 × seeds 42/43/44. `no_carry`/`swa` are
supported for implementation and forward/backward/cache verification.
`no_carry` retains delayed Right updates and phase readout, removing only
recurrent inheritance. **The full benchmark sweep and no-carry training were
not run during implementation.** Quantifying the benefit of carry on this task
requires a future ablation comparison on the same task.

Evaluation generates data from `(semantics, seed, T, k, m)` and stores example
fingerprints independent of inference batch partitioning. Evaluation plans use
compact token/metadata tensors for one length at a time, with one-hot tensors
materialized per batch. Memory use is approximately `O(N*T + B*T*k)`.
Training, validation, and final evaluation seeds must differ; defaults are
42/20261007/20261006. `--eval-chunk-size` and the evaluator's `--chunk-size`
use `TraceRelayCache`. Chunk logits are concatenated in token order before
applying the target shift once, so targets are preserved across chunk boundaries.

Each length reports close accuracy, close CE, close count, sequence count, and
chance accuracy. Matching-opener distance `close_index - open_index` uses bins
`1..8, 9..16, 17..32, 33..64, 65..128, 129..256, 257..512, 513..1024, 1025..2048, >2048`.
Close accuracy is also recorded by pre-close depth `1..m` and by the sequence's
maximum depth. Empty bins remain **count=0, accuracy/CE=null**. No global score
combines lengths, and no hard bound is asserted that no-carry must fail at a
particular distance.

Outputs are `protocol.json`, `metrics.jsonl`, `best.pt`, `last.pt`, and
`results.json`. Final evaluation also saves `metrics.csv`, `distance_metrics.csv`,
`depth_metrics.csv`, and `max_depth_metrics.csv`. The checkpoint format is
**`trace-relay-dyck-v1`**, including optimizer, online generator, and RNG state,
physical-token/close-target counts, best step, validation metrics, and the
early-stop streak. `--resume` restores the full state into a fresh output
directory. With identical settings, it reproduces the same updates as
uninterrupted training. The protocol stores the git commit, source hashes,
k/m, vocabulary, geometry, seed and length settings, reference URLs, and a
description identifying this as a separate diagnostic. The default W&B project
is **`trace relay dyck`**.

```bash
# Implementation smoke: carry only, CPU, k=4/m=4. Not benchmark evidence.
.venv/bin/python -m dyck.train --output outputs/dyck-smoke \
  --device cpu --bracket-types 4 --max-depth 4 \
  --hidden-size 32 --intermediate-size 64 --heads 2 --trace-widths 32 16 32 \
  --left-windows 7 15 7 --right-windows 3 --relay-strides 4 \
  --train-min-length 16 --train-max-length 64 --validation-lengths 16 32 64 \
  --steps 500 --batch-size 32 --validation-examples 64 --eval-every 100 \
  --eval-lengths 16 32 64 128 --eval-examples 128 --final-evaluation

# Single inheritance run, CUDA BF16, FlashAttention-2, W&B online.
tmux new-session -d -s dyck 'SEED=42 CENTER_WIDTH=64 bash dyck/train.sh'
tmux attach -t dyck

# Carry-only 9-run sweep: dry-run; no training starts.
.venv/bin/python -m dyck.sweep --output-root outputs/dyck-carry-sweep
# Explicit launch when requested. Existing output/logs are never overwritten.
tmux new-session -d -s dyck-carry-sweep \
  '.venv/bin/python -u -m dyck.sweep --output-root outputs/dyck-carry-sweep --execute'

# Resume to a larger total budget, preserving all training state.
.venv/bin/python -m dyck.train --output outputs/dyck-resumed \
  --resume outputs/dyck-trace_relay-center64-seed42/last.pt \
  --device cuda --precision bf16 --steps 40000 --early-stop-accuracy .99 --early-stop-passes 2

# ID/OOD evaluation of one fixed best checkpoint; 2,000 sequences per length.
.venv/bin/python -m dyck.evaluate --checkpoint outputs/dyck-trace_relay-center64-seed42/best.pt \
  --output outputs/dyck-trace_relay-center64-seed42/recheck \
  --lengths 32 64 128 256 512 1024 2048 4096 --examples-per-length 2000 \
  --batch-size 64 --seed 20261006 --device cuda --precision bf16

# Optional matplotlib; never imported by training/evaluation.
MPLCONFIGDIR=/tmp/trace-relay-mpl-config .venv/bin/python -m dyck.plot \
  outputs/dyck-carry-sweep/dyck-*/final_eval/results.json --output outputs/dyck-figures
```

Plotting saves mean ± sample SD by length, matching distance, and pre-close
depth. Distance/depth plots are drawn separately for each T to keep length
distributions separate. It produces aggregate CSV and PNG/PDF/SVG files without
enforcing monotonicity or a width ordering on observations. Carry-gap plots
are generated only when no-carry results have the **same center, training seed,
evaluation seed, k/m, geometry, and fingerprint**. Without a baseline, the gap
is neither estimated nor filled with zero.

## Files and scope

| File | Purpose |
|---|---|
| `configuration_trace_relay.py` | Validated per-layer configuration |
| `modeling_trace_relay.py` | Local attention, stride scan, bounded cache, backbone, causal LM |
| `__init__.py` | Package exports and local AutoClass registration |
| `experiment_utils.py` | Shared task configuration, precision, RNG, JSON, resource counts, and optional W&B tracking |
| `equal_repeats/` | Binary-sequence task, classifier, training, evaluation, sweep, plotting, and launcher |
| `dyck/` | Bracket task, predictor, training, evaluation, sweep, plotting, and launcher |
| `most_freq/` | Independent Most-Freq task, model, training, evaluation, sweeps, and plots |
| `paper/` | Manuscript, frozen evidence, figures, tables, and reproduction scripts |

The eager backend gathers neighborhood tensors, with work/storage proportional
to sequence length times window size. The FlashAttention-2 backend uses fused
local attention with compact Q/K/V buffers; Python span packing adds overhead.
The phase scan remains a PyTorch implementation. Attention weight outputs,
gradient checkpointing, distributed model sharding, and packed
independent sequences are not implemented. Run independent documents separately
or reset their caches. Small example runs establish execution and causal consistency,
not trained quality, convergence, throughput, or long-context numerical accuracy.
