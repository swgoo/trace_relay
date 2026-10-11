# BabyLM 2026 Strict + BLiMP

This task directory provides Strict training and BLiMP evaluation for the
TraceRelay causal language model. It uses the existing core recurrence.

## Model

The full architecture is defined in `configs/unet16_102m.json`.

- Parameters: **102,131,392 (102.13M)**, with independent input embeddings and LM head.
- Hidden sizes: `[736, 368, 192, 368, 736]`.
- Stage depths: `[3, 3, 4, 3, 3]`, totaling 16 layers with 8 heads per layer and an FFN ratio of 4.
- Same-position stage skips: `0→4`, `1→3` (layers `2→13`, `5→10`).
- Recurrent inheritance enabled, RoPE, zero dropout.

| Layers (zero-indexed) | Hidden | Left | Right | Relay stride |
|---|---:|---:|---:|---:|
| 0 / 15 | 736 | 16 | 16 | 17 |
| 1 / 14 | 736 | 32 | 32 | 33 |
| 2 / 13 | 736 | 64 | 64 | 65 |
| 3 / 12 | 368 | 128 | 128 | 129 |
| 4 / 11 | 368 | 256 | 256 | 257 |
| 5 / 10 | 368 | 512 | 256 | 257 |
| 6 / 9 | 192 | 1024 | 256 | 257 |
| 7 / 8 | 192 | 2048 | 256 | 257 |

Windows count neighboring positions, excluding the current position. Right
updates are delivered only after `W=R+1` positions, preserving causality.
The two center layers can attend to the entire preceding context within a
2048-token training input through Left attention.

## Official byte tokenizer

The implementation uses the official
[Transformers ByT5Tokenizer](https://huggingface.co/docs/transformers/model_doc/byt5):
`ByT5Tokenizer(extra_ids=0, bos_token="</s>")`.
Tokenization requires no tokenizer training or pretrained model weights.

- `len(tokenizer) = 259`: PAD=0, EOS=1, UNK=2, UTF-8 byte IDs=3..258.
- BOS reuses EOS=1. No ByT5 sentinel extra IDs are added.
- `tokenizer.vocab_size` is 256, so the model must use **`len(tokenizer)`**.
- Literal strings such as `</s>` are encoded as bytes using `split_special_tokens=True`.
- The prepared stream is `[EOS] line1-bytes [EOS] line2-bytes [EOS] ...`.
  Empty lines are omitted, and each nonempty source line is separated by EOS.
- A sequence length of 2048 means **2048 byte tokens**, not 2048 words.

## Data preparation

```bash
python -m pip install -e '.[babylm,test]'
bash babylm/prepare.sh --output data/babylm2026-strict100m
```

The following official repositories are downloaded at pinned revisions:

- [2026 Strict 100M training data](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict):
  `9e57baaaa91ac3c638746be14d1d5fa6c789f4cf` (the default corpus).
- [2026 Strict evaluation data](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Evals):
  `8d52da9424a9ff30b9e8266c4f751aba9c504233`; only the BLiMP
  `full_eval/blimp_filtered` and `fast_eval/blimp_fast` directories are downloaded.

The training repository has no separate LM validation split. By default, 1% of
source lines are held out using SHA256. Identical lines always enter the same
split, even across source files. The resulting training corpus is therefore
slightly smaller than the official 100M-word corpus. BLiMP does not influence
this split. `manifest.json` records source and prepared file hashes, word/token
counts, the split seed, and preprocessing settings.

Local data, including small offline smoke inputs, can also be supplied explicitly:

```bash
bash babylm/prepare.sh --output data/babylm-local \
  --train-files /path/to/train/*.txt --validation-files /path/to/dev/*.txt \
  --skip-blimp
```

When explicit validation files are provided, matching source lines are excluded
from training. Existing prepared outputs are never overwritten. The default
training budget is 100,000,000 whitespace-delimited words and can be reduced with
`--max-train-words`. Strict is the supported and default track. At the word limit,
the final line is cut at a word boundary and subsequent training lines are omitted.

To download BLiMP alone:

```bash
.venv/bin/python -c 'from babylm.prepare import download_blimp; download_blimp("data/babylm2026-strict100m/blimp")'
```

To prepare the Strict 100M byte corpus without downloading BLiMP:

```bash
bash babylm/prepare.sh --track strict --output data/babylm2026-strict100m --skip-blimp
```

## Strict training configuration

These settings were used for three epochs of training on Strict.
They are distinct from the CLI defaults.

| Setting | Value |
|---|---:|
| Model | 16-layer U-Net, 102.13M parameters, independent embeddings / LM head |
| Tokenizer | ByT5 byte, vocabulary 259 |
| Training / holdout words | 99,034,161 / 965,839 |
| Training duration | 3 epochs |
| Sequence length | 2048 byte tokens |
| Microbatch | 11 sequences |
| Effective batch | 16 sequences, up to 32,768 target tokens/update |
| Microbatch composition | 11 + 5 sequences, gradient accumulation 2 |
| AdamW learning rate | Maximum `1.5e-4`, minimum `1.5e-5` |
| Adam betas / weight decay | `(0.9, 0.95)` / `0.1` |
| Warmup | 100 updates |
| Schedule | Cosine cycle, approximately one epoch per half-cycle |
| Gradient clipping | 1.0 |
| Validation / checkpoint | Every 500 updates |
| Validation coverage | Entire holdout |
| Training logs | Every 10 updates |
| Seed | 42 |
| Precision / backend | BF16 autocast, FP32 master weights + AdamW / FlashAttention-2 |
| W&B project | `TraceRelayBabyLM` |

The LR smoothly alternates between falling and rising within its fixed bounds.

Example command for three epochs with these settings:

```bash
bash babylm/train.sh \
  --data data/babylm2026-strict100m \
  --config babylm/configs/unet16_102m.json \
  --output outputs/babylm-strict-3epoch-seed42 \
  --device cuda --precision bfloat16 --attn-implementation flash_attention_2 \
  --epochs 3 --sequence-length 2048 \
  --batch-size 11 --gradient-accumulation 2 --effective-batch-size 16 \
  --eval-batch-size 2 --validation-max-blocks 0 \
  --learning-rate 0.00015 --min-lr-ratio 0.1 --warmup-steps 100 \
  --lr-schedule cosine_cycle --cosine-half-cycle-epochs 1.001157 \
  --weight-decay 0.1 --grad-clip 1.0 \
  --eval-every 500 --save-every 500 --log-every 10 \
  --seed 42 --wandb-mode online --wandb-project TraceRelayBabyLM
```

The training stream is divided into fixed-length blocks and shuffled each epoch.
Recurrent caches are not passed between blocks. The final short block uses a
padding mask and `-100` labels; accumulated losses are weighted by the number of
valid target tokens. Word markers are used only for exposure accounting and are
never supplied to the model.

`best/` contains the model with the lowest LM cross entropy over the full holdout.
`last/` also stores the tokenizer, optimizer/scaler, RNG state, and shuffle position.
BLiMP is not used for training or checkpoint selection.

The evaluated model was trained for three epochs on Strict.
Its validation loss was **1.0132**, with byte-token perplexity
**2.7543**. Its training protocol and weights are stored in the run's
`protocol.json` and `best/` directory.

## BLiMP evaluation

```bash
bash babylm/eval.sh \
  --checkpoint /path/to/strict-run/best \
  --data data/babylm2026-strict100m/blimp/evaluation_data/full_eval/blimp_filtered \
  --output outputs/babylm-strict-blimp \
  --device cuda --precision bfloat16 --attn-implementation flash_attention_2 \
  --batch-size 32 --seed 42
```

The final evaluation used GPU BF16 / FlashAttention-2 with a batch size of 32 pairs.
On BabyLM-filtered BLiMP (**67 tasks / 59,875 pairs**), macro accuracy was **65.60%**
and micro accuracy was **65.50%**. Since tasks contain different numbers of pairs,
the **macro average of task accuracies** is the primary score. Micro accuracy is
also saved, along with data file hashes and task/pair counts.

### Detailed results

The tables below report the complete evaluation of the best checkpoint after
three epochs of training on Strict, using evaluation seed 42, BF16, and
FlashAttention-2. All 67 tasks were evaluated; no per-task subsampling was used.

| Metric | Result |
|---|---:|
| Holdout LM cross entropy | 1.0132 |
| Holdout byte-token perplexity | 2.7543 |
| BLiMP task-macro accuracy | 65.60% |
| BLiMP pair-micro accuracy | 65.50% |
| BLiMP strict pair-micro accuracy (`good > bad`; ties counted as incorrect) | 65.49% |
| Correct / evaluated pairs | 39,219 / 59,875 |
| Tasks | 67 |
| Tied sentence scores | 7 |
| Chance reference | 50% |

Field and phenomenon accuracies below pool all pairs within each group. They are
pair-weighted accuracies, whereas the headline macro score averages the 67 task
accuracies equally. Counts reflect the BabyLM-filtered evaluation set.

#### By field

| Field | Accuracy | Correct / pairs | Ties |
|---|---:|---:|---:|
| morphology | 79.54% | 12,558 / 15,788 | 0 |
| semantics | 60.41% | 5,050 / 8,359 | 0 |
| syntax | 59.73% | 13,826 / 23,147 | 2 |
| syntax/semantics | 61.88% | 7,785 / 12,581 | 5 |

#### By linguistic phenomenon

| Phenomenon | Accuracy | Correct / pairs | Ties |
|---|---:|---:|---:|
| `anaphor_agreement` | 73.55% | 1,399 / 1,902 | 0 |
| `argument_structure` | 67.34% | 4,089 / 6,072 | 2 |
| `binding` | 66.45% | 4,326 / 6,510 | 5 |
| `control_raising` | 63.79% | 2,783 / 4,363 | 0 |
| `determiner_noun_agreement` | 88.66% | 6,023 / 6,793 | 0 |
| `ellipsis` | 65.64% | 1,070 / 1,630 | 0 |
| `filler_gap_dependency` | 65.32% | 4,036 / 6,179 | 0 |
| `irregular_forms` | 91.59% | 1,743 / 1,903 | 0 |
| `island_effects` | 43.26% | 3,222 / 7,448 | 0 |
| `npi_licensing` | 42.87% | 2,684 / 6,261 | 0 |
| `quantifiers` | 79.93% | 3,042 / 3,806 | 0 |
| `s-selection` | 77.50% | 1,409 / 1,818 | 0 |
| `subject_verb_agreement` | 65.38% | 3,393 / 5,190 | 0 |

<details>
<summary>All 67 task results</summary>

| Task | Accuracy | Correct / pairs | Ties |
|---|---:|---:|---:|
| `adjunct_island` | 60.78% | 564 / 928 | 0 |
| `anaphor_gender_agreement` | 63.54% | 617 / 971 | 0 |
| `anaphor_number_agreement` | 84.00% | 782 / 931 | 0 |
| `animate_subject_passive` | 68.38% | 612 / 895 | 0 |
| `animate_subject_trans` | 86.35% | 797 / 923 | 0 |
| `causative` | 52.44% | 429 / 818 | 0 |
| `complex_NP_island` | 41.84% | 354 / 846 | 0 |
| `coordinate_structure_constraint_complex_left_branch` | 15.78% | 143 / 906 | 0 |
| `coordinate_structure_constraint_object_extraction` | 51.63% | 490 / 949 | 0 |
| `determiner_noun_agreement_1` | 96.34% | 895 / 929 | 0 |
| `determiner_noun_agreement_2` | 95.17% | 886 / 931 | 0 |
| `determiner_noun_agreement_irregular_1` | 81.20% | 553 / 681 | 0 |
| `determiner_noun_agreement_irregular_2` | 89.15% | 731 / 820 | 0 |
| `determiner_noun_agreement_with_adj_2` | 87.99% | 828 / 941 | 0 |
| `determiner_noun_agreement_with_adj_irregular_1` | 79.53% | 571 / 718 | 0 |
| `determiner_noun_agreement_with_adj_irregular_2` | 84.76% | 712 / 840 | 0 |
| `determiner_noun_agreement_with_adjective_1` | 90.78% | 847 / 933 | 0 |
| `distractor_agreement_relational_noun` | 34.77% | 274 / 788 | 0 |
| `distractor_agreement_relative_clause` | 20.21% | 176 / 871 | 0 |
| `drop_argument` | 72.17% | 664 / 920 | 0 |
| `ellipsis_n_bar_1` | 47.51% | 381 / 802 | 0 |
| `ellipsis_n_bar_2` | 83.21% | 689 / 828 | 0 |
| `existential_there_object_raising` | 75.49% | 613 / 812 | 0 |
| `existential_there_quantifiers_1` | 96.02% | 893 / 930 | 0 |
| `existential_there_quantifiers_2` | 62.57% | 570 / 911 | 0 |
| `existential_there_subject_raising` | 74.03% | 684 / 924 | 0 |
| `expletive_it_object_raising` | 68.25% | 518 / 759 | 0 |
| `inchoative` | 47.95% | 410 / 855 | 0 |
| `intransitive` | 55.53% | 482 / 868 | 0 |
| `irregular_past_participle_adjectives` | 96.46% | 927 / 961 | 0 |
| `irregular_past_participle_verbs` | 86.62% | 816 / 942 | 0 |
| `irregular_plural_subject_verb_agreement_1` | 71.64% | 576 / 804 | 0 |
| `irregular_plural_subject_verb_agreement_2` | 85.54% | 763 / 892 | 0 |
| `left_branch_island_echo_question` | 34.42% | 326 / 947 | 0 |
| `left_branch_island_simple_question` | 33.54% | 319 / 951 | 0 |
| `matrix_question_npi_licensor_present` | 11.19% | 104 / 929 | 0 |
| `npi_present_1` | 30.58% | 278 / 909 | 0 |
| `npi_present_2` | 36.21% | 331 / 914 | 0 |
| `only_npi_licensor_present` | 43.65% | 385 / 882 | 0 |
| `only_npi_scope` | 56.99% | 477 / 837 | 0 |
| `passive_1` | 85.24% | 716 / 840 | 2 |
| `passive_2` | 85.60% | 773 / 903 | 0 |
| `principle_A_c_command` | 51.06% | 483 / 946 | 0 |
| `principle_A_case_1` | 100.00% | 912 / 912 | 0 |
| `principle_A_case_2` | 73.77% | 675 / 915 | 5 |
| `principle_A_domain_1` | 99.89% | 913 / 914 | 0 |
| `principle_A_domain_2` | 52.90% | 484 / 915 | 0 |
| `principle_A_domain_3` | 51.22% | 482 / 941 | 0 |
| `principle_A_reconstruction` | 38.99% | 377 / 967 | 0 |
| `regular_plural_subject_verb_agreement_1` | 90.00% | 801 / 890 | 0 |
| `regular_plural_subject_verb_agreement_2` | 84.97% | 803 / 945 | 0 |
| `sentential_negation_npi_licensor_present` | 99.02% | 910 / 919 | 0 |
| `sentential_negation_npi_scope` | 22.85% | 199 / 871 | 0 |
| `sentential_subject_island` | 39.96% | 384 / 961 | 0 |
| `superlative_quantifiers_1` | 78.45% | 768 / 979 | 0 |
| `superlative_quantifiers_2` | 82.25% | 811 / 986 | 0 |
| `tough_vs_raising_1` | 26.90% | 255 / 948 | 0 |
| `tough_vs_raising_2` | 77.50% | 713 / 920 | 0 |
| `transitive` | 70.85% | 615 / 868 | 0 |
| `wh_island` | 66.88% | 642 / 960 | 0 |
| `wh_questions_object_gap` | 61.23% | 526 / 859 | 0 |
| `wh_questions_subject_gap` | 90.42% | 812 / 898 | 0 |
| `wh_questions_subject_gap_long_distance` | 98.25% | 842 / 857 | 0 |
| `wh_vs_that_no_gap` | 93.03% | 801 / 861 | 0 |
| `wh_vs_that_no_gap_long_distance` | 95.20% | 833 / 875 | 0 |
| `wh_vs_that_with_gap` | 18.50% | 170 / 919 | 0 |
| `wh_vs_that_with_gap_long_distance` | 5.71% | 52 / 910 | 0 |

</details>

These values are taken from the saved `results.json`; `metrics.csv` contains the
same group statistics, and `predictions.jsonl` contains the individual pair scores.

### Scoring

Each sentence is conditioned on `[EOS-as-BOS] + sentence bytes`. Scores sum
`log p(byte | prefix)` from the first byte through the last. EOS targets and
padding are excluded. There is no length normalization or temperature tuning.
Sentences are never truncated: inputs exceeding `--max-length` cause an error.
`--chunk-size 64` supports chunked scoring while preserving the prefix in the
recurrent cache.

Like the [official causal scoring implementation](https://github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py),
the evaluator sums sentence log probabilities. Ties use reproducible random
choices keyed by seed and pair ID. These choices can differ from the official
pipeline's sequential RNG, so tie counts and strict `good > bad` accuracy are
also recorded.

This evaluator constructs byte targets directly using the official ByT5 slow
tokenizer interface. It does not require the official pipeline's
`return_offsets_mapping`. Scores are therefore not claimed to be identical to
an official pipeline run. BOS handling, tie policy, and numerical precision are
recorded in the result metadata. Models and tokenizers are saved in HF format.

Outputs:

- `results.json`: macro/micro accuracy, scores by task/linguistic phenomenon/field, and checkpoint/config/hashes.
- `metrics.csv`: accuracy/count/ties for each group.
- `predictions.jsonl`: source sentences, good/bad log probabilities, and predictions for each pair.

Use `--data` to specify a BLiMP JSONL directory. For a quick smoke evaluation,
`--max-pairs-per-task 2` marks the result with `partial_evaluation=true`.
This directory provides LM training and BLiMP evaluation; it does not automatically
produce the other benchmarks or word-budget checkpoints required for a full
BabyLM leaderboard submission.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Three smoke tests cover the final model/tokenizer/LR settings, data preparation →
training → checkpoint saving → BLiMP evaluation on a small fixture, and an
optimizer update plus sentence scoring with the actual GPU configuration.
The GPU test uses the 102.13M model, sequence length 2048, microbatches 11+5,
and BF16 / FlashAttention-2. It is skipped when CUDA is unavailable.
