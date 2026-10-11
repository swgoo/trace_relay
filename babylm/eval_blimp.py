"""BLiMP minimal-pair accuracy using summed causal sentence log-probabilities."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ..modeling_trace_relay import TraceRelayForCausalLM
except ImportError:  # python -m babylm.eval_blimp from the checkout
    from modeling_trace_relay import TraceRelayForCausalLM
from .common import (FORMAT, autocast, check_runtime, encode_text, load_tokenizer,
                     provenance, read_json, sha256, tokenizer_hashes, write_json)


def read_pairs(directory, max_pairs_per_task=None):
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise ValueError(f"No BLiMP .jsonl files in {directory}")
    if max_pairs_per_task is not None and max_pairs_per_task < 1:
        raise ValueError("max_pairs_per_task must be positive")
    records, identities, counts = [], set(), defaultdict(int)
    for path in files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("sentence_good", "sentence_bad"):
                    if not isinstance(row.get(key), str) or not row[key]:
                        raise ValueError(f"Missing/non-string {key}: {path}:{line_number}")
                task = str(row.get("UID", path.stem))
                if max_pairs_per_task and counts[task] >= max_pairs_per_task:
                    continue
                pair_id = str(row.get("pairID", line_number))
                identity = (task, pair_id)
                if identity in identities:
                    raise ValueError(f"Duplicate BLiMP pair: {identity}")
                identities.add(identity)
                counts[task] += 1
                records.append({"task": task, "pair_id": pair_id,
                                "field": row.get("field", "unknown").replace("syntax_semantics", "syntax/semantics"),
                                "linguistics_term": row.get("linguistics_term", "unknown"),
                                "sentence_good": row["sentence_good"], "sentence_bad": row["sentence_bad"]})
    if not records:
        raise ValueError("No BLiMP pairs found")
    return records, {p.name: sha256(p) for p in files}


@torch.no_grad()
def score_sentences(model, tokenizer, sentences, *, device="cpu", precision="float32",
                    chunk_size=0, max_length=8192):
    """Score every text token, including the first, conditioned on EOS as BOS.

    EOS is NOT a scored target. Padding is excluded. No length normalization,
    temperature fitting, or truncation. Chunking preserves the recurrent cache.
    """
    encoded = [encode_text(tokenizer, text) for text in sentences]
    if not encoded or any(not ids for ids in encoded):
        raise ValueError("Sentences must contain at least one token")
    width = max(map(len, encoded))
    if max_length and width > max_length:
        raise ValueError(f"Sentence has {width} tokens, exceeding --max-length {max_length}; no truncation is allowed")
    inputs = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, ids in enumerate(encoded):
        inputs[row, :len(ids)] = torch.tensor([tokenizer.bos_token_id] + ids[:-1], device=device)
        targets[row, :len(ids)] = torch.tensor(ids, device=device)
        mask[row, :len(ids)] = True
    scores = torch.zeros(len(encoded), device=device, dtype=torch.float64)
    cache = None
    size = chunk_size or width
    for start in range(0, width, size):
        end = min(width, start + size)
        with autocast(device, precision):
            result = model(input_ids=inputs[:, start:end], attention_mask=mask[:, start:end],
                           use_cache=bool(chunk_size), past_key_values=cache)
        cache = result.past_key_values if chunk_size else None
        log_probs = F.log_softmax(result.logits.float(), dim=-1)
        selected = log_probs.gather(-1, targets[:, start:end, None]).squeeze(-1)
        scores += selected.masked_fill(~mask[:, start:end], 0).double().sum(-1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Non-finite BLiMP sentence scores")
    return scores.cpu().tolist()


def choose_pair(good, bad, *, task, pair_id, seed, tie_policy):
    if good > bad:
        return 0
    if bad > good or tie_policy == "incorrect":
        return 1
    # Official pipeline breaks ties randomly. Key the random choice to the item
    # here so changing evaluation batch size/order does not change tie decisions.
    digest = hashlib.sha256(f"{seed}\0{task}\0{pair_id}".encode()).digest()
    return digest[0] % 2


def aggregate(predictions):
    rows = []
    for grouping in ("task", "linguistics_term", "field"):
        groups = defaultdict(list)
        for pred in predictions:
            groups[pred[grouping]].append(pred)
        for name, group in sorted(groups.items()):
            correct = sum(p["correct"] for p in group)
            rows.append({"grouping": grouping, "name": name, "count": len(group),
                         "correct": correct, "accuracy": correct / len(group),
                         "ties": sum(p["tie"] for p in group)})
    task_rows = [r for r in rows if r["grouping"] == "task"]
    return {"macro_accuracy": sum(r["accuracy"] for r in task_rows) / len(task_rows),
            "micro_accuracy": sum(p["correct"] for p in predictions) / len(predictions),
            "strict_micro_accuracy": sum(p["good_log_probability"] > p["bad_log_probability"]
                                         for p in predictions) / len(predictions),
            "pair_count": len(predictions), "task_count": len(task_rows),
            "tie_count": sum(p["tie"] for p in predictions), "chance_accuracy": 0.5,
            "groups": rows}


def evaluate(args):
    if args.batch_size < 1 or args.chunk_size < 0 or args.max_length < 0:
        raise ValueError("Invalid evaluation batch/chunk/length setting")
    device = check_runtime(args.device, args.precision, args.attn_implementation)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Evaluation output is not empty: {args.output}")
    tokenizer = load_tokenizer(args.checkpoint)
    model = TraceRelayForCausalLM.from_pretrained(args.checkpoint, attn_implementation="eager", dtype=torch.float32)
    model = model.to(device).eval()
    model.set_attn_implementation(args.attn_implementation)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("Checkpoint model/tokenizer vocabulary mismatch")
    pairs, hashes = read_pairs(args.data, args.max_pairs_per_task)
    predictions = []
    args.output.mkdir(parents=True)
    with (args.output / "predictions.jsonl").open("w") as handle:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            sentences = [p[key] for p in batch for key in ("sentence_good", "sentence_bad")]
            scores = score_sentences(model, tokenizer, sentences, device=device, precision=args.precision,
                                     chunk_size=args.chunk_size, max_length=args.max_length)
            for i, pair in enumerate(batch):
                good, bad = scores[2 * i:2 * i + 2]
                chosen = choose_pair(good, bad, task=pair["task"], pair_id=pair["pair_id"],
                                     seed=args.seed, tie_policy=args.tie_policy)
                pred = {**pair, "good_log_probability": good, "bad_log_probability": bad,
                        "margin": good - bad, "prediction": chosen, "correct": chosen == 0,
                        "tie": good == bad}
                predictions.append(pred)
                handle.write(json.dumps(pred, ensure_ascii=False, allow_nan=False) + "\n")
            if start == 0 or (start // args.batch_size + 1) % 50 == 0:
                print(f"BLiMP: {len(predictions)}/{len(pairs)} pairs", flush=True)
    results = aggregate(predictions)
    checkpoint_files = [*args.checkpoint.glob("*.safetensors"), args.checkpoint / "config.json"]
    results["metadata"] = {"format": FORMAT,
                           "checkpoint": str(args.checkpoint.resolve()),
                           "checkpoint_sha256": {**{p.name: sha256(p) for p in sorted(checkpoint_files)},
                                                 **tokenizer_hashes(args.checkpoint)},
                           "tokenizer_type": "byte", "vocab_size": len(tokenizer),
                           "dataset": str(args.data.resolve()), "dataset_sha256": hashes,
                           "partial_evaluation": args.max_pairs_per_task is not None,
                           "max_pairs_per_task": args.max_pairs_per_task,
                           "score": "sum log p(token | EOS prefix, previous tokens); exclude EOS target",
                           "temperature": 1.0, "length_normalized": False,
                           "tie_policy": args.tie_policy, "seed": args.seed,
                           "precision": args.precision, "attn_implementation": args.attn_implementation,
                           "chunk_size": args.chunk_size, "model_config": model.config.to_dict(),
                           "provenance": provenance()}
    if (args.checkpoint / "training_metadata.json").exists():
        results["metadata"]["training"] = read_json(args.checkpoint / "training_metadata.json")
    write_json(args.output / "results.json", results)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results["groups"][0]))
        writer.writeheader()
        writer.writerows(results["groups"])
    print(json.dumps({k: v for k, v in results.items() if k not in ("groups", "metadata")}), flush=True)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/babylm2026-strict100m/blimp/evaluation_data/full_eval/blimp_filtered"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2, help="Pairs per batch (twice as many sentences)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--attn-implementation", choices=["eager", "flash_attention_2"], default="eager")
    p.add_argument("--chunk-size", type=int, default=0, help="0=full sequence; >0=cached continuation")
    p.add_argument("--max-length", type=int, default=8192, help="Reject longer sentences; 0=no limit")
    p.add_argument("--max-pairs-per-task", type=int, help="Smoke evaluation only; marks results as partial")
    p.add_argument("--tie-policy", choices=["seeded_random", "incorrect"], default="seeded_random")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    evaluate(parser().parse_args())
