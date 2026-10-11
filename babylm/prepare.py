"""Prepare pinned BabyLM 2026 corpora with the official ByT5 byte tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .common import (FORMAT, encode_text, make_tokenizer, provenance,
                     sha256, tokenizer_hashes, write_json)

TRAIN_REPO = "BabyLM-community/BabyLM-2026-Strict"
TRAIN_REVISION = "9e57baaaa91ac3c638746be14d1d5fa6c789f4cf"
TRACKS = {
    "strict": {"repo_id": TRAIN_REPO, "revision": TRAIN_REVISION, "word_budget": 100_000_000},
}
EVAL_REPO = "BabyLM-community/BabyLM-2026-Strict-Evals"
EVAL_REVISION = "8d52da9424a9ff30b9e8266c4f751aba9c504233"


def download_blimp(directory, revision=EVAL_REVISION):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=EVAL_REPO, repo_type="dataset", revision=revision,
                                  local_dir=str(directory),
                                  allow_patterns=["evaluation_data/full_eval/blimp_filtered/*.jsonl",
                                                  "evaluation_data/fast_eval/blimp_fast/*.jsonl"]))


def validation_line(text, seed, fraction):
    # Identical lines always go to the same split, including across source files.
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def iter_examples(train_files, validation_files, *, seed, validation_fraction, max_train_words,
                  heldout, counters=None):
    """Shared split/budget selection for tokenizer training and stream packing."""
    train_words = 0
    for path in train_files + validation_files:
        explicit_valid = path in validation_files
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.rstrip("\r\n")
                if not text.strip():
                    continue
                if not explicit_valid and hashlib.sha256(text.encode()).digest() in heldout:
                    if counters is not None:
                        counters["skipped_overlap"] += 1
                    continue
                split = "validation" if explicit_valid or (not validation_files and
                    validation_line(text, seed, validation_fraction)) else "train"
                spans = list(re.finditer(r"\S+", text))
                if split == "train":
                    remaining = max_train_words - train_words
                    if remaining == 0:
                        continue
                    if len(spans) > remaining:
                        spans = spans[:remaining]
                        text = text[:spans[-1].end()]
                    train_words += len(spans)
                yield split, text, spans


def word_markers(text, spans, tokens):
    # These exposure counters are never supplied to the model.
    ends = np.zeros(len(tokens) + 1, dtype="u1")
    byte_offset, last_end = 0, 0
    for match in spans:
        byte_offset += len(text[last_end:match.end()].encode("utf-8"))
        ends[byte_offset - 1] = 1
        last_end = match.end()
    return ends


def prepare(output, train_files, *, validation_files=None, validation_fraction=0.01,
            seed=42, max_train_words=None, source=None, track="strict"):
    output = Path(output)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if track not in TRACKS:
        raise ValueError(f"Unknown BabyLM track: {track}")
    budget = TRACKS[track]["word_budget"]
    max_train_words = budget if max_train_words is None else max_train_words
    if not 1 <= max_train_words <= budget:
        raise ValueError(f"{track} max_train_words must be in [1, {budget:,}]")
    if (output / "manifest.json").exists() or (output / "train.bin").exists():
        raise FileExistsError(f"Prepared data already exists: {output}; choose a new directory")
    train_files = sorted({Path(p).resolve() for p in train_files})
    validation_files = sorted({Path(p).resolve() for p in (validation_files or [])})
    if not train_files or any(not p.is_file() for p in train_files + validation_files):
        raise ValueError("Provide existing train/validation text files")
    if set(train_files) & set(validation_files):
        raise ValueError("Train and validation files must be disjoint")
    hashes = {str(p): sha256(p) for p in train_files + validation_files}
    if {hashes[str(p)] for p in train_files} & {hashes[str(p)] for p in validation_files}:
        raise ValueError("Train and validation files contain an identical file")
    output.mkdir(parents=True, exist_ok=True)
    # Explicit local validation files also exclude duplicate text from training.
    heldout = set()
    for path in validation_files:
        with path.open(encoding="utf-8") as handle:
            heldout.update(hashlib.sha256(line.rstrip("\r\n").encode()).digest()
                           for line in handle if line.strip())
    def examples(counters=None):
        return iter_examples(train_files, validation_files, seed=seed,
                             validation_fraction=validation_fraction, max_train_words=max_train_words,
                             heldout=heldout, counters=counters)

    tokenizer = make_tokenizer()
    tokenizer.save_pretrained(output / "tokenizer")
    stats = {split: {"lines": 0, "words": 0, "tokens": 1} for split in ("train", "validation")}
    counters = {"skipped_overlap": 0}
    with ExitStack() as stack:
        writers = {split: (stack.enter_context((output / f"{split}.bin").open("wb")),
                           stack.enter_context((output / f"{split}.words.bin").open("wb")))
                   for split in stats}
        for token_file, word_file in writers.values():
            np.array([tokenizer.bos_token_id], dtype="<u2").tofile(token_file)
            word_file.write(b"\0")
        for split, text, spans in examples(counters):
            tokens = encode_text(tokenizer, text)
            ends = word_markers(text, spans, tokens)
            token_file, word_file = writers[split]
            np.asarray(tokens + [tokenizer.eos_token_id], dtype="<u2").tofile(token_file)
            ends.tofile(word_file)
            stats[split]["lines"] += 1
            stats[split]["words"] += len(spans)
            stats[split]["tokens"] += len(tokens) + 1
            if sum(s["lines"] for s in stats.values()) % 100000 == 0:
                print(f"Prepared streams: {stats}", flush=True)
    if any(s["lines"] == 0 for s in stats.values()):
        raise ValueError("Empty train/validation split; provide more text or explicit validation files")
    manifest = {"format": FORMAT,
                "track": f"2026-{track}",
                "dataset": source or {"type": "local_text"},
                "source_sha256": hashes, "tokenizer": f"transformers.{type(tokenizer).__name__}",
                "tokenizer_type": "byte", "tokenizer_sha256": tokenizer_hashes(output / "tokenizer"),
                "tokenizer_training": None,
                "vocab_size": len(tokenizer), "extra_ids": 0, "bos_equals_eos": True,
                "literal_special_tokens": "encode as bytes", "dtype": "<u2",
                "split_method": "explicit_files" if validation_files else "sha256_line_holdout",
                "validation_fraction": None if validation_files else validation_fraction,
                "split_seed": seed, "max_train_words": max_train_words,
                "word_definition": "Unicode whitespace-delimited words (regex \\S+)",
                "excluded_duplicate_validation_lines": counters["skipped_overlap"],
                "splits": stats, "provenance": provenance()}
    manifest["prepared_sha256"] = {p.name: sha256(p) for p in sorted(output.glob("*.bin"))}
    write_json(output / "manifest.json", manifest)
    return manifest


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=list(TRACKS), default="strict")
    parser.add_argument("--output", type=Path, default=Path("data/babylm2026-strict100m"))
    parser.add_argument("--train-files", type=Path, nargs="+")
    parser.add_argument("--validation-files", type=Path, nargs="+")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--max-train-words", type=int, help="Defaults to the selected track's word budget")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-revision", help="Override the selected track's pinned revision")
    parser.add_argument("--eval-revision", default=EVAL_REVISION)
    parser.add_argument("--skip-blimp", action="store_true")
    return parser


def main():
    command = parser()
    args = command.parse_args()
    if (args.output / "manifest.json").exists() or (args.output / "train.bin").exists():
        command.error("Output already contains prepared data; choose a new directory")
    track = TRACKS[args.track]
    revision = args.train_revision or track["revision"]
    source = None
    if args.train_files is None:
        from huggingface_hub import snapshot_download
        raw = Path(snapshot_download(repo_id=track["repo_id"], repo_type="dataset",
                     revision=revision, local_dir=str(args.output / "raw"),
                     allow_patterns=["*.train.txt", "README.md"]))
        args.train_files = sorted(raw.glob("*.train.txt"))
        source = {"repo_id": track["repo_id"], "revision": revision, "track": f"2026-{args.track}"}
    prepare(args.output, args.train_files, validation_files=args.validation_files,
            validation_fraction=args.validation_fraction, seed=args.seed,
            max_train_words=args.max_train_words, source=source,
            track=args.track)
    if not args.skip_blimp:
        download_blimp(args.output / "blimp", args.eval_revision)
        write_json(args.output / "blimp" / "source.json",
                   {"repo_id": EVAL_REPO, "revision": args.eval_revision})


if __name__ == "__main__":
    main()
