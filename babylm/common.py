from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import ByT5Tokenizer

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "trace-relay-babylm-byte-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance():
    files = [ROOT / "configuration_trace_relay.py", ROOT / "modeling_trace_relay.py",
             *sorted((ROOT / "babylm").glob("*.py"))]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files}}


def make_tokenizer():
    # Official Transformers implementation; no pretrained model or learned vocabulary.
    # EOS doubles as a causal sequence prefix. Literal special-token strings in
    # corpus text are encoded as bytes, rather than injected as control tokens.
    return ByT5Tokenizer(extra_ids=0, bos_token="</s>", split_special_tokens=True,
                         model_max_length=2**31 - 1, padding_side="right")


def load_tokenizer(path):
    # A tokenizer does not need the checkpoint's custom model config/code.
    # AutoTokenizer may inspect that config and prompt for trust_remote_code.
    name = read_json(Path(path) / "tokenizer_config.json").get("tokenizer_class")
    if name != "ByT5Tokenizer":
        raise ValueError("Expected a saved ByT5 byte tokenizer")
    tokenizer = ByT5Tokenizer.from_pretrained(path, local_files_only=True)
    if (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) != (0, 1, 1):
        raise ValueError("Expected PAD=0 and BOS=EOS=1")
    if len(tokenizer) != 259:
        raise ValueError("Expected 259-token ByT5 vocabulary")
    return tokenizer


def tokenizer_hashes(path):
    path = Path(path)
    names = ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]
    return {name: sha256(path / name) for name in names if (path / name).is_file()}


def encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)


def autocast(device, precision):
    if precision == "float32":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[precision]
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def check_runtime(device, precision, backend):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu --precision float32 --attn-implementation eager")
    if backend == "flash_attention_2" and (device.type != "cuda" or precision == "float32"):
        raise ValueError("FlashAttention-2 requires CUDA and bfloat16/float16")
    if precision == "float16" and device.type != "cuda":
        raise ValueError("float16 training is supported only on CUDA")
    if device.type == "cuda" and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bfloat16; choose float16")
    return device
