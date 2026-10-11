"""Train a fresh TraceRelay LM; select checkpoints by held-out LM loss."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

try:
    from ..configuration_trace_relay import TraceRelayConfig
    from ..modeling_trace_relay import TraceRelayForCausalLM
except ImportError:  # python -m babylm.train from the checkout
    from configuration_trace_relay import TraceRelayConfig
    from modeling_trace_relay import TraceRelayForCausalLM
from .common import (FORMAT, ROOT, autocast, check_runtime, load_tokenizer,
                     provenance, read_json, sha256, tokenizer_hashes, write_json)
from .data import PackedTokens


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/babylm2026-strict100m"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "babylm/configs/unet16_102m.json")
    loading = p.add_mutually_exclusive_group()
    loading.add_argument("--resume", type=Path, help="Path to this run's last checkpoint")
    loading.add_argument("--init-from", type=Path,
                         help="Initialize all model weights from a checkpoint; start fresh optimizer and training counters")
    p.add_argument("--resume-new-data", action="store_true",
                   help="Continue --resume on a new corpus in a new output directory; preserve optimizer and step-based LR cycle, reset data position and validation best")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--attn-implementation", choices=["eager", "flash_attention_2"], default="flash_attention_2")
    p.add_argument("--sequence-length", type=int, default=2048, help="Tokenizer tokens per packed input")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, help="Optional optimizer-step budget within the epoch budget")
    p.add_argument("--stop-after-steps", type=int, help="Save and pause early without changing the LR schedule")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--effective-batch-size", type=int,
                   help="Sequences per optimizer update; overrides batch-size * gradient-accumulation and permits a smaller final microbatch")
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--validation-max-blocks", type=int, default=0,
                   help="Fixed validation prefix, or 0 for the full held-out stream")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-schedule", choices=["cosine", "cosine_cycle"], default="cosine",
                   help="Cosine decay, or continuous alternating cosine rises/falls after its first minimum")
    p.add_argument("--cosine-half-cycle-epochs", type=float, default=10.0,
                   help="Epochs from an LR minimum to maximum, or maximum to minimum")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default="TraceRelayBabyLM")
    p.add_argument("--wandb-name")
    p.add_argument("--wandb-new-run", action="store_true",
                   help="Start a new W&B run when resuming, e.g. after rolling back to a saved checkpoint")
    p.add_argument("--wandb-history-step-offset", type=int,
                   help="Preserve monotonic W&B history when replaying a checkpoint in the same run; optimizer steps remain unchanged")
    return p


def lr_at(step, total_steps, warmup_steps, minimum, *, cycle_steps=None):
    warmup = min(warmup_steps, total_steps)
    if step <= warmup:
        ratio = step / max(1, warmup)
    elif cycle_steps is not None and step > total_steps:
        # Start at the previous cosine's minimum, without a restart jump.
        progress = ((step - total_steps) % (2 * cycle_steps)) / cycle_steps
        ratio = minimum + (1 - minimum) * (1 - math.cos(math.pi * progress)) / 2
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        ratio = minimum + (1 - minimum) * (1 + math.cos(math.pi * min(1, progress))) / 2
    return ratio


def initialize_from_checkpoint(checkpoint, config):
    """Copy all trained model weights, including the independent output head."""
    checkpoint = Path(checkpoint)
    source = TraceRelayForCausalLM.from_pretrained(checkpoint, attn_implementation="eager", dtype=torch.float32)
    source_config, target_config = source.config.to_dict(), config.to_dict()
    ignored = {"dtype", "architectures", "auto_map", "transformers_version",
               "_name_or_path", "use_cache"}
    changed = [key for key in source_config.keys() | target_config.keys()
               if key not in ignored and source_config.get(key) != target_config.get(key)]
    if changed:
        raise ValueError(f"Initialization model configuration differs: {sorted(changed)}")
    # Preserve separate Parameter storage while copying every trained weight.
    model = TraceRelayForCausalLM(config).float()
    model.load_state_dict(source.state_dict(), strict=True, assign=False)
    if not torch.equal(model.get_output_embeddings().weight, source.get_output_embeddings().weight):
        raise RuntimeError("Initialized output head does not match source weights")
    if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("Input embeddings and output weights must have separate storage")
    info = {"checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": {p.name: sha256(p) for p in sorted(
                [*checkpoint.glob("*.safetensors"), checkpoint / "config.json"])},
            "tokenizer_sha256": tokenizer_hashes(checkpoint),
            "lm_head_initialization": "copied trained source output weights",
            "optimizer_state": "fresh", "training_counters": "reset"}
    if (checkpoint / "training_metadata.json").exists():
        info["source_training"] = read_json(checkpoint / "training_metadata.json")
    return model, info


def collate(dataset, indices, device):
    batch = default_collate([dataset[int(i)] for i in indices])
    return {k: v.to(device) for k, v in batch.items()}


def token_loss(model, batch):
    # Explicit next-token labels from the packed stream: no lost boundary target
    # and no second shift (the backbone's labels= interface shifts internally).
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                           batch["labels"].reshape(-1), ignore_index=-100, reduction="sum")


@torch.no_grad()
def validate(model, dataset, args, device):
    model.eval()
    count = len(dataset) if args.validation_max_blocks == 0 else min(len(dataset), args.validation_max_blocks)
    total_loss, total_tokens = 0.0, 0
    for start in range(0, count, args.eval_batch_size):
        batch = collate(dataset, range(start, min(count, start + args.eval_batch_size)), device)
        with autocast(device, args.precision):
            loss = token_loss(model, batch)
        total_loss += loss.item()
        total_tokens += int((batch["labels"] != -100).sum())
    nll = total_loss / total_tokens
    if not math.isfinite(nll):
        raise FloatingPointError("Non-finite validation loss")
    model.train()
    perplexity = math.exp(min(nll, 700))
    return {"loss": nll, "token_perplexity": perplexity,
            "byte_token_perplexity": perplexity,
            "target_tokens": total_tokens, "blocks": count}


def save_checkpoint(directory, model, tokenizer, protocol, progress, state=None):
    directory = Path(directory)
    pending = directory.with_name(directory.name + ".pending")
    previous = directory.with_name(directory.name + ".previous")
    # Only these run-owned temporary directories are replaced.
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    model.save_pretrained(pending)
    tokenizer.save_pretrained(pending)
    write_json(pending / "protocol.json", protocol)
    write_json(pending / "training_metadata.json", progress)
    if state is not None:
        torch.save(state, pending / "training_state.pt")
    if previous.exists():
        shutil.rmtree(previous)
    if directory.exists():
        directory.rename(previous)
    pending.rename(directory)
    if previous.exists():
        shutil.rmtree(previous)


def train(args):
    if args.resume_new_data and not args.resume:
        raise ValueError("--resume-new-data requires --resume")
    for name in ("epochs", "batch_size", "eval_batch_size", "gradient_accumulation", "sequence_length",
                 "eval_every", "save_every", "log_every", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.validation_max_blocks < 0 or args.warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("Invalid validation/warmup/minimum LR setting")
    if args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    if not math.isfinite(args.cosine_half_cycle_epochs) or args.cosine_half_cycle_epochs <= 0:
        raise ValueError("cosine_half_cycle_epochs must be finite and positive")
    if args.wandb_history_step_offset is not None and args.wandb_history_step_offset < 0:
        raise ValueError("W&B history step offset must be nonnegative")
    if any(value is not None and value < 1 for value in (args.max_steps, args.stop_after_steps)):
        raise ValueError("Step limits must be positive")
    if args.effective_batch_size is not None and args.effective_batch_size < args.batch_size:
        raise ValueError("effective_batch_size must be >= batch_size")
    device = check_runtime(args.device, args.precision, args.attn_implementation)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manifest = read_json(args.data / "manifest.json")
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported prepared data format")
    for name, expected in manifest["prepared_sha256"].items():
        if sha256(args.data / name) != expected:
            raise ValueError(f"Prepared data hash mismatch: {name}")
    tokenizer = load_tokenizer(args.data / "tokenizer")
    if manifest["vocab_size"] != len(tokenizer):
        raise ValueError("Prepared data/tokenizer vocabulary mismatch")
    if "tokenizer_sha256" in manifest and tokenizer_hashes(args.data / "tokenizer") != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer hash mismatch")
    config = TraceRelayConfig.from_dict(read_json(args.config))
    expected = (len(tokenizer), tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    if (config.vocab_size, config.pad_token_id, config.bos_token_id, config.eos_token_id) != expected:
        raise ValueError(f"Model config/tokenizer mismatch; expected vocab/PAD/BOS/EOS={expected}")
    config.use_cache = False
    config._attn_implementation = "eager"
    config.dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.precision]
    training = PackedTokens(args.data, "train", args.sequence_length)
    validation = PackedTokens(args.data, "validation", args.sequence_length)
    group_size = args.effective_batch_size or args.batch_size * args.gradient_accumulation
    epoch_steps = math.ceil(len(training) / group_size)
    total_steps = min(args.epochs * epoch_steps, args.max_steps or args.epochs * epoch_steps)
    half_cycle_steps = max(1, round(args.cosine_half_cycle_epochs * epoch_steps))
    schedule_steps = half_cycle_steps if args.lr_schedule == "cosine_cycle" else total_steps
    schedule = {"type": args.lr_schedule, "first_trough_step": schedule_steps,
                "half_cycle_steps": half_cycle_steps if args.lr_schedule == "cosine_cycle" else None,
                "half_cycle_epochs": args.cosine_half_cycle_epochs if args.lr_schedule == "cosine_cycle" else None}
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    track = manifest.get("track", manifest.get("dataset", {}).get("track", "2026-strict"))
    protocol = {"format": manifest["format"], "task": f"babylm-{track}-causal-byte-lm",
                "options": options, "model_config": json.loads(json.dumps(config.to_dict())),
                "data_manifest_sha256": sha256(args.data / "manifest.json"), "data": manifest,
                "planned_steps": total_steps, "checkpoint_selection": "minimum held-out LM loss",
                "steps_per_epoch": epoch_steps, "lr_schedule": schedule,
                "effective_batch_size": group_size,
                "microbatches_per_full_update": math.ceil(group_size / args.batch_size),
                "blimp_used_for_training_or_selection": False, "provenance": provenance(),
                "torch_version": torch.__version__}
    state = None
    data_transition = False
    if args.resume:
        old = read_json(args.resume / "protocol.json")
        data_transition = old["data_manifest_sha256"] != protocol["data_manifest_sha256"]
        if args.resume_new_data and not data_transition:
            raise ValueError("--resume-new-data requires a different data manifest")
        if data_transition and not args.resume_new_data:
            raise ValueError("Resume data differs; an intentional corpus change requires --resume-new-data")
        # Weight-only initialization is historical provenance on later resumes;
        # --resume never reinitializes the model or its optimizer.
        options["init_from"] = old["options"].get("init_from")
        mutable = {"resume", "resume_new_data", "stop_after_steps", "output", "data", "config", "device", "cpu_threads",
                   "wandb_mode", "wandb_project", "wandb_name", "wandb_new_run", "log_every", "save_every", "epochs",
                   "wandb_history_step_offset"}
        if args.epochs < old["options"]["epochs"]:
            raise ValueError("Resume cannot reduce the epoch budget")
        old_schedule = old.get("lr_schedule", {
            "type": "cosine", "first_trough_step": old.get("lr_schedule_steps", old["planned_steps"]),
            "half_cycle_steps": None, "half_cycle_epochs": None})
        if args.lr_schedule != old_schedule["type"]:
            raise ValueError("Resume must preserve the LR schedule selected at initialization")
        if old_schedule["type"] == "cosine_cycle" and (
                args.lr_schedule != "cosine_cycle"
                or args.cosine_half_cycle_epochs != old_schedule["half_cycle_epochs"]
                or (not data_transition and "data_epoch_start_step" not in old
                    and half_cycle_steps != old_schedule["half_cycle_steps"])):
            raise ValueError("Resume must preserve the existing cosine cycle and its period")
        if old_schedule["type"] == "cosine_cycle":
            schedule["half_cycle_steps"] = old_schedule["half_cycle_steps"]
            schedule["half_cycle_epochs"] = old_schedule["half_cycle_epochs"]
        changed = [k for k in options if k not in mutable and options[k] != old["options"].get(k)]
        if changed:
            raise ValueError(f"Resume protocol differs: {changed}; data and training settings must match")
        # Backend/dtype fields may be rewritten by save_pretrained; compare the
        # original architecture, not transient checkpoint serialization fields.
        if old["model_config"] != protocol["model_config"]:
            raise ValueError("Resume model configuration differs")
        if data_transition:
            if args.output.resolve() == args.resume.resolve().parent or (
                    args.output.exists() and any(args.output.iterdir())):
                raise ValueError("Resume on new data requires a new empty output directory")
            source_tokenizer = load_tokenizer(args.resume)
            if (source_tokenizer.get_vocab() != tokenizer.get_vocab()
                    or (source_tokenizer.pad_token_id, source_tokenizer.bos_token_id, source_tokenizer.eos_token_id)
                    != (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)):
                raise ValueError("Resume on new data requires an identical tokenizer")
        elif args.output.resolve() != args.resume.resolve().parent:
            raise ValueError("Resume into the checkpoint's existing run directory")
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        model = TraceRelayForCausalLM.from_pretrained(args.resume, attn_implementation="eager", dtype=torch.float32)
        if data_transition:
            old.setdefault("data_transitions", []).append({
                "checkpoint": str(args.resume.resolve()), "step": state["step"],
                "previous_epoch": state["epoch"], "previous_cursor": state["cursor"],
                "previous_data": old["data"], "previous_data_manifest_sha256": old["data_manifest_sha256"],
                "new_data": manifest, "new_data_manifest_sha256": protocol["data_manifest_sha256"],
                "previous_best_validation_loss": state["best_validation_loss"],
                "optimizer": "preserved", "lr_phase_and_step_period": "preserved",
                "data_position": "reset", "validation_best": "reset", "provenance": provenance(),
            })
            old["data_epoch_start_step"] = state["step"]
            old["task"], old["data"] = protocol["task"], manifest
            old["data_manifest_sha256"] = protocol["data_manifest_sha256"]
            state["epoch"], state["cursor"], state["best_validation_loss"] = 0, 0, None
        total_steps = min(old.get("data_epoch_start_step", 0) + args.epochs * epoch_steps,
                          args.max_steps or math.inf)
        protocol = old
        schedule_steps = old_schedule["first_trough_step"]
        schedule["first_trough_step"] = schedule_steps
        protocol.setdefault("initial_options", dict(old["options"]))
        protocol["lr_schedule"] = schedule
        protocol["lr_schedule_steps"] = schedule_steps
        protocol["options"] = options
        protocol["planned_steps"] = total_steps
        protocol["steps_per_epoch"] = epoch_steps
        protocol.setdefault("resumptions", []).append({
            "checkpoint": str(args.resume.resolve()), "step": state["step"],
            "planned_steps": total_steps, "lr_schedule_steps": schedule_steps,
            "previous_lr_schedule": old_schedule, "lr_schedule": dict(schedule),
            "parent_wandb_id": state.get("wandb_id"), "new_wandb_run": args.wandb_new_run,
            "provenance": provenance(),
        })
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"Output is not empty: {args.output}")
        if args.init_from:
            source_tokenizer = load_tokenizer(args.init_from)
            if (source_tokenizer.get_vocab() != tokenizer.get_vocab()
                    or (source_tokenizer.pad_token_id, source_tokenizer.bos_token_id, source_tokenizer.eos_token_id)
                    != (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)):
                raise ValueError("Initialization checkpoint tokenizer differs from prepared data")
            model, protocol["initialization"] = initialize_from_checkpoint(args.init_from, config)
        else:
            model = TraceRelayForCausalLM(config)
    args.output.mkdir(parents=True, exist_ok=True)
    model = model.float().to(device)
    model.set_attn_implementation(args.attn_implementation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "float16")
    step, epoch, cursor, tokens, words, wall = 0, 0, 0, 0, 0, 0.0
    best_loss = None
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, epoch, cursor = state["step"], state["epoch"], state["cursor"]
        tokens, words, wall, best_loss = state["tokens"], state["words"], state["wall_seconds"], state["best_validation_loss"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    history_offset = args.wandb_history_step_offset
    if history_offset is None:
        history_offset = state.get("wandb_history_step_offset", 0) if state and not args.wandb_new_run else 0
    protocol["options"]["wandb_history_step_offset"] = history_offset
    protocol["wandb_history_step_offset"] = history_offset
    protocol["parameter_count"] = sum(p.numel() for p in model.parameters())
    write_json(args.output / "protocol.json", protocol)
    tracking = None
    if args.wandb_mode != "disabled":
        import wandb
        resume_id = state.get("wandb_id") if state and not args.wandb_new_run else None
        tracking = wandb.init(project=args.wandb_project, name=args.wandb_name or args.output.name,
                              mode=args.wandb_mode, dir=str(args.output), config=protocol,
                              id=resume_id, allow_val_change=bool(state),
                              resume="must" if resume_id and args.wandb_mode == "online" else "allow")
        if state:
            tracking.config.update(protocol, allow_val_change=True)
        if history_offset:
            tracking.define_metric("step")
            tracking.define_metric("train/*", step_metric="step")
            tracking.define_metric("validation/*", step_metric="step")
    if state:
        print(json.dumps({"event": "resume", "step": step, "epoch": epoch, "cursor": cursor,
                          "planned_steps": total_steps, "lr_schedule": protocol["lr_schedule"],
                          "wandb_id": tracking.id if tracking else None,
                          "wandb_history_step_offset": history_offset}), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model.train()

    def progress():
        return {"format": protocol["format"], "step": step, "epoch": epoch, "cursor": cursor,
                "tokens": tokens, "words": words, "wall_seconds": wall + time.monotonic() - started,
                "best_validation_loss": best_loss, "parameter_count": protocol["parameter_count"],
                "wandb_history_step": step + history_offset}

    def save_last():
        snapshot = {**progress(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "wandb_id": tracking.id if tracking else None,
                    "wandb_history_step_offset": history_offset}
        save_checkpoint(args.output / "last", model, tokenizer, protocol, progress(), snapshot)

    def log(row):
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
        if tracking:
            tracking.log({k: v for k, v in row.items() if k != "event"}, step=step + history_offset)

    def validation_event():
        nonlocal best_loss
        results = validate(model, validation, args, device)
        log({"event": "validation", "step": step, **{f"validation/{k}": v for k, v in results.items()}})
        if best_loss is None or results["loss"] < best_loss:
            best_loss = results["loss"]
            save_checkpoint(args.output / "best", model, tokenizer, protocol, progress())
        return results

    initial_validation = validation_event() if not state or data_transition else None
    if data_transition:
        save_last()
    final_validation = initial_validation
    halt_step = min(total_steps, args.stop_after_steps or total_steps)
    while step < halt_step and epoch < args.epochs:
        order = torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + epoch))
        indices = order[cursor:cursor + group_size]
        batches = [collate(training, indices[i:i + args.batch_size], device)
                   for i in range(0, len(indices), args.batch_size)]
        group_tokens = sum(int((b["labels"] != -100).sum()) for b in batches)
        group_words = sum(int(b["words"].sum()) for b in batches)
        lr = args.learning_rate * lr_at(
            step + 1, protocol["lr_schedule"]["first_trough_step"], args.warmup_steps, args.min_lr_ratio,
            cycle_steps=protocol["lr_schedule"]["half_cycle_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch in batches:
            with autocast(device, args.precision):
                loss = token_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
            total_loss += loss.item()
            scaler.scale(loss / group_tokens).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        cursor += len(indices)
        tokens += group_tokens
        words += group_words
        if cursor == len(training):
            epoch, cursor = epoch + 1, 0
        if step == 1 or step % args.log_every == 0 or step == halt_step:
            elapsed = wall + time.monotonic() - started
            row = {"event": "train", "step": step, "train/loss": total_loss / group_tokens,
                   "train/gradient_norm": float(norm), "train/learning_rate": lr,
                   "tokens": tokens, "words": words, "epoch": epoch,
                   "wall_seconds": elapsed, "tokens_per_second": tokens / elapsed}
            if device.type == "cuda":
                row.update({"train/peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "train/peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            log(row)
        if step % args.eval_every == 0 or step == halt_step:
            final_validation = validation_event()
        if step % args.save_every == 0 or step == halt_step:
            save_last()
    if step >= halt_step and final_validation is None:
        final_validation = validation_event()
        save_last()
    summary = {**progress(), "status": "complete" if step >= total_steps else "paused",
               "initial_validation": initial_validation, "final_validation": final_validation,
               "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_json(args.output / "summary.json", summary)
    if tracking:
        tracking.finish()
    return summary


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
