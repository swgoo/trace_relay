"""Shared configuration, runtime accounting and tracking for sequence tasks."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import random

import torch

from configuration_trace_relay import TraceRelayConfig


def model_config(args):
    """Build the common backbone settings; task adapters add their own metadata."""
    backend = args.attn_implementation
    if backend == "auto":
        backend = (
            "flash_attention_2"
            if args.device == "cuda" and args.precision != "fp32"
            and importlib.util.find_spec("flash_attn") is not None
            else "eager"
        )
    if backend == "flash_attention_2" and (args.device != "cuda" or args.precision == "fp32"):
        raise ValueError("FlashAttention-2 requires CUDA with auto/BF16 precision")
    if args.precision == "bf16" and args.device != "cuda":
        raise ValueError("BF16 requires CUDA")
    skips = []
    for pair in args.skip_pairs:
        try:
            source, destination = pair.split(":")
            skips.append([int(source), int(destination)])
        except ValueError as error:
            raise ValueError("skip-pairs must look like 0:2") from error

    def per_layer(values):
        return values[0] if len(values) == 1 else values

    config = TraceRelayConfig(
        vocab_size=1, hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
        num_hidden_layers=len(args.trace_widths), num_attention_heads=args.heads,
        trace_width=args.trace_widths, left_window=per_layer(args.left_windows),
        right_window=per_layer(args.right_windows), relay_stride=per_layer(args.relay_strides),
        skip_pairs=skips, recurrent_carry=args.model != "no_carry", relay_enabled=args.model != "swa",
        pad_token_id=None, bos_token_id=None, eos_token_id=None, use_cache=False,
    )
    return dict(family=args.model, attention_backend=backend, backbone=config.to_dict())


def precision_context(device, precision="auto"):
    if precision == "bf16" and str(device).split(":")[0] != "cuda":
        raise ValueError("BF16 evaluation requires CUDA")
    if str(device).startswith("cuda") and precision != "fp32":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def rng_state():
    return dict(
        torch=torch.get_rng_state(), python=random.getstate(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resource_counts(model, sequence_length=None, batch_size=1, cache_dtype=None):
    """Count TraceRelay inference tensors, including diagnostic no-carry tails."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sequence_length is not None and (type(sequence_length) is not int or sequence_length < 0):
        raise ValueError("sequence_length must be a nonnegative integer")
    config = model.config
    if config["family"] not in ("trace_relay", "no_carry", "swa"):
        raise ValueError("Resource accounting requires a TraceRelay task adapter")
    h = config["backbone"]["hidden_size"]
    weight_dtype = next(model.parameters()).dtype
    cache_dtype = weight_dtype if cache_dtype is None else cache_dtype
    hidden_bytes = torch.empty((), dtype=weight_dtype).element_size()
    layout = {}
    persistent_elements = persistent_bytes = attention_elements = attention_bytes = 0
    transient_elements = transient_bytes = 0
    for index, layer in enumerate(model.backbone.layers):
        enabled = config["backbone"]["relay_enabled"]
        history = layer.history_size if sequence_length is None else min(sequence_length, layer.history_size)
        phases = batch_size * layer.stride * config["backbone"]["trace_width"][index] if enabled else 0
        validity = batch_size * layer.stride if enabled else 0
        if sequence_length == 0:
            phases = validity = 0
        raw = batch_size * history * h if enabled else 0
        memory, mask = batch_size * history * h, batch_size * history
        layout[str(index)] = dict(
            raw_hidden=raw, memory_hidden=memory, mask=mask,
            phases=phases, phase_valid=validity,
        )
        attention_elements += raw + memory + mask
        attention_bytes += (raw + memory) * hidden_bytes + mask
        if config["backbone"]["recurrent_carry"]:
            persistent_elements += phases + validity
            persistent_bytes += phases * 4 + validity
        else:
            transient_elements += phases + validity
            transient_bytes += phases * 4 + validity
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    scope = "bounded maximum" if sequence_length is None else f"after {sequence_length} tokens"
    return dict(
        parameters=parameters, parameter_count=parameters, trainable_parameters=trainable,
        persistent_state_elements=persistent_elements, persistent_state_bytes=persistent_bytes,
        attention_cache_elements=attention_elements, attention_cache_bytes=attention_bytes,
        transient_state_elements=transient_elements, transient_state_bytes=transient_bytes,
        cache_elements=persistent_elements + attention_elements + transient_elements,
        cache_bytes=persistent_bytes + attention_bytes + transient_bytes,
        cache_layout=layout,
        cache_kind="bounded raw and memory-augmented hidden histories (not projected K/V)",
        cache_scope=scope, batch_size=batch_size, sequence_length=sequence_length,
        hidden_history_dtype=str(weight_dtype), projected_cache_dtype=str(cache_dtype),
    )


class ExperimentTracking:
    """Optional W&B tracking; disabled runs write only their local JSON logs."""

    def __init__(self, directory, protocol, mode="disabled", project="trace relay", entity=None, name=None):
        self.directory = Path(directory)
        self.protocol = protocol
        self.mode, self.project, self.entity = mode, project, entity
        self.name = name or self.directory.name
        self.run = None
        self.identity = None

    def __enter__(self):
        if self.mode != "disabled":
            import wandb

            self.run = wandb.init(
                project=self.project, entity=self.entity, name=self.name, mode=self.mode,
                dir=str(self.directory), config=self.protocol, save_code=False, job_type="training",
                settings=wandb.Settings(disable_git=True, console="off"),
            )
            self.run.define_metric("step")
            self.run.define_metric("*", step_metric="step")
            self.identity = dict(
                id=self.run.id, entity=self.run.entity, project=self.run.project,
                path=f"{self.run.entity}/{self.run.project}/{self.run.id}", url=self.run.url,
            )
            write_json(self.directory / "wandb_run.json", self.identity)
            print(f"W&B run: {self.run.url}", flush=True)
        return self

    def log(self, row):
        if self.run is None:
            return
        phase = row.get("phase", "training")
        record = {"step": row["step"]}
        for key, value in row.items():
            if key not in ("step", "metrics") and isinstance(value, (str, bool, int, float)):
                record[f"{phase}/{key}"] = value
        record.update({f"{phase}/val/{key}": float(value) for key, value in row.get("metrics", {}).items()})
        self.run.log(record)

    def summary(self, prefix, payload):
        if self.run is not None:
            self.run.summary.update({f"{prefix}/{key}": value for key, value in payload.items()})

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            self.run.summary["pipeline/status"] = "failed" if exc_type else "completed"
            self.run.finish(exit_code=1 if exc_type else 0)
