"""Configuration for the standalone TraceRelay decoder."""

from __future__ import annotations

import math

from transformers import PreTrainedConfig


def _per_layer(name, value, count, minimum):
    values = [value] * count if isinstance(value, int) else value
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must be an integer or a list of {count} integers")
    if any(type(item) is not int or item < minimum for item in values):
        raise ValueError(f"{name} entries must be integers >= {minimum}")
    return list(values)


class TraceRelayConfig(PreTrainedConfig):
    """A scalar window/width setting broadcasts to every decoder layer.

    Windows count neighboring positions, excluding the center. Right attention
    produces delayed updates from raw input; Left attention reads the residual
    merge of lower hidden and persistent phase readout. Only delayed Right
    updates change phases. Eviction-aligned defaults use equal left/right
    windows and ``relay_stride = left_window + 1``. Shorter Right windows
    are also eviction-aligned when ``right_window <= left_window``.
    Each Right update is delivered strictly after its entire right context: ``relay_stride > right_window``.
    Explicit windows and strides are independent: long Left search horizons may
    overlap several recurrent relay steps. Optional forward skip pairs add a
    source layer's output to the destination layer's input at the same position.
    """

    model_type = "trace_relay"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32_000,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        trace_width=64,
        left_window=32,
        right_window=None,
        relay_stride=None,
        phase_update_scale=math.pi,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        position_encoding="rope",
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        layer_order="right_scan_left",
        skip_pairs=None,
        recurrent_carry=True,
        relay_enabled=True,
        **kwargs,
    ):
        if layer_order != "right_scan_left":
            raise ValueError("Expected layer_order='right_scan_left'; older architectures require their original implementation")
        self.layer_order = layer_order
        for name, value in {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        for name, value in {"recurrent_carry": recurrent_carry, "relay_enabled": relay_enabled}.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            setattr(self, name, value)
        if skip_pairs is None:
            skip_pairs = []
        if not isinstance(skip_pairs, (list, tuple)):
            raise ValueError("skip_pairs must be a list of [source, destination] layer pairs")
        self.skip_pairs = []
        for pair in skip_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(i) is not int for i in pair):
                raise ValueError("Each skip pair must contain two integer layer indices")
            source, destination = pair
            if not 0 <= source < destination < num_hidden_layers:
                raise ValueError("Skip pairs must point forward: 0 <= source < destination < num_hidden_layers")
            if [source, destination] in self.skip_pairs:
                raise ValueError("Duplicate skip pairs are not allowed")
            self.skip_pairs.append([source, destination])
        # A canonical order also makes equivalent topologies share a cache signature.
        self.skip_pairs.sort(key=lambda pair: (pair[1], pair[0]))
        for name, value, minimum in (
            ("trace_width", trace_width, 1),
            ("left_window", left_window, 0),
        ):
            setattr(self, name, _per_layer(name, value, num_hidden_layers, minimum))
        self.right_window = (
            list(self.left_window)
            if right_window is None
            else _per_layer("right_window", right_window, num_hidden_layers, 0)
        )
        unknown_windows = sorted(name for name in kwargs if name.endswith("_window"))
        if unknown_windows:
            raise ValueError(f"Unknown window settings: {', '.join(unknown_windows)}; use left_window and right_window")
        if any(width % num_attention_heads for width in self.trace_width):
            raise ValueError("Each trace_width must be divisible by num_attention_heads")
        if position_encoding not in ("rope", "relative_bias"):
            raise ValueError("position_encoding must be 'rope' or 'relative_bias'")
        if position_encoding == "rope" and any((width // num_attention_heads) % 2 for width in self.trace_width):
            raise ValueError("RoPE requires an even per-head trace width in every layer")
        self.position_encoding = position_encoding
        self.relay_stride = (
            [left + 1 for left in self.left_window]
            if relay_stride is None
            else _per_layer("relay_stride", relay_stride, num_hidden_layers, 1)
        )
        if any(delay <= right for delay, right in zip(self.relay_stride, self.right_window)):
            raise ValueError("Every relay_stride must exceed its right_window")
        for name, value in {
            "phase_update_scale": phase_update_scale,
            "rms_norm_eps": rms_norm_eps,
            "initializer_range": initializer_range,
            "rope_theta": rope_theta,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        for name, value in {"attention_dropout": attention_dropout, "hidden_dropout": hidden_dropout}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
            setattr(self, name, value)
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be a boolean")
        self.use_cache = use_cache
        kwargs.pop("is_decoder", None)
        if kwargs.pop("is_encoder_decoder", False):
            raise ValueError("TraceRelay is a decoder-only model")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            is_encoder_decoder=False,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if "layer_order" not in config_dict:
            raise ValueError("Checkpoint predates right_scan_left; use its original implementation")
        # v0.1 checkpoints predate this field and contain learned bias weights.
        # Preserve their architecture when loading instead of dropping weights.
        config_dict = dict(config_dict)
        config_dict.setdefault("position_encoding", "relative_bias")
        return super().from_dict(config_dict, **kwargs)


TraceRelayConfig.register_for_auto_class()
