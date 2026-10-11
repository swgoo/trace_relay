
"""TraceRelay U-Net layouts with one hidden dimension per layer.

``hidden_size`` controls a uniform stack; ``hidden_sizes`` and ``depths``
control stages. Attention projections and recurrent phases use those same
hidden dimensions.
"""
from __future__ import annotations

import math
from inspect import signature
from transformers import PreTrainedConfig


def _ints(name, value, count, minimum=1):
    seq = [value] * count if type(value) is int else value
    if not isinstance(seq, (list, tuple)) or len(seq) != count:
        raise ValueError(
            f"{name} must be an integer or a list of {count} integers"
        )
    if any(type(x) is not int or x < minimum for x in seq):
        raise ValueError(
            f"{name} values must be integers >= {minimum}"
        )
    return list(seq)


def _stage_to_layers(name, value, depths, minimum=0):
    """Scalar, per-stage, or per-layer value -> per-layer list."""
    count, stages = sum(depths), len(depths)

    if type(value) is int:
        return _ints(name, value, count, minimum)

    if isinstance(value, (list, tuple)) and len(value) == stages:
        vals = _ints(name, value, stages, minimum)
        return [
            x
            for x, repeat in zip(vals, depths)
            for _ in range(repeat)
        ]

    return _ints(name, value, count, minimum)


def _pairs(name, pairs, limit):
    if not isinstance(pairs, (list, tuple)):
        raise ValueError(f"{name} must be a list of pairs")

    result = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{name} must contain pairs")

        a, b = pair
        if (
            type(a) is not int
            or type(b) is not int
            or not 0 <= a < b < limit
        ):
            raise ValueError(
                f"{name} requires 0 <= source < destination < {limit}"
            )
        if [a, b] in result:
            raise ValueError(f"Duplicate {name} pair: {pair}")
        result.append([a, b])

    return sorted(result, key=lambda p: (p[1], p[0]))


class TraceRelayConfig(PreTrainedConfig):
    model_type = "trace_relay"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32_000,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,
        left_window=32,
        right_window=None,
        relay_stride=None,
        # Optional stage-wise API.
        hidden_sizes=None,
        depths=None,
        left_windows=None,
        right_windows=None,
        relay_strides=None,
        mlp_ratios=None,
        stage_num_attention_heads=None,
        skip_stage_pairs=None,
        # Original model options.
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
        tie_word_embeddings=False,
        layer_order="right_scan_left",
        skip_pairs=None,
        recurrent_carry=True,
        relay_enabled=True,
        **kwargs,
    ):
        if tie_word_embeddings is not False:
            raise ValueError("TraceRelay requires independent input embeddings and output weights")
        removed_fields = sorted(
            name for name in ("trace_width", "trace_widths") if name in kwargs
        )
        if removed_fields:
            raise TypeError(
                f"Unsupported width settings: {removed_fields}; "
                "use hidden_size or hidden_sizes"
            )
        if layer_order != "right_scan_left":
            raise ValueError(
                "Only layer_order='right_scan_left' is supported"
            )
        self.layer_order = layer_order
        self.vocab_size = _ints("vocab_size", vocab_size, 1)[0]

        if (hidden_sizes is None) != (depths is None):
            raise ValueError(
                "hidden_sizes and depths must be supplied together"
            )

        staged = hidden_sizes is not None

        if staged:
            if (
                not isinstance(hidden_sizes, (list, tuple))
                or not hidden_sizes
            ):
                raise ValueError(
                    "hidden_sizes must be a nonempty stage list"
                )

            self.hidden_sizes = _ints(
                "hidden_sizes", hidden_sizes, len(hidden_sizes)
            )
            self.depths = _ints(
                "depths", depths, len(self.hidden_sizes)
            )
            self.num_hidden_layers = sum(self.depths)

            # HF embedding / LM-head width.
            self.hidden_size = self.hidden_sizes[0]

            self.layer_stage_indices = [
                s for s, d in enumerate(self.depths)
                for _ in range(d)
            ]
            self.layer_hidden_sizes = [
                self.hidden_sizes[s]
                for s in self.layer_stage_indices
            ]

            heads = (
                num_attention_heads
                if stage_num_attention_heads is None
                else stage_num_attention_heads
            )
            self.stage_num_attention_heads = _ints(
                "num_attention_heads", heads, len(self.depths)
            )

            # HF scalar alias.
            self.num_attention_heads = (
                self.stage_num_attention_heads[0]
            )
            self.layer_num_attention_heads = [
                self.stage_num_attention_heads[s]
                for s in self.layer_stage_indices
            ]

            if mlp_ratios is None:
                mlp_ratios = [
                    intermediate_size / hidden_size
                ] * len(self.depths)
            elif (
                isinstance(mlp_ratios, (int, float))
                and not isinstance(mlp_ratios, bool)
            ):
                mlp_ratios = [
                    float(mlp_ratios)
                ] * len(self.depths)

            if (
                not isinstance(mlp_ratios, (list, tuple))
                or len(mlp_ratios) != len(self.depths)
            ):
                raise ValueError(
                    "mlp_ratios must have one value per stage"
                )

            if any(
                isinstance(r, bool)
                or not isinstance(r, (int, float))
                or not math.isfinite(r)
                or r <= 0
                for r in mlp_ratios
            ):
                raise ValueError(
                    "mlp_ratios must be positive, finite numbers"
                )

            self.mlp_ratios = list(mlp_ratios)
            self.layer_intermediate_sizes = [
                max(1, round(h * self.mlp_ratios[s]))
                for h, s in zip(
                    self.layer_hidden_sizes,
                    self.layer_stage_indices,
                )
            ]
            self.intermediate_size = (
                self.layer_intermediate_sizes[0]
            )

            # Keep original stage values for serialization.
            self.left_windows = left_windows
            self.right_windows = right_windows
            self.relay_strides = relay_strides

            self.left_window = _stage_to_layers(
                "left_window",
                left_window if left_windows is None else left_windows,
                self.depths,
            )

            right_input = (
                right_window if right_windows is None
                else right_windows
            )
            self.right_window = (
                self.left_window.copy()
                if right_input is None
                else _stage_to_layers(
                    "right_window", right_input, self.depths
                )
            )

            stride_input = (
                relay_stride if relay_strides is None
                else relay_strides
            )
            self.relay_stride = (
                [w + 1 for w in self.left_window]
                if stride_input is None
                else _stage_to_layers(
                    "relay_stride", stride_input, self.depths, 1
                )
            )

            if skip_stage_pairs is not None:
                stage_pairs = _pairs(
                    "skip_stage_pairs",
                    skip_stage_pairs,
                    len(self.depths),
                )
                boundaries = [0]
                for depth in self.depths:
                    boundaries.append(boundaries[-1] + depth)

                # Last layer of source stage ->
                # first layer of destination stage.
                mapped = [
                    [boundaries[src + 1] - 1, boundaries[dst]]
                    for src, dst in stage_pairs
                ]

                if (
                    skip_pairs is not None
                    and _pairs(
                        "skip_pairs",
                        skip_pairs,
                        self.num_hidden_layers,
                    ) != _pairs(
                        "skip_pairs",
                        mapped,
                        self.num_hidden_layers,
                    )
                ):
                    raise ValueError(
                        "skip_pairs conflicts with skip_stage_pairs"
                    )

                self.skip_pairs = _pairs(
                    "skip_pairs", mapped, self.num_hidden_layers
                )
                self.skip_stage_pairs = stage_pairs
            else:
                self.skip_pairs = _pairs(
                    "skip_pairs",
                    [] if skip_pairs is None else skip_pairs,
                    self.num_hidden_layers,
                )
                self.skip_stage_pairs = None

        else:
            stage_options = (
                left_windows,
                right_windows,
                relay_strides,
                mlp_ratios,
                stage_num_attention_heads,
                skip_stage_pairs,
            )
            if any(x is not None for x in stage_options):
                raise ValueError(
                    "Stage-wise fields require hidden_sizes and depths"
                )

            self.hidden_sizes = None
            self.depths = None
            self.left_windows = None
            self.right_windows = None
            self.relay_strides = None
            self.mlp_ratios = None
            self.stage_num_attention_heads = None
            self.skip_stage_pairs = None

            self.hidden_size = _ints(
                "hidden_size", hidden_size, 1
            )[0]
            self.intermediate_size = _ints(
                "intermediate_size", intermediate_size, 1
            )[0]
            self.num_hidden_layers = _ints(
                "num_hidden_layers", num_hidden_layers, 1
            )[0]
            self.num_attention_heads = _ints(
                "num_attention_heads", num_attention_heads, 1
            )[0]

            self.layer_stage_indices = [
                0
            ] * self.num_hidden_layers
            self.layer_hidden_sizes = [
                self.hidden_size
            ] * self.num_hidden_layers
            self.layer_intermediate_sizes = [
                self.intermediate_size
            ] * self.num_hidden_layers
            self.layer_num_attention_heads = [
                self.num_attention_heads
            ] * self.num_hidden_layers

            self.left_window = _ints(
                "left_window", left_window, self.num_hidden_layers, 0
            )
            self.right_window = (
                self.left_window.copy()
                if right_window is None
                else _ints(
                    "right_window",
                    right_window,
                    self.num_hidden_layers,
                    0,
                )
            )
            self.relay_stride = (
                [w + 1 for w in self.left_window]
                if relay_stride is None
                else _ints(
                    "relay_stride",
                    relay_stride,
                    self.num_hidden_layers,
                )
            )
            self.skip_pairs = _pairs(
                "skip_pairs",
                [] if skip_pairs is None else skip_pairs,
                self.num_hidden_layers,
            )

        # Per-layer validation.
        for i, (width, heads) in enumerate(
            zip(self.layer_hidden_sizes, self.layer_num_attention_heads)
        ):
            if width % heads:
                raise ValueError(
                    f"layer_hidden_sizes[{i}] must be divisible "
                    f"by attention heads ({heads})"
                )
            if (
                position_encoding == "rope"
                and (width // heads) % 2
            ):
                raise ValueError(
                    f"RoPE needs even head dimension at layer {i}"
                )

        if any(
            w <= r
            for w, r in zip(
                self.relay_stride, self.right_window
            )
        ):
            raise ValueError(
                "Every relay_stride must exceed its right_window"
            )

        if position_encoding not in (
            "rope", "relative_bias"
        ):
            raise ValueError(
                "position_encoding must be 'rope' or 'relative_bias'"
            )
        self.position_encoding = position_encoding

        positive_options = {
            "phase_update_scale": phase_update_scale,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_norm_eps,
            "initializer_range": initializer_range,
        }
        for name, value in positive_options.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )
            setattr(self, name, value)

        dropout_options = {
            "attention_dropout": attention_dropout,
            "hidden_dropout": hidden_dropout,
        }
        for name, value in dropout_options.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < 1
            ):
                raise ValueError(
                    f"{name} must be in [0, 1)"
                )
            setattr(self, name, value)

        for name, value in {
            "recurrent_carry": recurrent_carry,
            "relay_enabled": relay_enabled,
            "use_cache": use_cache,
        }.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            setattr(self, name, value)

        unknown_windows = sorted(
            k for k in kwargs if k.endswith("_window")
        )
        if unknown_windows:
            raise ValueError(
                f"Unknown window settings: {unknown_windows}"
            )

        # Recompute these on checkpoint load.
        derived_fields = (
            "layer_hidden_sizes",
            "layer_intermediate_sizes",
            "layer_num_attention_heads",
            "layer_stage_indices",
        )
        for name in derived_fields:
            kwargs.pop(name, None)

        kwargs.pop("is_decoder", None)
        if kwargs.pop("is_encoder_decoder", False):
            raise ValueError("TraceRelay is decoder-only")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=use_cache,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        # Apply constructor overrides before deriving the per-layer geometry.
        # HF's default post-construction setattr would leave those lists stale.
        constructor_fields = (
            *signature(cls.__init__).parameters,
            "trace_width", "trace_widths",
        )
        for name in constructor_fields:
            if name not in ("self", "kwargs") and name in kwargs:
                config_dict[name] = kwargs.pop(name)
        if "layer_order" not in config_dict:
            raise ValueError("Config must declare layer_order='right_scan_left'")
        return super().from_dict(config_dict, **kwargs)


TraceRelayConfig.register_for_auto_class()
