"""Configuration for uniform-width TraceRelay with independent Q/K bandwidths.

The residual stream, V/O projections, and recurrent phase state all use the
same ``hidden_size`` at every layer.  Long vs. local attention is controlled by
per-layer read/write windows; Q/K scoring capacity is independently controlled
by the per-head ``read_qk_head_dim`` and ``write_qk_head_dim`` parameters.

Q/K widths are intentionally allowed to be narrower than V, but not wider.
The model implementation must project Q/K to ``num_attention_heads * qk_dim``
and V to ``hidden_size``. Both attentions look left, including the current
position. Writes enter the recurrent phase immediately, with no delivery
delay: write update -> optional delta attention -> strided phase scan -> read
attention -> FFN. Delta attention reads uncorrected increments and contributes
a bounded residual correction before the scan. The scan
inherits phase from ``t - (write_window[layer] + 1)``, just before the write window.
When using equal-width-only attention kernels, it
must zero-pad rotated Q/K to V's head dimension and explicitly supply the
softmax scale ``qk_dim ** -0.5``.
"""

from __future__ import annotations

import math
from inspect import signature

from transformers import PreTrainedConfig


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _per_layer_ints(
    name: str, value: int | list[int] | tuple[int, ...], count: int,
    *, minimum: int = 0,
) -> list[int]:
    """Normalize a scalar or an exact-length layer list (never a stage list)."""
    values = [value] * count if type(value) is int else value
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{name} must be an integer or a list of {count} integers")
    if any(type(v) is not int or v < minimum for v in values):
        raise ValueError(f"{name} values must be integers >= {minimum}")
    return list(values)


def _finite(name: str, value: float, *, strictly_positive: bool) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if strictly_positive else not 0 <= value < 1)
    ):
        constraint = "positive and finite" if strictly_positive else "in [0, 1)"
        raise ValueError(f"{name} must be {constraint}")
    return float(value)


class TraceRelayConfig(PreTrainedConfig):
    """TraceRelay config without U-Net stages or layer skip connections.

    ``read_qk_head_dim`` and ``write_qk_head_dim`` are per-head dimensions.
    Each accepts one integer for the whole stack or a list of length
    ``num_hidden_layers``.  ``None`` means full-width Q/K (``hidden_size //
    num_attention_heads``). Write Q/K defaults to full width independently
    of the read setting, so writes can remain wide while long reads
    are narrow.  V/O projections and recurrent phases remain full width.

    RoPE requires even Q/K dimensions.  The per-head Q/K dimension is limited
    to the V head dimension to permit padding-based use of FlashAttention-2.

    No ``relay_stride`` or ``layer_order`` config fields are supported. Phase
    carry uses stride ``write_window[i] + 1`` and includes the current write
    immediately; the stride is an inheritance distance, not a delivery delay.

    ``delta_window`` counts past raw increments in addition to the current one;
    zero disables the correction module in that layer. Its Q/K width is set by
    ``delta_qk_head_dim`` (full width by default). ``phase_correction_scale`` is
    a nonnegative bound in radians per coordinate, applied after tanh. Setting
    it to zero silences the correction without removing the module.
    ``delta_detach_inputs`` stops gradients through the correction's Q/K/V
    inputs only. Its forward values, trainable projections, and raw carry
    update are unchanged.
    """

    model_type = "trace_relay"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32_000,
        hidden_size: int = 256,
        intermediate_size: int = 768,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        read_window: int | list[int] = 32,
        write_window: int | list[int] | None = None,
        read_qk_head_dim: int | list[int] | None = None,
        write_qk_head_dim: int | list[int] | None = None,
        phase_update_scale: float = math.pi,
        delta_window: int | list[int] = 0,
        delta_qk_head_dim: int | list[int] | None = None,
        phase_correction_scale: float = math.pi,
        delta_detach_inputs: bool = False,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        position_encoding: str = "rope",
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool = False,
        recurrent_carry: bool = True,
        relay_enabled: bool = True,
        **kwargs,
    ) -> None:
        self.vocab_size = _positive_int("vocab_size", vocab_size)
        self.hidden_size = _positive_int("hidden_size", hidden_size)
        self.intermediate_size = _positive_int("intermediate_size", intermediate_size)
        self.num_hidden_layers = _positive_int("num_hidden_layers", num_hidden_layers)
        self.num_attention_heads = _positive_int("num_attention_heads", num_attention_heads)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        if position_encoding not in ("rope", "relative_bias"):
            raise ValueError("position_encoding must be 'rope' or 'relative_bias'")
        self.position_encoding = position_encoding

        n = self.num_hidden_layers
        self.read_window = _per_layer_ints("read_window", read_window, n)
        self.write_window = (
            self.read_window.copy() if write_window is None
            else _per_layer_ints("write_window", write_window, n)
        )
        self.delta_window = _per_layer_ints("delta_window", delta_window, n)
        v_dim = self.hidden_size // self.num_attention_heads
        self.read_qk_head_dim = _per_layer_ints(
            "read_qk_head_dim",
            v_dim if read_qk_head_dim is None else read_qk_head_dim,
            n,
            minimum=1,
        )
        self.write_qk_head_dim = _per_layer_ints(
            "write_qk_head_dim",
            v_dim if write_qk_head_dim is None else write_qk_head_dim,
            n,
            minimum=1,
        )
        self.delta_qk_head_dim = _per_layer_ints(
            "delta_qk_head_dim",
            v_dim if delta_qk_head_dim is None else delta_qk_head_dim,
            n,
            minimum=1,
        )
        for name, dims in (
            ("read_qk_head_dim", self.read_qk_head_dim),
            ("write_qk_head_dim", self.write_qk_head_dim),
            ("delta_qk_head_dim", self.delta_qk_head_dim),
        ):
            for i, d in enumerate(dims):
                if d > v_dim:
                    raise ValueError(
                        f"{name}[{i}]={d} exceeds V head dimension {v_dim}"
                    )
                if (self.position_encoding == "rope" and d % 2
                        and (name != "delta_qk_head_dim" or self.delta_window[i] > 0)):
                    raise ValueError(f"{name}[{i}] must be even for RoPE, got {d}")

        if (isinstance(phase_correction_scale, bool)
                or not isinstance(phase_correction_scale, (int, float))
                or not math.isfinite(phase_correction_scale) or phase_correction_scale < 0):
            raise ValueError("phase_correction_scale must be nonnegative and finite")
        self.phase_correction_scale = float(phase_correction_scale)
        if not relay_enabled and any(self.delta_window):
            raise ValueError("delta_window requires relay_enabled=True")

        for name, value in {
            "phase_update_scale": phase_update_scale,
            "rope_theta": rope_theta,
            "rms_norm_eps": rms_norm_eps,
            "initializer_range": initializer_range,
        }.items():
            setattr(self, name, _finite(name, value, strictly_positive=True))
        for name, value in {
            "attention_dropout": attention_dropout,
            "hidden_dropout": hidden_dropout,
        }.items():
            setattr(self, name, _finite(name, value, strictly_positive=False))
        for name, value in {
            "recurrent_carry": recurrent_carry,
            "relay_enabled": relay_enabled,
            "use_cache": use_cache,
            "delta_detach_inputs": delta_detach_inputs,
        }.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            setattr(self, name, value)

        kwargs.pop("is_decoder", None)
        if kwargs.pop("is_encoder_decoder", False):
            raise ValueError("TraceRelay is decoder-only")
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=use_cache,
            **kwargs,
        )

    @property
    def v_head_dim(self) -> int:
        """V dimension per attention head; phase width is ``hidden_size``."""
        return self.hidden_size // self.num_attention_heads

    @property
    def read_qk_width(self) -> list[int]:
        """Total Q/K projection width per layer for Read attention."""
        return [self.num_attention_heads * d for d in self.read_qk_head_dim]

    @property
    def write_qk_width(self) -> list[int]:
        """Total Q/K projection width per layer for Write attention."""
        return [self.num_attention_heads * d for d in self.write_qk_head_dim]

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        config_dict = dict(config_dict)
        # Hugging Face ordinarily applies overrides *after* construction.
        # Inject geometry overrides first so length/divisibility validation runs.
        constructor_fields = set(signature(cls.__init__).parameters) - {"self", "kwargs"}
        for name in constructor_fields.intersection(kwargs):
            config_dict[name] = kwargs.pop(name)
        return super().from_dict(config_dict, **kwargs)


TraceRelayConfig.register_for_auto_class()
