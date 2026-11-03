"""TraceRelay: causal write/read attention with an immediate recurrent phase scan."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from importlib.metadata import version

import torch
import torch.nn.functional as F
from torch import nn
from packaging.version import Version
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from .configuration_trace_relay import TraceRelayConfig
except ImportError:
    from configuration_trace_relay import TraceRelayConfig


def wrap_phase(angles):
    """Wrap in the accumulation dtype before returning readable FP32 angles."""
    return torch.remainder(angles, 2 * math.pi).float()


def phase_stride_scan(increments, stride, initial=None):
    """Solve theta[t] = wrap(theta[t-stride] + increments[t]).

    The current write is applied immediately. With stride = write_window + 1,
    the inherited phase is just before the current write's attention window.
    ``initial`` holds theta[-stride:0] in chronological order. Reshaping into
    stride-wide rows gives independent prefix sums along each phase lane.
    The returned tail supports continuation at arbitrary chunk boundaries.
    """
    batch, length, width = increments.shape
    if length == 0 or stride < 1:
        raise ValueError("phase_stride_scan requires a nonempty sequence and positive stride")
    if initial is None:
        initial = increments.new_zeros(batch, stride, width, dtype=torch.float32)
    if initial.shape != (batch, stride, width):
        raise ValueError("initial must have shape (batch, stride, hidden_size)")
    # FP64 preserves low-order phase bits in long scans; MPS has no FP64.
    accumulation_dtype = torch.float32 if increments.device.type == "mps" else torch.float64
    rows = F.pad(increments.to(accumulation_dtype), (0, 0, 0, (-length) % stride))
    rows = rows.reshape(batch, -1, stride, width)
    phases = wrap_phase(rows.cumsum(dim=1) + initial.to(accumulation_dtype).unsqueeze(1))
    phases = phases.reshape(batch, -1, width)[:, :length]
    tail = torch.cat((initial, phases), dim=1)[:, -stride:]
    return phases, tail


def phase_validity_stride_scan(write_valid, stride, initial=None):
    """Solve valid[t] = valid[t-stride] OR write_valid[t].

    Validity follows physical positions even through padding, just like phase
    carry. A masked position cannot initialize memory; a valid zero-angle write can.
    """
    batch, length = write_valid.shape
    if length == 0 or stride < 1:
        raise ValueError("phase_validity_stride_scan requires a nonempty sequence and positive stride")
    if initial is None:
        initial = write_valid.new_zeros(batch, stride)
    if write_valid.dtype != torch.bool or initial.dtype != torch.bool:
        raise ValueError("Phase validity must be boolean")
    if initial.shape != (batch, stride):
        raise ValueError("initial validity must have shape (batch, stride)")
    rows = F.pad(write_valid, (0, (-length) % stride)).reshape(batch, -1, stride)
    valid = (rows.cumsum(dim=1, dtype=torch.int32) > 0) | initial[:, None]
    valid = valid.reshape(batch, -1)[:, :length]
    tail = torch.cat((initial, valid), dim=1)[:, -stride:]
    return valid, tail


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden):
        normalized = hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(hidden.dtype) * self.weight


class LocalAttention(nn.Module):
    """Causal local attention over the current token and preceding window."""

    def __init__(self, hidden_size, heads, window, dropout, config=None, qk_head_dim=None):
        super().__init__()
        self.config = config
        self.heads, self.head_dim = heads, hidden_size // heads
        self.qk_head_dim = self.head_dim if qk_head_dim is None else qk_head_dim
        self.window, self.dropout = window, dropout
        self.q_proj = nn.Linear(hidden_size, heads * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, heads * self.qk_head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.position_encoding = "relative_bias" if config is None else config.position_encoding
        self.rope_theta = 10_000.0 if config is None else config.rope_theta
        if self.position_encoding == "relative_bias":
            self.relative_bias = nn.Parameter(torch.zeros(heads, window + 1))
        else:
            self.register_parameter("relative_bias", None)

    def _rotate(self, tensor, positions):
        # Compute frequencies at FP32 even when the model itself is FP16/BF16.
        frequencies = self.rope_theta ** (
            -torch.arange(0, self.qk_head_dim, 2, device=tensor.device, dtype=torch.float32) / self.qk_head_dim
        )
        angles = positions.float()[:, None] * frequencies[None]
        cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]
        even, odd = tensor.float()[..., 0::2], tensor.float()[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(tensor.dtype)

    def forward(self, query, source, centers, source_start, source_mask, query_mask):
        batch, length, _ = query.shape
        q = self.q_proj(query).reshape(batch, length, self.heads, self.qk_head_dim)
        k = self.k_proj(source).reshape(batch, -1, self.heads, self.qk_head_dim)
        v = self.v_proj(source).reshape(batch, -1, self.heads, self.head_dim)
        if self.position_encoding == "rope":
            q = self._rotate(q, centers)
            key_positions = torch.arange(source_start, source_start + source.shape[1], device=query.device)
            k = self._rotate(k, key_positions)
        backend = "eager" if self.config is None else self.config._attn_implementation
        if backend == "flash_attention_2":
            if self.position_encoding != "rope":
                raise ValueError("FlashAttention requires position_encoding='rope'; learned relative_bias needs eager")
            if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("FlashAttention-2 requires CUDA Q/K/V in float16 or bfloat16; use model.to() or autocast")
            result = _flash_local_attention(
                q, k, v, centers, source_start, source_mask, query_mask,
                self.window, self.dropout if self.training else 0.0,
                _load_flash_attention(),
            )
            return self.o_proj(result.reshape(batch, length, -1))
        if backend not in (None, "eager"):
            raise ValueError(f"Unsupported attention backend: {backend}")
        offsets = torch.arange(-self.window, 1, device=query.device)
        indices = centers[:, None] + offsets[None, :] - source_start
        in_bounds = (indices >= 0) & (indices < source.shape[1])
        indices = indices.clamp(0, source.shape[1] - 1)
        allowed = in_bounds[None] & source_mask[:, indices] & query_mask[:, :, None]
        k, v = k[:, indices], v[:, indices]
        scores = torch.einsum("bqhd,bqwhd->bqhw", q.float(), k.float()) / math.sqrt(self.qk_head_dim)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias.float()[None, None]
        scores = scores.masked_fill(~allowed[:, :, None], torch.finfo(scores.dtype).min)
        # An all-masked neighborhood returns zero, including fully padded rows.
        weights = scores.softmax(-1) * allowed[:, :, None]
        weights = F.dropout(weights, self.dropout, self.training).to(v.dtype)
        result = torch.einsum("bqhw,bqwhd->bqhd", weights, v).reshape(batch, length, -1)
        return self.o_proj(result)


@lru_cache(maxsize=1)
def _load_flash_attention():
    if Version(version("flash-attn")) < Version("2.6.0"):
        raise ImportError("TraceRelay requires flash-attn>=2.6.0 for its FlashAttention-2 backend")
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func


def _contiguous_mask_bounds(mask):
    """Return [start, end) per row; do not compress physical-time mask holes."""
    count = mask.sum(-1)
    start = mask.int().argmax(-1)
    end = mask.shape[1] - mask.flip(-1).int().argmax(-1)
    start = torch.where(count > 0, start, 0)
    end = torch.where(count > 0, end, 0)
    if not torch.all(end - start == count):
        raise ValueError("FlashAttention supports contiguous left/right padding only; use eager for internal mask holes")
    return torch.stack((start, end), -1).tolist()


def _flash_local_attention(q, k, v, centers, source_start, source_mask, query_mask, window, dropout, kernel):
    """Pack native local windows without expanding K/V or adding dummy queries.

    FA2 aligns unequal Q/K lengths at the bottom right. Trim each row's keys
    to end at its last real query center, then use the causal window (window, 0).
    RoPE still uses the original physical centers.
    The kernel parameter permits a dense oracle to test packing on CPU.
    """
    softmax_scale = q.shape[-1] ** -0.5
    # FA2 requires equal Q/K/V widths. Pad after RoPE and keep the Q/K scale.
    padding = v.shape[-1] - q.shape[-1]
    if padding:
        q, k = F.pad(q, (0, padding)), F.pad(k, (0, padding))
    if not torch.equal(centers, centers[0] + torch.arange(q.shape[1], device=q.device)):
        raise ValueError("FlashAttention query centers must be consecutive physical positions")
    center_start = int(centers[0])
    key_bounds = _contiguous_mask_bounds(source_mask)
    query_bounds = _contiguous_mask_bounds(query_mask)
    spans = []
    for batch, ((k0, k1), (q0, q1)) in enumerate(zip(key_bounds, query_bounds)):
        if q0 == q1 or k0 == k1:
            continue
        first_center = center_start + q0 - source_start
        last_center = center_start + q1 - 1 - source_start
        if first_center < k0 or last_center >= k1:
            raise ValueError("Unmasked query centers must refer to unmasked source positions")
        k0 = max(k0, first_center - window)
        k1 = last_center + 1
        spans.append((batch, q0, q1, k0, k1))
    # Keep zero gradients to all projections even for an all-padding batch.
    result = q * 0 + (k.float().sum() * 0 + v.float().sum() * 0).to(q.dtype)
    if spans:
        queries, keys, values = [], [], []
        cu_q, cu_k = [0], [0]
        for batch, q0, q1, k0, k1 in spans:
            queries.append(q[batch, q0:q1])
            keys.append(k[batch, k0:k1])
            values.append(v[batch, k0:k1])
            cu_q.append(cu_q[-1] + q1 - q0)
            cu_k.append(cu_k[-1] + k1 - k0)
        packed = kernel(
            torch.cat(queries), torch.cat(keys), torch.cat(values),
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=q.device),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=q.device),
            max_seqlen_q=max(b - a for a, b in zip(cu_q, cu_q[1:])),
            max_seqlen_k=max(b - a for a, b in zip(cu_k, cu_k[1:])),
            dropout_p=dropout, softmax_scale=softmax_scale,
            causal=True, window_size=(window, 0),
        )
        for (batch, q0, q1, _, _), packed_start in zip(spans, cu_q):
            result[batch, q0:q1] = packed[packed_start:packed_start + q1 - q0]
    return result


@dataclass
class TraceRelayLayerState:
    # Histories hold attention inputs after their respective RMSNorms.
    raw_hidden: torch.Tensor
    memory_hidden: torch.Tensor
    mask: torch.Tensor
    phases: torch.Tensor
    phase_valid: torch.Tensor
    # Uncorrected FP32 increments have their own window, independent of reads
    # and writes. None means this layer has no delta attention module.
    raw_delta: torch.Tensor | None = None
    delta_mask: torch.Tensor | None = None


class TraceRelayCache(Cache):
    """Bounded local buffers and the last write_window + 1 phases per layer.

    This mutable cache belongs to one sequence batch.
    Call detach() for truncated backpropagation.
    """

    def __init__(self, config):
        super().__init__(layers=[])
        self.signature = self.config_signature(config)
        self.states = [None] * config.num_hidden_layers
        self.seen_tokens = 0

    @staticmethod
    def config_signature(config):
        return (config.hidden_size, config.intermediate_size, config.num_hidden_layers,
                config.num_attention_heads, config.position_encoding, config.rope_theta,
                config.phase_update_scale, config.rms_norm_eps, config.recurrent_carry, config.relay_enabled,
                config.phase_correction_scale, config.delta_detach_inputs,
                *(tuple(getattr(config, name)) for name in (
            "read_window", "write_window", "read_qk_head_dim", "write_qk_head_dim",
            "delta_window", "delta_qk_head_dim",
        )))

    @property
    def is_compileable(self):
        return False

    @property
    def is_croppable(self):
        return False

    @property
    def is_initialized(self):
        return self.seen_tokens > 0

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def get_max_cache_shape(self, layer_idx=0):
        return -1

    def reorder_cache(self, beam_idx):
        for state in self.states:
            if state is not None:
                for field in fields(state):
                    tensor = getattr(state, field.name)
                    if tensor is not None:
                        setattr(state, field.name, tensor.index_select(0, beam_idx.to(tensor.device)))

    def batch_select_indices(self, indices):
        self.reorder_cache(indices)

    def batch_repeat_interleave(self, repeats):
        for state in self.states:
            if state is not None:
                for field in fields(state):
                    tensor = getattr(state, field.name)
                    if tensor is not None:
                        setattr(state, field.name, tensor.repeat_interleave(repeats, dim=0))

    def detach(self):
        for state in self.states:
            if state is not None:
                for field in fields(state):
                    tensor = getattr(state, field.name)
                    if tensor is not None:
                        setattr(state, field.name, tensor.detach())
        return self

    def reset(self):
        self.states = [None] * len(self.states)
        self.seen_tokens = 0

    def crop(self, max_length):
        raise NotImplementedError("TraceRelay cannot roll back discarded history; start a fresh cache")


class TraceRelayLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        h = config.hidden_size
        heads = config.num_attention_heads
        self.config = config
        self.relay_enabled = config.relay_enabled
        self.recurrent_carry = config.recurrent_carry
        self.stride = config.write_window[layer_idx] + 1
        self.history_size = max(config.read_window[layer_idx],
                                config.write_window[layer_idx] if self.relay_enabled else 0)
        self.scale = config.phase_update_scale
        if self.relay_enabled:
            self.write_input_norm = RMSNorm(h, config.rms_norm_eps)
        self.read_input_norm = RMSNorm(h, config.rms_norm_eps)
        self.read_attention = LocalAttention(
            h, heads,
            config.read_window[layer_idx], config.attention_dropout, config=config,
            qk_head_dim=config.read_qk_head_dim[layer_idx],
        )
        if self.relay_enabled:
            self.write_attention = LocalAttention(
                h, heads,
                config.write_window[layer_idx], config.attention_dropout, config=config,
                qk_head_dim=config.write_qk_head_dim[layer_idx],
            )
            self.write_phase = nn.Linear(h, h, bias=False)
        self.read_output = nn.Linear(h, h, bias=False)
        if self.relay_enabled:
            self.phase_output = nn.Linear(2 * h, h, bias=False)
        self.post_norm = RMSNorm(h, config.rms_norm_eps)
        ff = config.intermediate_size
        self.gate_proj = nn.Linear(h, ff, bias=False)
        self.up_proj = nn.Linear(h, ff, bias=False)
        self.down_proj = nn.Linear(ff, h, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        # Construct last so the existing layer initialization order is retained
        # when correction is disabled (including old checkpoints).
        self.delta_attention = (
            LocalAttention(h, heads, config.delta_window[layer_idx],
                           config.attention_dropout, config=config,
                           qk_head_dim=config.delta_qk_head_dim[layer_idx])
            if self.relay_enabled and config.delta_window[layer_idx] > 0 else None
        )

    def forward(self, hidden, mask, offset, state=None):
        if not self.relay_enabled:
            return self._forward_swa(hidden, mask, offset, state)
        raw = self.write_input_norm(hidden)
        if state is None:
            raw_source, source_mask = raw, mask
            old_length = 0
        else:
            raw_source = torch.cat((state.raw_hidden, raw), dim=1)
            source_mask = torch.cat((state.mask, mask), dim=1)
            old_length = state.raw_hidden.shape[1]
        source_start = offset - old_length
        positions = torch.arange(offset, offset + hidden.shape[1], device=hidden.device)

        # Write at t attends [t-write_window, t] and enters phase[t] immediately.
        # Q/K/V use lower-layer inputs, independent of this layer's phase/read path.
        write = self.write_attention(raw, raw_source, positions, source_start, source_mask, mask)
        write_delta = self.scale * self.write_phase(write).float().tanh()
        write_delta = write_delta * mask[:, :, None]
        delta_tail = delta_mask_tail = None
        if self.delta_attention is not None:
            # Attend only raw writes: corrected deltas never become attention
            # sources. This preserves parallel computation and direct gradients
            # into earlier write operations, including across cached chunks.
            delta_source = (write_delta if state is None else
                            torch.cat((state.raw_delta, write_delta), dim=1))
            delta_mask = mask if state is None else torch.cat((state.delta_mask, mask), dim=1)
            delta_start = offset + hidden.shape[1] - delta_source.shape[1]
            # Stop only the auxiliary input-gradient route. The uncorrected
            # write still enters the carry, and the attention weights train.
            delta_query = write_delta.detach() if self.config.delta_detach_inputs else write_delta
            attention_source = delta_source.detach() if self.config.delta_detach_inputs else delta_source
            correction = self.delta_attention(
                delta_query.to(hidden.dtype), attention_source.to(hidden.dtype),
                positions, delta_start, delta_mask, mask,
            )
            write_delta = write_delta + (
                self.config.phase_correction_scale * correction.float().tanh() * mask[:, :, None]
            )
            delta_tail = delta_source[:, -self.delta_attention.window:].clone()
            delta_mask_tail = delta_mask[:, -self.delta_attention.window:].clone()
        if self.recurrent_carry:
            phases, tail = phase_stride_scan(
                write_delta, self.stride, None if state is None else state.phases,
            )
            phase_valid, valid_tail = phase_validity_stride_scan(
                mask, self.stride, None if state is None else state.phase_valid,
            )
        else:
            # Keep immediate writes and local paths while ablating phase carry.
            phases, phase_valid = wrap_phase(write_delta), mask
            previous = (write_delta.new_zeros(hidden.shape[0], self.stride, hidden.shape[-1])
                        if state is None else state.phases)
            previous_valid = (mask.new_zeros(hidden.shape[0], self.stride)
                              if state is None else state.phase_valid)
            tail = torch.cat((previous, phases), dim=1)[:, -self.stride:]
            valid_tail = torch.cat((previous_valid, phase_valid), dim=1)[:, -self.stride:]
        features = torch.cat((phases.cos(), phases.sin()), dim=-1).to(hidden.dtype)
        features = features * phase_valid[:, :, None]
        # Expose the current write and accumulated phase before read Q/K/V.
        # phase_output has no bias, so uninitialized memory contributes zero.
        memory_hidden = hidden + self.dropout(self.phase_output(features))
        memory_for_attention = self.read_input_norm(memory_hidden)
        read_source = (memory_for_attention if state is None else
                       torch.cat((state.memory_hidden, memory_for_attention), dim=1))
        read = self.read_attention(memory_for_attention, read_source, positions, source_start, source_mask, mask)
        hidden = memory_hidden + self.dropout(self.read_output(read))
        normalized = self.post_norm(hidden)
        hidden = hidden + self.dropout(self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)))
        hidden = hidden * mask[:, :, None]
        # clone() prevents a small cached view from retaining an entire prefill
        # allocation under no_grad. Graphs remain intact during training.
        start = raw_source.shape[1] - min(self.history_size, raw_source.shape[1])
        next_state = TraceRelayLayerState(
            raw_source[:, start:].clone(),
            read_source[:, start:].clone(),
            source_mask[:, start:].clone(),
            tail.clone(),
            valid_tail.clone(),
            delta_tail,
            delta_mask_tail,
        )
        return hidden, next_state

    def _forward_swa(self, hidden, mask, offset, state):
        """The same read/FFN path, without write modules or phase storage."""
        memory_for_attention = self.read_input_norm(hidden)
        old_length = 0 if state is None else state.memory_hidden.shape[1]
        read_source = (memory_for_attention if state is None else
                       torch.cat((state.memory_hidden, memory_for_attention), dim=1))
        source_mask = mask if state is None else torch.cat((state.mask, mask), dim=1)
        positions = torch.arange(offset, offset + hidden.shape[1], device=hidden.device)
        read = self.read_attention(memory_for_attention, read_source, positions, offset - old_length, source_mask, mask)
        hidden = hidden + self.dropout(self.read_output(read))
        normalized = self.post_norm(hidden)
        hidden = hidden + self.dropout(self.down_proj(F.silu(self.gate_proj(normalized)) * self.up_proj(normalized)))
        hidden = hidden * mask[:, :, None]
        # Explicit start indexing handles L=0; ``[-0:]`` would keep everything.
        start = read_source.shape[1] - min(self.history_size, read_source.shape[1])
        next_state = TraceRelayLayerState(
            hidden[:, :0].clone(),
            read_source[:, start:].clone(),
            source_mask[:, start:].clone(),
            hidden.new_empty(hidden.shape[0], 0, hidden.shape[-1],
                             dtype=torch.float32),
            mask[:, :0].clone(),
        )
        return hidden, next_state


class TraceRelayPreTrainedModel(PreTrainedModel):
    config_class = TraceRelayConfig
    base_model_prefix = "model"
    _no_split_modules = ["TraceRelayLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_attention_backend = True

    @classmethod
    def _can_set_attn_implementation(cls):
        # LocalAttention reads the shared config on every call, so switching
        # needs no module replacement despite not using HF's mask dispatcher.
        return True

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, **kwargs):
        # Avoid HF's automatic kernel substitution: this backend explicitly
        # uses the flash-attn package and its FA2 varlen/local-window contract.
        if attn_implementation in (None, "eager"):
            return "eager"
        if attn_implementation != "flash_attention_2":
            raise ValueError("TraceRelay supports attn_implementation='eager' or 'flash_attention_2'")
        if self.config.position_encoding != "rope":
            raise ValueError("FlashAttention requires a RoPE checkpoint; relative_bias checkpoints must use eager")
        if self.config.v_head_dim > 256:
            raise ValueError("FlashAttention-2 requires a per-head hidden dimension <= 256")
        self._flash_attn_can_dispatch(2, is_init_check=is_init_check)
        _load_flash_attention()
        return "flash_attention_2"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, LocalAttention) and module.relative_bias is not None:
            nn.init.zeros_(module.relative_bias)


class TraceRelayModel(TraceRelayPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([TraceRelayLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None,
        inputs_embeds=None, use_cache=None, output_hidden_states=None,
        output_attentions=None, return_dict=None, position_ids=None,
        cache_position=None, **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported model arguments: {sorted(kwargs)}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        if output_attentions:
            raise NotImplementedError("Attention weight outputs are not implemented")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if past_key_values is not None and not use_cache:
            raise ValueError("Continuing a supplied cache requires use_cache=True")
        if past_key_values is not None and not isinstance(past_key_values, TraceRelayCache):
            raise TypeError("past_key_values must be a TraceRelayCache")
        cache = past_key_values if past_key_values is not None else (TraceRelayCache(self.config) if use_cache else None)
        if cache is not None and cache.signature != TraceRelayCache.config_signature(self.config):
            raise ValueError("Cache configuration does not match this model")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or hidden.shape[1] == 0:
            raise ValueError("Inputs must have a nonempty sequence and the configured hidden_size")
        batch, length = hidden.shape[:2]
        offset = 0 if cache is None else cache.seen_tokens
        if attention_mask is None:
            mask = torch.ones(batch, length, device=hidden.device, dtype=torch.bool)
        else:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch or attention_mask.shape[1] not in (length, offset + length):
                raise ValueError("attention_mask must cover the new tokens or the full cached prefix plus new tokens")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only zero and one")
            mask = attention_mask[:, -length:].to(device=hidden.device, dtype=torch.bool)
        # Position IDs are accepted for HF call compatibility. Relative offsets
        # and the physical token clock alone define this model's positions.
        if position_ids is not None and (position_ids.ndim != 2 or position_ids.shape[-1] != length or position_ids.shape[0] not in (1, batch)):
            raise ValueError("position_ids must have shape (1 or batch, new_sequence_length)")
        if cache is not None and offset:
            for state in cache.states:
                if state is None or any(history.shape[0] != batch or history.device != hidden.device or
                                        history.dtype != hidden.dtype
                                        for history in (state.raw_hidden, state.memory_hidden)):
                    raise ValueError("Cache batch, device, or dtype changed; create a fresh cache")
        hidden = hidden * mask[:, :, None]
        history = () if output_hidden_states else None
        next_states = []
        for index, layer in enumerate(self.layers):
            if output_hidden_states:
                history += (hidden,)
            hidden, state = layer(hidden, mask, offset, None if cache is None else cache.states[index])
            next_states.append(state)
        hidden = self.norm(hidden) * mask[:, :, None]
        if output_hidden_states:
            history += (hidden,)
        if cache is not None:
            cache.states = next_states
            cache.seen_tokens += length
        result = BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=cache, hidden_states=history)
        return result if return_dict else result.to_tuple()


class TraceRelayForCausalLM(TraceRelayPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = TraceRelayModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, embeddings):
        self.lm_head = embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self, input_ids=None, attention_mask=None, past_key_values=None,
        inputs_embeds=None, labels=None, use_cache=None, output_hidden_states=None,
        output_attentions=None, return_dict=None, position_ids=None,
        cache_position=None, logits_to_keep=0, num_items_in_batch=None, **kwargs,
    ):
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if labels is not None and logits_to_keep:
            raise ValueError("Training with labels requires logits_to_keep=0")
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, return_dict=True, position_ids=position_ids,
            cache_position=cache_position, **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state[:, -logits_to_keep:])
        loss = None
        if labels is not None:
            if labels.shape != outputs.last_hidden_state.shape[:2]:
                raise ValueError("labels must have the same batch and sequence dimensions as the new input")
            targets = labels[:, 1:].to(logits.device).clone()
            if attention_mask is not None:
                valid = attention_mask[:, -labels.shape[1]:].to(logits.device).bool()
                targets.masked_fill_(~(valid[:, :-1] & valid[:, 1:]), -100)
            flat_logits = logits[:, :-1].float().reshape(-1, self.config.vocab_size)
            total = F.cross_entropy(flat_logits, targets.reshape(-1), ignore_index=-100, reduction="sum")
            denominator = (targets != -100).sum() if num_items_in_batch is None else torch.as_tensor(num_items_in_batch, device=logits.device)
            loss = total / denominator.clamp_min(1)
        result = CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        return result if return_dict else result.to_tuple()

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, generation_mode=None, *args, **kwargs):
        mode = generation_mode if generation_mode is not None else generation_config.get_generation_mode()
        if mode.value not in ("greedy_search", "sample"):
            raise NotImplementedError("TraceRelay generate supports greedy decoding and sampling")
        if generation_config.cache_implementation is not None:
            raise ValueError("TraceRelay uses its own cache; omit cache_implementation")
        supplied = model_kwargs.get("past_key_values")
        if supplied is not None:
            if not isinstance(supplied, TraceRelayCache) or not generation_config.use_cache:
                raise ValueError("Supply a TraceRelayCache with use_cache=True")
            if generation_config.num_return_sequences != 1:
                raise ValueError("A supplied cache requires num_return_sequences=1")
            supplied._is_user_defined = True
        elif generation_config.use_cache:
            model_kwargs["past_key_values"] = TraceRelayCache(self.config)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, use_cache=True, next_sequence_length=None,
        is_first_iteration=False, **kwargs,
    ):
        if next_sequence_length is not None:
            if next_sequence_length <= 0:
                raise ValueError("Generation requires at least one unprocessed input token")
            input_ids = input_ids[:, -next_sequence_length:]
        model_inputs = {"input_ids": input_ids}
        if inputs_embeds is not None and is_first_iteration:
            embeds = inputs_embeds if next_sequence_length is None else inputs_embeds[:, -next_sequence_length:]
            model_inputs = {"inputs_embeds": embeds}
        model_inputs.update(attention_mask=attention_mask, past_key_values=past_key_values, use_cache=use_cache)
        for name in ("logits_to_keep", "output_hidden_states", "output_attentions"):
            if name in kwargs:
                model_inputs[name] = kwargs[name]
        return model_inputs


TraceRelayModel.register_for_auto_class("AutoModel")
TraceRelayForCausalLM.register_for_auto_class("AutoModelForCausalLM")
