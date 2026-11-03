"""One-hot, aligned-state adapter around the unchanged TraceRelay backbone.

Each input row contains only the current modular-addition operand. The backbone
keeps its existing input and output normalization, causal write/read order, and
phasor recurrence. Targets are consumed only by ``aligned_loss``.
"""
import copy
from contextlib import contextmanager
import hashlib
import io
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from ..configuration_trace_relay import TraceRelayConfig
    from ..modeling_trace_relay import TraceRelayModel
except ImportError:
    from configuration_trace_relay import TraceRelayConfig
    from modeling_trace_relay import TraceRelayModel
from .utils import choose_backend
from .receptive_field import direct_receptive_field


EXPERIMENT_VERSION = 'modular-addition-full-qk-phasor-v1'
CHECKPOINT_FORMAT = 'trace-relay-modular-addition-v1'


def create_model(config, eval_chunk_size=0):
    return ModularAdditionModel(config, eval_chunk_size)


def normalize_config(config):
    """Supply disabled-delta defaults when comparing/loading older checkpoints."""
    result = copy.deepcopy(config)
    if result.get('experiment_version') == EXPERIMENT_VERSION:
        core = TraceRelayConfig(**result['backbone'])
        for name in ('delta_window', 'delta_qk_head_dim', 'phase_correction_scale', 'delta_detach_inputs'):
            result['backbone'].setdefault(name, getattr(core, name))
    return result


def _direct_field(core):
    return direct_receptive_field(core.read_window, core.write_window,
        delta_windows=core.delta_window if core.phase_correction_scale else None)


def build_config(*, modulus=5, hidden_size=128, intermediate_size=512, heads=4, layers=3,
                 read_windows=5, write_windows=5, delta_windows=0,
                 phase_correction_scale=math.pi, delta_detach_inputs=False,
                 device='cpu', precision='auto', attn_implementation='auto'):
    """Build recurrent layers with full read/write Q/K dimensions.

    A core window of five counts five *past* positions plus the current one.
    Thus the default phase inheritance stride is six, and removing inheritance
    leaves the default three-layer direct field at width 31 (radius 30).
    A scalar window is shared by all layers; a list must match the layer count.
    """
    if type(modulus) is not int or modulus < 2:
        raise ValueError('modulus must be an integer >= 2')
    core = TraceRelayConfig(vocab_size=1, hidden_size=hidden_size,
        intermediate_size=intermediate_size, num_hidden_layers=layers,
        num_attention_heads=heads, read_window=read_windows,
        write_window=write_windows, read_qk_head_dim=None, write_qk_head_dim=None,
        delta_window=delta_windows, phase_correction_scale=phase_correction_scale,
        delta_detach_inputs=delta_detach_inputs,
        position_encoding='rope', attention_dropout=0., hidden_dropout=0.,
        use_cache=False, recurrent_carry=True, relay_enabled=True,
        pad_token_id=None, bos_token_id=None, eos_token_id=None)
    return dict(task='modular_addition', experiment_version=EXPERIMENT_VERSION,
                modulus=modulus, initial_state=0, input_encoding='one_hot',
                input_dimension=modulus, output_classes=modulus,
                class_order=list(range(modulus)), supervision='aligned_cot',
                attention_backend=choose_backend(device, precision, attn_implementation),
                direct_receptive_field=_direct_field(core),
                backbone=core.to_dict())


class ModularAdditionModel(nn.Module):
    """Project operands, execute TraceRelay, and predict every aligned state."""

    def __init__(self, config, eval_chunk_size=0):
        super().__init__()
        self.config = normalize_config(config)
        modulus = config.get('modulus')
        if (config.get('task') != 'modular_addition' or
                config.get('experiment_version') != EXPERIMENT_VERSION or
                type(modulus) is not int or modulus < 2 or
                config.get('initial_state') != 0 or config.get('input_encoding') != 'one_hot' or
                config.get('input_dimension') != modulus or config.get('output_classes') != modulus or
                config.get('class_order') != list(range(modulus)) or
                config.get('supervision') != 'aligned_cot'):
            raise ValueError('Modular Addition semantics/input/output mismatch')
        if type(eval_chunk_size) is not int or eval_chunk_size < 0:
            raise ValueError('eval_chunk_size must be a nonnegative integer')
        core = TraceRelayConfig(**config['backbone'])
        if (not core.relay_enabled or not core.recurrent_carry or
                core.read_qk_head_dim != [core.v_head_dim] * core.num_hidden_layers or
                core.write_qk_head_dim != [core.v_head_dim] * core.num_hidden_layers or
                any(dw and qk != core.v_head_dim for dw, qk in
                    zip(core.delta_window, core.delta_qk_head_dim))):
            raise ValueError('Expected recurrent layers and full read/write Q/K dimensions')
        if config.get('direct_receptive_field') != _direct_field(core):
            raise ValueError('Direct receptive field disagrees with backbone computation graph')
        if config.get('attention_backend') not in ('eager', 'flash_attention_2'):
            raise ValueError('Unknown attention backend')
        core._attn_implementation = config['attention_backend']
        self.modulus, self.eval_chunk_size = modulus, eval_chunk_size
        # Same bias-free one-hot projection and default Linear initialization as
        # Most-Freq; normalization is supplied by the unchanged core layers.
        self.input_proj = nn.Linear(modulus, core.hidden_size, bias=False)
        self.backbone = TraceRelayModel(core)
        self.backbone.set_input_embeddings(nn.Identity())
        self.output_head = nn.Linear(core.hidden_size, modulus)

    def _prepare(self, inputs, attention_mask):
        if (inputs.ndim != 3 or not inputs.is_floating_point() or
                inputs.shape[-1] != self.modulus or min(inputs.shape[:2]) < 1):
            raise ValueError('Expected nonempty floating [batch, sequence, modulus] inputs')
        if attention_mask is None:
            attention_mask = torch.ones(inputs.shape[:2], device=inputs.device, dtype=torch.bool)
        if (attention_mask.shape != inputs.shape[:2] or
                not ((attention_mask == 0) | (attention_mask == 1)).all()):
            raise ValueError('Binary attention mask must match batch/sequence dimensions')
        attention_mask = attention_mask.to(device=inputs.device, dtype=torch.bool)
        # Keep the residual/cache dtype stable under BF16 autocast, matching
        # the preceding benchmark adapters and the core's cache contract.
        hidden = self.input_proj(inputs.to(self.input_proj.weight.dtype))
        hidden = hidden.to(self.input_proj.weight.dtype) * attention_mask[..., None]
        return hidden, attention_mask

    def forward_cached(self, inputs, attention_mask=None, cache=None):
        hidden, attention_mask = self._prepare(inputs, attention_mask)
        result = self.backbone(inputs_embeds=hidden, attention_mask=attention_mask,
                               use_cache=True, past_key_values=cache)
        return self.output_head(result.last_hidden_state), result.past_key_values

    def forward(self, inputs, attention_mask=None):
        if not self.training and self.eval_chunk_size:
            cache, parts = None, []
            for start in range(0, inputs.shape[1], self.eval_chunk_size):
                stop = start + self.eval_chunk_size
                mask = None if attention_mask is None else attention_mask[:, start:stop]
                scores, cache = self.forward_cached(inputs[:, start:stop], mask, cache)
                parts.append(scores)
            if not parts:
                raise ValueError('Expected a nonempty sequence')
            return torch.cat(parts, dim=1)
        hidden, attention_mask = self._prepare(inputs, attention_mask)
        result = self.backbone(inputs_embeds=hidden, attention_mask=attention_mask, use_cache=False)
        return self.output_head(result.last_hidden_state)

    def set_phase_correction_scale(self, scale):
        """Set runtime correction while retaining the immutable training config."""
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale < 0):
            raise ValueError('Correction scale must be nonnegative and finite')
        self.backbone.config.phase_correction_scale = float(scale)

    @contextmanager
    def temporary_phase_correction_scale(self, scale):
        previous = self.backbone.config.phase_correction_scale
        self.set_phase_correction_scale(scale)
        try:
            yield self
        finally:
            self.set_phase_correction_scale(previous)

    def effective_config(self):
        """Describe the runtime graph, including scheduled zero correction."""
        result = copy.deepcopy(self.config)
        result['backbone']['phase_correction_scale'] = self.backbone.config.phase_correction_scale
        result['direct_receptive_field'] = _direct_field(self.backbone.config)
        return result

    @contextmanager
    def temporary_recurrent_carry(self, enabled):
        """Ablate only inheritance for independent evaluation sequences.

        Immediate writes/read attention are preserved. Cache signatures reject
        reusing a cache across this change. Saved experiment config and weights
        retain the original recurrent model definition.
        """
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        previous = self.backbone.config.recurrent_carry
        previous_layers = [layer.recurrent_carry for layer in self.backbone.layers]
        self.backbone.config.recurrent_carry = enabled
        for layer in self.backbone.layers:
            layer.recurrent_carry = enabled
        try:
            yield self
        finally:
            self.backbone.config.recurrent_carry = previous
            for layer, value in zip(self.backbone.layers, previous_layers):
                layer.recurrent_carry = value


def aligned_loss(logits, targets, attention_mask=None):
    """Mean CE with equal weight at every valid input-aligned state position."""
    if (logits.ndim != 3 or targets.shape != logits.shape[:-1] or
            targets.dtype != torch.long or logits.shape[-1] < 2):
        raise ValueError('Expected [batch, sequence, classes] logits and int64 aligned targets')
    if attention_mask is None:
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if (attention_mask.shape != targets.shape or
            not ((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError('Binary attention mask must match aligned targets')
    valid = attention_mask.bool()
    if not valid.any():
        raise ValueError('At least one supervised input position is required')
    return F.cross_entropy(logits[valid].float(), targets[valid])


def estimate_flops(config, batch_size, sequence_length, training=True):
    """Approximate linear and ideal local-attention FLOPs (backward: 2x forward).

    Excludes normalization, nonlinearities, phase scan, optimizer, and padding
    work in attention kernels. This is an analytical estimate, not a profiler.
    """
    if any(type(value) is not int or value < 1 for value in (batch_size, sequence_length)):
        raise ValueError('Batch size and sequence length must be positive integers')
    core = config['backbone']
    hidden, heads, ff = core['hidden_size'], core['num_attention_heads'], core['intermediate_size']
    length = sequence_length

    def attention(window, qk):
        linear = length * (2 * hidden * heads * qk + 2 * hidden * hidden)
        past = min(window, length - 1)
        edges = length * (past + 1) - past * (past + 1) // 2
        return 2 * (linear + edges * (heads * qk + hidden))

    forward = 2 * length * hidden * (config['input_dimension'] + config['output_classes'])
    for index, rw in enumerate(core['read_window']):
        forward += attention(rw, core['read_qk_head_dim'][index])
        forward += attention(core['write_window'][index], core['write_qk_head_dim'][index])
        if core.get('delta_window', [0] * core['num_hidden_layers'])[index]:
            forward += attention(core['delta_window'][index], core['delta_qk_head_dim'][index])
        forward += 2 * length * (4 * hidden * hidden + 3 * hidden * ff)
    return int(batch_size * forward * (3 if training else 1))


def save_checkpoint(path, model, **state):
    if {'format', 'config', 'model'} & state.keys():
        raise ValueError('Reserved checkpoint key')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if isinstance(model, ModularAdditionModel):
        state['runtime_phase_correction_scale'] = model.backbone.config.phase_correction_scale
    torch.save(dict(format=CHECKPOINT_FORMAT, config=model.config,
                    model={name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                    **state), temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device='cpu', precision='auto', attention_backend=None, chunk_size=0):
    raw = Path(path).read_bytes()
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if state.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'Expected {CHECKPOINT_FORMAT}')
    state['config'] = normalize_config(state['config'])
    config = copy.deepcopy(state['config'])
    config['attention_backend'] = choose_backend(device, precision,
        config['attention_backend'] if attention_backend is None else attention_backend)
    model = create_model(config, eval_chunk_size=chunk_size).to(device).eval()
    model.load_state_dict(state['model'], strict=True)
    if isinstance(model, ModularAdditionModel) and 'runtime_phase_correction_scale' in state:
        model.set_phase_correction_scale(state['runtime_phase_correction_scale'])
    return model, state, hashlib.sha256(raw).hexdigest()
