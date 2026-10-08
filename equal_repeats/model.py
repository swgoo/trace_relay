"""Two-symbol one-hot / three-class adapter; TraceRelay core is unchanged."""
import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from configuration_trace_relay import TraceRelayConfig
from modeling_trace_relay import TraceRelayModel
from .task import SYMBOLS, SEMANTICS_VERSION

CHECKPOINT_FORMAT = 'trace-relay-equal-repeats-v1'


class TraceRelayEqualRepeatsModel(nn.Module):
    def __init__(self, config, eval_chunk_size=0):
        super().__init__()
        if config.get('vocabulary') != list(SYMBOLS) or config.get('semantics_version') != SEMANTICS_VERSION:
            raise ValueError('Equal Repeats vocabulary/semantics mismatch')
        if eval_chunk_size < 0:
            raise ValueError('eval_chunk_size must be nonnegative')
        self.config, self.eval_chunk_size = dict(config), eval_chunk_size
        backbone_config = TraceRelayConfig(**config['backbone'])
        backbone_config._attn_implementation = config['attention_backend']
        self.input_proj = nn.Linear(2, backbone_config.hidden_size, bias=False)
        self.backbone = TraceRelayModel(backbone_config)
        self.backbone.set_input_embeddings(nn.Identity())
        self.classifier = nn.Linear(backbone_config.hidden_size, 3)

    def forward(self, inputs, attention_mask):
        if inputs.ndim != 3 or inputs.shape[-1] != 2 or min(inputs.shape[:2]) < 1:
            raise ValueError('Expected [batch, nonempty tokens, 2] one-hot input')
        if attention_mask.shape != inputs.shape[:2] or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError('Attention mask must be binary and match input shape')
        mask = attention_mask.bool()
        lengths = mask.long().sum(-1)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
        if torch.any(lengths == 0) or not torch.equal(mask, positions < lengths[:, None]):
            raise ValueError('Nonempty sequences and right padding are required')
        hidden = self.input_proj(inputs.to(self.input_proj.weight.dtype)*mask[..., None])
        hidden = hidden.to(self.input_proj.weight.dtype)
        batch_index, last = torch.arange(len(inputs), device=inputs.device), lengths-1
        if self.training or not self.eval_chunk_size:
            output = self.backbone(inputs_embeds=hidden, attention_mask=mask, use_cache=False).last_hidden_state
            pooled = output[batch_index, last]
        else:
            cache = None
            pooled = hidden.new_zeros((len(inputs), hidden.shape[-1]))
            for start in range(0, hidden.shape[1], self.eval_chunk_size):
                stop = min(start+self.eval_chunk_size, hidden.shape[1])
                output = self.backbone(inputs_embeds=hidden[:, start:stop], attention_mask=mask[:, start:stop],
                                       past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                selected = output.last_hidden_state[batch_index, (last-start).clamp(0, stop-start-1)]
                pooled = torch.where(((last >= start) & (last < stop))[:, None], selected, pooled)
        return self.classifier(pooled)


def save_equal_repeats_checkpoint(path, model, **metadata):
    if {'format', 'config', 'model'} & metadata.keys():
        raise ValueError('Reserved checkpoint metadata key')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(format=CHECKPOINT_FORMAT, config=model.config,
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()}, **metadata), temporary)
    temporary.replace(path)


def load_equal_repeats_checkpoint(path, device='cpu', chunk_size=0, attention_backend=None):
    data = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'Expected {CHECKPOINT_FORMAT} checkpoint')
    config = dict(checkpoint['config'])
    if attention_backend is not None:
        if attention_backend not in ('eager', 'flash_attention_2'):
            raise ValueError('Unsupported attention backend')
        config['attention_backend'] = attention_backend
    model = TraceRelayEqualRepeatsModel(config, chunk_size)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint, hashlib.sha256(data).hexdigest()
