"""One-hot bracket input and causal per-token close-type head around unchanged TraceRelay."""
import hashlib
import io
from pathlib import Path

import torch
from torch import nn

from configuration_trace_relay import TraceRelayConfig
from modeling_trace_relay import TraceRelayModel
from .task import SEMANTICS_VERSION, validate_language, vocabulary

CHECKPOINT_FORMAT = 'trace-relay-dyck-v1'


class TraceRelayDyckModel(nn.Module):
    def __init__(self, config, eval_chunk_size=0):
        super().__init__()
        k, m = config['bracket_types'], config['max_depth']
        validate_language(k, m)
        if config.get('vocabulary') != vocabulary(k) or config.get('semantics_version') != SEMANTICS_VERSION:
            raise ValueError('Dyck vocabulary/semantics mismatch')
        if eval_chunk_size < 0:
            raise ValueError('eval_chunk_size must be nonnegative')
        self.config, self.eval_chunk_size = dict(config), eval_chunk_size
        backbone_config = TraceRelayConfig(**config['backbone'])
        backbone_config._attn_implementation = config['attention_backend']
        self.input_proj = nn.Linear(2*k, backbone_config.hidden_size, bias=False)
        self.backbone = TraceRelayModel(backbone_config)
        self.backbone.set_input_embeddings(nn.Identity())
        self.close_head = nn.Linear(backbone_config.hidden_size, k)

    def forward(self, inputs, attention_mask):
        if inputs.ndim != 3 or inputs.shape[-1] != 2*self.config['bracket_types'] or min(inputs.shape[:2]) < 1:
            raise ValueError('Expected [batch, nonempty tokens, 2*k] one-hot input')
        if attention_mask.shape != inputs.shape[:2] or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError('Attention mask must be binary and match input shape')
        mask = attention_mask.bool()
        lengths = mask.long().sum(-1)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None, :]
        if torch.any(lengths == 0) or not torch.equal(mask, positions < lengths[:, None]):
            raise ValueError('Nonempty sequences and right padding are required')
        hidden = self.input_proj(inputs.to(self.input_proj.weight.dtype)*mask[..., None])
        # Cache histories and residuals retain the same dtype across prefill and continuation.
        hidden = hidden.to(self.input_proj.weight.dtype)
        if self.training or not self.eval_chunk_size:
            output = self.backbone(inputs_embeds=hidden, attention_mask=mask, use_cache=False).last_hidden_state
        else:
            cache, outputs = None, []
            for start in range(0, hidden.shape[1], self.eval_chunk_size):
                stop = min(start+self.eval_chunk_size, hidden.shape[1])
                result = self.backbone(inputs_embeds=hidden[:, start:stop], attention_mask=mask[:, start:stop],
                                       past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                outputs.append(result.last_hidden_state)
            output = torch.cat(outputs, dim=1)
        return self.close_head(output)


def save_dyck_checkpoint(path, model, **metadata):
    if {'format', 'config', 'model'} & metadata.keys():
        raise ValueError('Reserved checkpoint metadata key')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(format=CHECKPOINT_FORMAT, config=model.config,
                    model={k:v.detach().cpu() for k,v in model.state_dict().items()}, **metadata), temporary)
    temporary.replace(path)


def load_dyck_checkpoint(path, device='cpu', chunk_size=0, attention_backend=None):
    data = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'Expected {CHECKPOINT_FORMAT} checkpoint')
    config = dict(checkpoint['config'])
    if attention_backend is not None:
        if attention_backend not in ('eager', 'flash_attention_2'):
            raise ValueError('Unsupported attention backend')
        config['attention_backend'] = attention_backend
    model = TraceRelayDyckModel(config, chunk_size)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint, hashlib.sha256(data).hexdigest()
