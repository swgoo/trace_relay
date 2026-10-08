"""Independent one-hot adapter and unconstrained EOS decoding over the reference core."""
import copy
import hashlib
import io
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from configuration_trace_relay import TraceRelayConfig
from modeling_trace_relay import TraceRelayModel, TraceRelayCache
from .task import SEMANTICS_VERSION, validate_source, output_to_input
from .utils import choose_backend

CHECKPOINT_FORMAT = 'trace-relay-most-freq-v1'


def build_config(*, vocab_size=4, model='trace_relay', hidden_size=64, intermediate_size=256, heads=4,
                 center_width=64, trace_widths=None, left_windows=(15,15,15), right_windows=(7,7,7),
                 relay_strides=(8,8,8), skip_pairs=(), device='cpu', precision='auto', attn_implementation='auto'):
    if type(vocab_size) is not int or vocab_size < 1 or model not in ('trace_relay','no_carry','swa'):
        raise ValueError('Invalid vocabulary or model family')
    widths = [hidden_size, center_width, hidden_size] if trace_widths is None else list(trace_widths)
    def setting(values):
        values=list(values)
        if not values:raise ValueError('Window/stride lists cannot be empty')
        return values[0] if len(values)==1 else values
    core = TraceRelayConfig(vocab_size=1, hidden_size=hidden_size, intermediate_size=intermediate_size,
        num_hidden_layers=len(widths), num_attention_heads=heads, trace_width=widths,
        left_window=setting(left_windows), right_window=setting(right_windows),
        relay_stride=setting(relay_strides), skip_pairs=list(skip_pairs), layer_order='right_scan_left',
        position_encoding='rope', attention_dropout=0., hidden_dropout=0., use_cache=False,
        recurrent_carry=model!='no_carry', relay_enabled=model!='swa',
        pad_token_id=None, bos_token_id=None, eos_token_id=None)
    backend=choose_backend(device, precision, attn_implementation)
    return dict(task='most_freq', semantics_version=SEMANTICS_VERSION, family=model,
                vocab_size=vocab_size, input_vocabulary_size=vocab_size+3, output_classes=vocab_size+1,
                tie_break='first_appearance', attention_backend=backend, backbone=core.to_dict())


class MostFreqModel(nn.Module):
    def __init__(self, config, eval_chunk_size=0):
        super().__init__()
        self.config=copy.deepcopy(config)
        v=config['vocab_size']
        if (config.get('task')!='most_freq' or config.get('semantics_version')!=SEMANTICS_VERSION
            or config.get('tie_break')!='first_appearance' or config.get('input_vocabulary_size')!=v+3
            or config.get('output_classes')!=v+1 or type(v) is not int or v<1):
            raise ValueError('Most-Freq semantics/vocabulary mismatch')
        if type(eval_chunk_size) is not int or eval_chunk_size<0:
            raise ValueError('eval_chunk_size must be nonnegative')
        self.vocab_size,self.eval_chunk_size=v,eval_chunk_size
        core=TraceRelayConfig(**config['backbone'])
        family=config['family']
        if family not in ('trace_relay','no_carry','swa') or core.relay_enabled!=(family!='swa') or core.recurrent_carry!=(family!='no_carry'):
            raise ValueError('Model family and recurrence flags disagree')
        if config['attention_backend'] not in ('eager','flash_attention_2'):
            raise ValueError('Unknown backend')
        core._attn_implementation=config['attention_backend']
        self.input_proj=nn.Linear(v+3,core.hidden_size,bias=False)
        self.backbone=TraceRelayModel(core)
        self.backbone.set_input_embeddings(nn.Identity())
        self.output_head=nn.Linear(core.hidden_size,v+1)

    def _hidden(self, inputs, mask):
        if inputs.ndim!=3 or inputs.shape[-1]!=self.vocab_size+3 or min(inputs.shape[:2])<1:
            raise ValueError('Expected [batch,tokens,V+3] inputs')
        if mask.shape!=inputs.shape[:2] or not ((mask==0)|(mask==1)).all():
            raise ValueError('Binary mask must match inputs')
        # Cache histories use a stable FP32 hidden dtype under BF16 autocast, as the core requires.
        return self.input_proj(inputs.to(self.input_proj.weight.dtype)*mask[...,None]).to(self.input_proj.weight.dtype)

    def forward_cached(self, inputs, attention_mask, cache=None):
        result=self.backbone(inputs_embeds=self._hidden(inputs,attention_mask), attention_mask=attention_mask,
                             use_cache=True, past_key_values=cache)
        return self.output_head(result.last_hidden_state),result.past_key_values

    def forward(self, inputs, attention_mask):
        if not self.training and self.eval_chunk_size:
            cache=None;parts=[]
            for start in range(0,inputs.shape[1],self.eval_chunk_size):
                logits,cache=self.forward_cached(inputs[:,start:start+self.eval_chunk_size],attention_mask[:,start:start+self.eval_chunk_size],cache)
                parts.append(logits)
            return torch.cat(parts,1)
        result=self.backbone(inputs_embeds=self._hidden(inputs,attention_mask),attention_mask=attention_mask,use_cache=False)
        return self.output_head(result.last_hidden_state)

    @torch.no_grad()
    def generate(self, source, *, use_cache=True, prefill_chunk_size=0):
        validate_source(source,self.vocab_size)
        if type(prefill_chunk_size) is not int or prefill_chunk_size<0:
            raise ValueError('prefill_chunk_size must be nonnegative')
        source=source.to(self.input_proj.weight.device)
        size=source.shape[0];v=self.vocab_size
        prefix=torch.cat((source,source.new_full((size,1),v)),1)
        mask=torch.ones_like(prefix,dtype=torch.bool)
        def full(ids, valid):
            hidden=self._hidden(F.one_hot(ids,v+3).float(),valid)
            return self.output_head(self.backbone(inputs_embeds=hidden,attention_mask=valid,use_cache=False).last_hidden_state)
        prior=self.training;self.eval()
        try:
            if use_cache:
                cache=None;chunk=prefill_chunk_size or prefix.shape[1]
                for start in range(0,prefix.shape[1],chunk):
                    logits,cache=self.forward_cached(F.one_hot(prefix[:,start:start+chunk],v+3).float(),mask[:,start:start+chunk],cache)
            else:logits=full(prefix,mask)
            finished=torch.zeros(size,dtype=torch.bool,device=source.device)
            tokens=source.new_full((size,v+1),-100)
            lengths=source.new_zeros(size)
            decisions=[]
            for step in range(v+1):
                scores=logits[:,-1].float()
                if not scores.isfinite().all():raise FloatingPointError('Non-finite generation logits')
                choice=scores.argmax(-1)
                active=~finished
                tokens[active,step]=choice[active]
                lengths+=active.long()
                decisions.append(scores)
                finished=finished|(active&(choice==v))
                if finished.all() or step==v:break
                # Ended rows contribute no new valid token; keep the batch clock without oracle length.
                valid=~finished
                ids=output_to_input(choice,v).masked_fill(~valid,v+2)[:,None]
                if use_cache:
                    logits,cache=self.forward_cached(F.one_hot(ids,v+3).float(),valid[:,None],cache)
                else:
                    prefix=torch.cat((prefix,ids),1);mask=torch.cat((mask,valid[:,None]),1)
                    logits=full(prefix,mask)
            return dict(tokens=tokens, lengths=lengths, terminated=finished,
                        logits=torch.stack(decisions,1))
        finally:self.train(prior)


def save_checkpoint(path, model, **state):
    if {'format','config','model'}&state.keys():raise ValueError('Reserved checkpoint key')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(dict(format=CHECKPOINT_FORMAT,config=model.config,
        model={name:tensor.detach().cpu() for name,tensor in model.state_dict().items()},**state),temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device='cpu', precision='auto', attention_backend=None, chunk_size=0):
    raw=Path(path).read_bytes();state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if state.get('format')!=CHECKPOINT_FORMAT:raise ValueError(f'Expected {CHECKPOINT_FORMAT}')
    config=copy.deepcopy(state['config'])
    backend=config['attention_backend'] if attention_backend is None else attention_backend
    config['attention_backend']=choose_backend(device,precision,backend)
    model=MostFreqModel(config,chunk_size).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    return model,state,hashlib.sha256(raw).hexdigest()
