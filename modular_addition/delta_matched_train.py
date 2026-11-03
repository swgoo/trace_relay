"""Run the existing trainer with common weights matched across delta placements."""
import copy
import torch
from torch import nn
from . import train
from .model import create_model
from .receptive_field import direct_receptive_field

INITIALIZATION = {
    'method': 'shared-baseline-weights-and-independent-delta-stream-v2',
    'common_weights': 'Exact baseline construction under the training seed, copied into every delta variant.',
    'delta_weights': 'CPU RNG seed (training_seed + 10000019) % 2**63; reset per active layer; identical initialization with independent parameters.',
    'rng_after_initialization': 'Restored to the state after baseline construction; data RNG streams unchanged.',
}

def create_matched_model(config, eval_chunk_size=0):
    windows = config['backbone']['delta_window']
    if not any(windows):
        return create_model(config, eval_chunk_size)
    seed = torch.initial_seed()
    base_config = copy.deepcopy(config)
    core = base_config['backbone']
    core['delta_window'] = [0] * core['num_hidden_layers']
    base_config['direct_receptive_field'] = direct_receptive_field(core['read_window'], core['write_window'])
    baseline = create_model(base_config, eval_chunk_size)
    # Model initialization uses CPU tensors. Restore its RNG to the same state
    # as the baseline; this does not rewind or reseed any CUDA RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = create_model(config, eval_chunk_size)
        missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
        assert missing and not unexpected and all('.delta_attention.' in key for key in missing)
        for layer in model.backbone.layers:
            if layer.delta_attention is not None:
                torch.random.default_generator.manual_seed((seed + 10000019) % 2**63)
                for module in layer.delta_attention.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0., std=core['initializer_range'])
    return model

def main():
    train.create_model = create_matched_model
    original = train.provenance
    def provenance():
        return dict(original(), initialization=INITIALIZATION)
    train.provenance = provenance
    train.main()

if __name__ == '__main__':
    main()
