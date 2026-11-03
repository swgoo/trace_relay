"""Vectorized, reproducible modular-addition streams and held-out pairs."""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn.functional as F


GENERATOR_VERSION = 'modular-addition.iid-uniform.v1'
SEMANTICS_VERSION = 'inclusive-prefix-cumsum-modulo.v1'


@dataclass(frozen=True)
class TaskConfig:
    modulus: int = 5
    supervision: str = 'aligned_cot'

    def __post_init__(self):
        if type(self.modulus) is not int or self.modulus < 2:
            raise ValueError('modulus must be an integer at least two')
        if self.supervision != 'aligned_cot':
            raise ValueError('Only equally weighted aligned_cot supervision is supported')

    def to_dict(self):
        return asdict(self)


def targets_from_values(values, modulus=5):
    """The label includes the current input; no BOS or target feedback is used."""
    TaskConfig(modulus)
    if values.ndim != 2 or values.dtype != torch.long:
        raise ValueError('values must be a rank-two long tensor')
    if not values.numel() or bool(((values < 0) | (values >= modulus)).any()):
        raise ValueError('values must be nonempty integers in [0, modulus)')
    return values.cumsum(dim=-1).remainder(modulus)


def batch_from_values(values, modulus=5, *, device='cpu'):
    targets = targets_from_values(values, modulus)
    return dict(values=values.to(device),
                inputs=F.one_hot(values, num_classes=modulus).float().to(device),
                targets=targets.to(device))


def make_batch(batch_size, length, generator, *, modulus=5, device='cpu'):
    """Draw exclusively from the supplied generator, including when moving to CUDA."""
    TaskConfig(modulus)
    if type(batch_size) is not int or batch_size < 1 or type(length) is not int or length < 1:
        raise ValueError('batch_size and length must be positive integers')
    if not isinstance(generator, torch.Generator):
        raise ValueError('An explicit torch.Generator is required')
    values = torch.randint(modulus, (batch_size, length), generator=generator,
                           device=generator.device, dtype=torch.long)
    return batch_from_values(values, modulus, device=device)


def source_fingerprint(source):
    """Stable digest of examples and targets, independent of chunking and device."""
    digest = hashlib.sha256()
    for name in ('values', 'targets'):
        tensor = source[name].detach().to(device='cpu', dtype=torch.long).contiguous()
        digest.update(name.encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def source_summary(source):
    return dict(example_count=int(source['targets'].shape[0]),
                length=int(source['targets'].shape[1]),
                fingerprint=source['fingerprint'] if 'fingerprint' in source else source_fingerprint(source),
                pair_count=int(source.get('pair_count', 0)),
                metadata=source.get('metadata', {}))


def make_counterfactual_pairs(pair_count, length, generator, *, r_direct, modulus=5):
    """Change x_1 by a nonzero offset; all following inputs are identical.

    Members are interleaved [pair0-a, pair0-b, pair1-a, pair1-b, ...]. Their
    complete target vectors differ by the same nonzero modular offset. The
    unchanged suffix contains at least r_direct tokens, hiding the changed
    symbol from a final-position model with no recurrent inheritance.
    """
    TaskConfig(modulus)
    if type(pair_count) is not int or pair_count < 1:
        raise ValueError('pair_count must be positive')
    if type(r_direct) is not int or r_direct < 1:
        raise ValueError('r_direct must be positive')
    if type(length) is not int or length <= r_direct:
        raise ValueError('Counterfactual pairs require length > r_direct')
    base = make_batch(pair_count, length, generator, modulus=modulus)['values']
    offsets = torch.randint(1, modulus, (pair_count,), generator=generator,
                            device=generator.device).cpu()
    changed = base.clone()
    changed[:, 0] = (changed[:, 0] + offsets).remainder(modulus)
    members = torch.stack((base, changed), dim=1).reshape(2 * pair_count, length)
    source = batch_from_values(members, modulus)
    source['pair_count'] = pair_count
    source['metadata'] = dict(
        kind='first-symbol-offset-shared-suffix', member_layout='interleaved',
        changed_position=1, shared_suffix_steps=length - 1, r_direct=r_direct,
        modular_offsets=offsets.tolist(),
        both_final_definition='Both final-position predictions correct per pair',
        both_sequence_definition='Every position correct in both members per pair',
        discrimination_definition='Final-position predictions differ per pair',
        suffix_only_both_final_accuracy=0.0,
        suffix_only_discrimination_accuracy=0.0,
        ordinary_uniform_chance=1 / modulus)
    source['fingerprint'] = source_fingerprint(source)
    return source


def _heldout_seed(seed, length, kind):
    # Independent namespaces keep a length stable when other lengths are added.
    payload = f'{GENERATOR_VERSION}:{seed}:{length}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)


def make_validation_sources(lengths, examples, paired_examples, seed, *, r_direct, modulus=5):
    """CPU held-out sets; construction never touches global or training RNG state.

    paired_examples counts members, so 256 requests 128 counterfactual pairs.
    Infeasible lengths explicitly carry a skipped reason instead of replacing
    the requested hard pair with an easier example.
    """
    TaskConfig(modulus)
    if type(r_direct) is not int or r_direct < 1:
        raise ValueError('r_direct must be positive')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(paired_examples) is not int or paired_examples < 0 or paired_examples % 2:
        raise ValueError('paired_examples must be a nonnegative even member count')
    lengths = tuple(lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError('lengths must be nonempty and unique')
    sources = {}
    for length in lengths:
        generator = torch.Generator().manual_seed(_heldout_seed(seed, length, 'ordinary'))
        ordinary = make_batch(examples, length, generator, modulus=modulus)
        ordinary['metadata'] = dict(kind='iid_uniform', seed=seed,
                                    generator_version=GENERATOR_VERSION)
        ordinary['fingerprint'] = source_fingerprint(ordinary)
        paired = None
        if not paired_examples:
            skipped = 'paired_examples is zero'
        elif length <= r_direct:
            skipped = f'length {length} <= attention-only receptive field {r_direct}; no changed prefix can precede a shared suffix of {r_direct} steps'
        else:
            generator = torch.Generator().manual_seed(_heldout_seed(seed, length, 'paired'))
            paired = make_counterfactual_pairs(paired_examples // 2, length, generator,
                                              r_direct=r_direct, modulus=modulus)
            skipped = None
        sources[str(length)] = dict(ordinary=ordinary, paired=paired,
                                    paired_skipped_reason=skipped)
    return sources


fixed_sources = make_validation_sources
