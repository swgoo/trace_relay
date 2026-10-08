"""Exact-length bounded Dyck words; stack metadata is never model input.

Uniform bounded untyped Dyck paths, with independent uniform types on opens.
This controlled close-type diagnostic does not reproduce a historical PDFA corpus.
"""
from functools import lru_cache
import hashlib
import json

import torch
from torch.nn import functional as F

SEMANTICS_VERSION = 'dyck-km-fixed-length-close-v1'
REFERENCE_URLS = [
    'https://aclanthology.org/2020.emnlp-main.156/',
    'https://github.com/john-hewitt/dyckkm-constructions',
    'https://github.com/princeton-nlp/dyck-transformer',
]
DISTANCE_BINS = ((1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256),
                 (257, 512), (513, 1024), (1025, 2048), (2049, None))


def validate_length(length):
    if type(length) is not int or length < 2 or length % 2:
        raise ValueError('Dyck length must be an even integer >= 2')


def validate_language(bracket_types, max_depth):
    if any(type(x) is not int or x < 1 for x in (bracket_types, max_depth)):
        raise ValueError('bracket_types and max_depth must be positive integers')


def vocabulary(bracket_types):
    return [f'open_{i}' for i in range(bracket_types)] + [f'close_{i}' for i in range(bracket_types)]


@lru_cache(maxsize=128)
def _completion_weights(length, depth_limit):
    """Row-normalized float64 DP. Candidate weights share the same row scale."""
    ways = torch.zeros(length+1, depth_limit+1, dtype=torch.float64)
    ways[0, 0] = 1.
    for remaining in range(1, length+1):
        ways[remaining, :-1] += ways[remaining-1, 1:]
        ways[remaining, 1:] += ways[remaining-1, :-1]
        ways[remaining] /= ways[remaining].max()
    return ways


def _sample_plan(batch_size, length, bracket_types, max_depth, generator):
    validate_length(length)
    validate_language(bracket_types, max_depth)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('Use a seeded CPU torch.Generator')
    limit = min(max_depth, length//2)
    ways = _completion_weights(length, limit)
    depth = torch.zeros(batch_size, dtype=torch.long)
    peak = depth.clone()
    stack = torch.zeros(batch_size, limit, dtype=torch.long)
    positions = torch.zeros_like(stack)
    tokens = torch.empty(batch_size, length, dtype=torch.long)
    labels = torch.full_like(tokens, -100)
    distance = torch.full_like(tokens, -1)
    preclose = torch.full_like(tokens, -1)
    rows = torch.arange(batch_size)
    # Generator-only stack loop, vectorized over examples; the backbone stays parallel.
    for t in range(length):
        prior = ways[length-t-1]
        ow = prior[(depth+1).clamp(max=limit)] * (depth < limit)
        cw = prior[(depth-1).clamp(min=0)] * (depth > 0)
        opening = torch.rand(batch_size, generator=generator, dtype=torch.float64) < ow/(ow+cw)
        kind = torch.randint(bracket_types, (batch_size,), generator=generator)
        top = (depth-1).clamp(min=0)
        closing_type = stack[rows, top]
        tokens[:, t] = torch.where(opening, kind, bracket_types+closing_type)
        labels[~opening, t] = closing_type[~opening]
        distance[~opening, t] = t-positions[rows[~opening], top[~opening]]
        preclose[~opening, t] = depth[~opening]
        stack[rows[opening], depth[opening]] = kind[opening]
        positions[rows[opening], depth[opening]] = t
        depth += torch.where(opening, 1, -1)
        peak = torch.maximum(peak, depth)
    if depth.any():
        raise RuntimeError('Invalid DP path: unfinished stack')
    return dict(tokens=tokens, close_labels=labels, matching_distance=distance,
                preclose_depth=preclose, max_depth_reached=peak)


def _materialize_batch(plan, bracket_types):
    tokens = plan['tokens']
    size, length = tokens.shape
    return dict(inputs=F.one_hot(tokens, num_classes=2*bracket_types).float(),
                attention_mask=torch.ones(size, length, dtype=torch.bool),
                close_labels=plan['close_labels'].clone(), matching_distance=plan['matching_distance'].clone(),
                preclose_depth=plan['preclose_depth'].clone(), max_depth_reached=plan['max_depth_reached'].clone(),
                lengths=torch.full((size,), length, dtype=torch.long))


def generate_dyck_batch(batch_size, length, *, bracket_types, max_depth, generator):
    return _materialize_batch(_sample_plan(batch_size, length, bracket_types, max_depth, generator), bracket_types)


def parse_dyck(tokens, *, bracket_types, max_depth):
    """Independent strict Python stack oracle, not used by the sampler/model."""
    validate_language(bracket_types, max_depth)
    values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    validate_length(len(values))
    stack, peak = [], 0
    labels, distances, depths, openers = [], [], [], []
    for t, value in enumerate(values):
        if type(value) is not int or not 0 <= value < 2*bracket_types:
            raise ValueError('Invalid bracket token')
        if value < bracket_types:
            stack.append((value, t))
            peak = max(peak, len(stack))
            if len(stack) > max_depth:
                raise ValueError('Maximum depth exceeded')
            labels.append(-100); distances.append(-1); depths.append(-1); openers.append(-1)
        else:
            if not stack or stack[-1][0] != value-bracket_types:
                raise ValueError('Mismatched closing bracket')
            depths.append(len(stack))
            kind, opener = stack.pop()
            labels.append(kind); distances.append(t-opener); openers.append(opener)
    if stack:
        raise ValueError('Unfinished stack')
    return dict(close_labels=torch.tensor(labels), matching_distance=torch.tensor(distances),
                preclose_depth=torch.tensor(depths), matching_open_index=torch.tensor(openers),
                max_depth_reached=torch.tensor(peak))


def _evaluation_plan(count, length, seed, bracket_types, max_depth):
    if type(seed) is not int or seed < 0:
        raise ValueError('Evaluation seed must be a nonnegative integer')
    identity = f'{SEMANTICS_VERSION}:{seed}:{length}:{bracket_types}:{max_depth}'
    digest = hashlib.sha256(identity.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % 2**63)
    plan = _sample_plan(count, length, bracket_types, max_depth, generator)
    fingerprint = hashlib.sha256(json.dumps([SEMANTICS_VERSION, length, bracket_types, max_depth, count]).encode())
    fingerprint.update(plan['tokens'].numpy().astype('<i8', copy=False).tobytes())
    return plan, fingerprint.hexdigest()


def iter_evaluation_batches(count, length, *, seed, batch_size, bracket_types, max_depth, plan=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Evaluation batch_size must be positive')
    if plan is None:
        plan, _ = _evaluation_plan(count, length, seed, bracket_types, max_depth)
    for start in range(0, count, batch_size):
        yield _materialize_batch({k:v[start:start+batch_size] for k,v in plan.items()}, bracket_types)


def shifted_close_targets(close_labels, attention_mask):
    """Target at t+1 belongs to logits at t; ignore openings and padding."""
    if close_labels.ndim != 2 or attention_mask.shape != close_labels.shape:
        raise ValueError('Close labels and attention mask must have matching [batch,tokens] shapes')
    return close_labels[:, 1:].masked_fill(~(attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()), -100)


def close_loss(logits, close_labels, attention_mask):
    if logits.shape[:2] != close_labels.shape:
        raise ValueError('Logits and labels must align to physical token positions')
    targets = shifted_close_targets(close_labels, attention_mask)
    if not (targets != -100).any():
        raise ValueError('At least one valid close target is required')
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
