"""Source sampling and exact frequency/first-appearance semantics; no model helpers."""
import hashlib
import json

import torch
from torch.nn import functional as F

SEMANTICS_VERSION = 'most-freq-first-appearance-eos-v1'
SAMPLER_VERSION = 'most-freq-source-v1'
SAMPLERS = ('separated_counts', 'iid_uniform')


def validate_sampler(vocab_size, sampler, min_count_gap, length=None):
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('vocab_size must be a positive integer')
    if sampler not in SAMPLERS:
        raise ValueError(f'Unknown sampler: {sampler}')
    if type(min_count_gap) is not int or min_count_gap < 1:
        raise ValueError('min_count_gap must be a positive integer')
    if length is not None:
        if type(length) is not int or length < 1:
            raise ValueError('Source length must be a positive integer')
        minimum = vocab_size + min_count_gap*vocab_size*(vocab_size-1)//2
        if sampler == 'separated_counts' and length < minimum:
            raise ValueError(f'separated_counts requires T >= {minimum}')


def validate_source(source, vocab_size):
    validate_sampler(vocab_size, 'iid_uniform', 1)
    if not isinstance(source, torch.Tensor) or source.dtype != torch.long or source.ndim != 2 or min(source.shape) < 1:
        raise ValueError('Source must be nonempty int64 [batch,T]')
    if ((source < 0) | (source >= vocab_size)).any():
        raise ValueError('Source symbol outside vocabulary')


def oracle(source, vocab_size):
    """Return ranks, counts and first positions from the actual source, including absent markers."""
    validate_source(source, vocab_size)
    size, length = source.shape
    counts = source.new_zeros(size, vocab_size).scatter_add_(1, source, torch.ones_like(source))
    positions = torch.arange(length, device=source.device).expand(size, -1)
    first = source.new_full((size, vocab_size), length)
    first.scatter_reduce_(1, source, positions, reduce='amin', include_self=True)
    # Keys are unique among present symbols: equal counts have different first positions.
    order = (-counts*(length+1)+first).argsort(-1)
    ranks = [row[hist[row] > 0].tolist() for row, hist in zip(order, counts)]
    return dict(rankings=ranks, counts=counts, first_positions=first)


def output_to_input(classes, vocab_size):
    if classes.dtype != torch.long or ((classes < 0) | (classes > vocab_size)).any():
        raise ValueError('Output classes must be int64 symbols or EOS class V')
    return torch.where(classes == vocab_size, vocab_size+1, classes)


def materialize(source, vocab_size):
    validate_source(source, vocab_size)
    info = oracle(source, vocab_size)
    size, length = source.shape
    outputs = [ranking+[vocab_size] for ranking in info['rankings']]
    width = max(map(len, outputs))
    ids = source.new_full((size, length+1+width), vocab_size+2)
    ids[:, :length] = source
    ids[:, length] = vocab_size
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, :length+1] = True
    targets = source.new_full((size, width), -100)
    output_mask = torch.zeros_like(targets, dtype=torch.bool)
    for index, answer in enumerate(outputs):
        target = source.new_tensor(answer)
        stop = length+1+len(answer)
        ids[index, length+1:stop] = output_to_input(target, vocab_size)
        labels[index, length+1:stop] = target
        mask[index, length+1:stop] = True
        targets[index, :len(answer)] = target
        output_mask[index, :len(answer)] = True
    return dict(inputs=F.one_hot(ids, vocab_size+3).float(), attention_mask=mask,
                target_labels=labels, target_classes=targets, target_mask=output_mask,
                source=source.clone(), source_lengths=source.new_full((size,), length), **info)


def sample_source(batch_size, length, *, vocab_size=4, sampler='separated_counts', min_count_gap=4, generator):
    validate_sampler(vocab_size, sampler, min_count_gap, length)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('A seeded CPU torch.Generator is required')
    if sampler == 'iid_uniform':
        return torch.randint(vocab_size, (batch_size, length), generator=generator)
    minimum = 1+torch.arange(vocab_size)*min_count_gap
    remaining = length-int(minimum.sum())
    extra = torch.zeros(batch_size, vocab_size, dtype=torch.long)
    buckets = torch.randint(vocab_size, (batch_size, remaining), generator=generator)
    extra.scatter_add_(1, buckets, torch.ones_like(buckets))
    ascending = minimum+extra.sort(-1).values
    rows = []
    for counts in ascending:
        symbol_order = torch.randperm(vocab_size, generator=generator)
        mapped = torch.empty_like(counts)
        mapped[symbol_order] = counts
        source = torch.repeat_interleave(torch.arange(vocab_size), mapped)
        rows.append(source[torch.randperm(length, generator=generator)])
    return torch.stack(rows)


def generate_batch(batch_size, length, *, vocab_size=4, sampler='separated_counts', min_count_gap=4, generator):
    source = sample_source(batch_size, length, vocab_size=vocab_size, sampler=sampler,
                           min_count_gap=min_count_gap, generator=generator)
    return materialize(source, vocab_size)


def evaluation_plan(count, length, seed, *, vocab_size=4, sampler='separated_counts', min_count_gap=4):
    validate_sampler(vocab_size, sampler, min_count_gap, length)
    if type(count) is not int or count < 1 or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Positive evaluation count and seed in 0..2**63-1 required')
    identity = [SAMPLER_VERSION, sampler, seed, length, vocab_size, min_count_gap]
    rows = []
    digest = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode())
    for index in range(count):
        key = hashlib.sha256(json.dumps(identity+[index], separators=(',', ':')).encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'little') % 2**63)
        source = sample_source(1, length, vocab_size=vocab_size, sampler=sampler,
                               min_count_gap=min_count_gap, generator=generator)[0]
        rows.append(source)
        # Fixed integer serialization without a NumPy dependency.
        digest.update(bytes(source.tolist()) if vocab_size <= 256 else json.dumps(source.tolist()).encode())
    return torch.stack(rows), digest.hexdigest()


def shifted_targets(labels, attention_mask):
    if labels.ndim != 2 or labels.dtype != torch.long or attention_mask.shape != labels.shape:
        raise ValueError('Labels/mask must match [batch,physical_tokens]')
    if not ((attention_mask == 0) | (attention_mask == 1)).all():
        raise ValueError('Mask must be binary')
    return labels[:, 1:].masked_fill(~(attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()), -100)


def sequence_loss(logits, labels, attention_mask):
    if logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError('Logits and physical labels must align')
    target = shifted_targets(labels, attention_mask)
    if not target.ne(-100).any():
        raise ValueError('At least one valid target is required')
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), target.flatten(), ignore_index=-100)
