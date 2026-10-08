"""Evaluate fixed Dyck checkpoints by length, matching distance and pre-close depth."""
import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from .task import (SEMANTICS_VERSION, DISTANCE_BINS, _evaluation_plan, iter_evaluation_batches,
                       shifted_close_targets, validate_length)
from experiment_utils import precision_context, resource_counts, write_json
from .model import load_dyck_checkpoint


def validate_lengths(lengths):
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError('Lengths must be a nonempty list of unique integers')
    for length in lengths:
        validate_length(length)


def _metrics(count, correct, ce):
    return dict(close_count=int(count), correct_count=int(correct),
                close_accuracy=float(correct/count) if count else None,
                close_cross_entropy=float(ce/count) if count else None)


@torch.no_grad()
def evaluate_length(model, length, *, examples=2000, batch_size=64, seed=20261006,
                    device='cpu', precision='auto'):
    validate_length(length)
    k, m = model.config['bracket_types'], model.config['max_depth']
    plan, digest = _evaluation_plan(examples, length, seed, k, m)
    # [count, correct, CE sum]; keep zero-count bins explicitly in every report.
    total = torch.zeros(3, dtype=torch.float64)
    distance_stats = torch.zeros(len(DISTANCE_BINS), 3, dtype=torch.float64)
    depth_stats = torch.zeros(m, 3, dtype=torch.float64)
    max_depth_stats = torch.zeros(m, 3, dtype=torch.float64)
    prior_mode = model.training
    model.eval()
    try:
        for batch in iter_evaluation_batches(examples, length, seed=seed, batch_size=batch_size,
                                             bracket_types=k, max_depth=m, plan=plan):
            mask = batch['attention_mask'].to(device)
            targets = shifted_close_targets(batch['close_labels'].to(device), mask)
            with precision_context(device, precision):
                logits = model(batch['inputs'].to(device), mask)
            if logits.shape != (*mask.shape, k) or not torch.isfinite(logits).all():
                raise ValueError('Expected finite [batch,tokens,k] logits')
            valid = targets != -100
            losses = F.cross_entropy(logits[:, :-1].float()[valid], targets[valid], reduction='none').double().cpu()
            correct = (logits[:, :-1].argmax(-1)[valid] == targets[valid]).double().cpu()
            valid = valid.cpu()
            distances = batch['matching_distance'][:, 1:][valid]
            depths = batch['preclose_depth'][:, 1:][valid]
            peaks = batch['max_depth_reached'][:, None].expand_as(valid)[valid]
            total += torch.tensor([len(correct), correct.sum(), losses.sum()], dtype=torch.float64)
            for i, (low, high) in enumerate(DISTANCE_BINS):
                selected = distances >= low
                if high is not None:
                    selected &= distances <= high
                distance_stats[i] += torch.tensor([selected.sum(), correct[selected].sum(), losses[selected].sum()])
            for values, stats in ((depths, depth_stats), (peaks, max_depth_stats)):
                stats[:, 0] += torch.bincount(values-1, minlength=m)
                stats[:, 1] += torch.bincount(values-1, weights=correct, minlength=m)
                stats[:, 2] += torch.bincount(values-1, weights=losses, minlength=m)
    finally:
        model.train(prior_mode)
    distance_rows = [dict(bin=f'{low}..{high}' if high is not None else '>2048',
                          lower=low, upper=high, **_metrics(*stats))
                     for (low, high), stats in zip(DISTANCE_BINS, distance_stats)]
    depth_rows = [dict(preclose_depth=i+1, **_metrics(*stats)) for i, stats in enumerate(depth_stats)]
    peak_rows = [dict(max_depth_reached=i+1, **_metrics(*stats)) for i, stats in enumerate(max_depth_stats)]
    return dict(length=length, bracket_types=k, max_depth=m, sequence_count=examples, **_metrics(*total),
                chance_accuracy=1/k, examples_sha256=digest, distance_metrics=distance_rows,
                depth_metrics=depth_rows, max_depth_metrics=peak_rows,
                max_depth_sequence_counts=torch.bincount(plan['max_depth_reached'], minlength=m+1)[1:].tolist())


def evaluate_lengths(model, lengths, **kwargs):
    validate_lengths(lengths)
    return [evaluate_length(model, length, **kwargs) for length in lengths]


def evaluate_checkpoint(checkpoint, output, *, lengths, examples=2000, batch_size=64, seed=20261006,
                        device='cpu', precision='auto', chunk_size=0, attention_backend=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    validate_lengths(lengths)
    model, metadata, digest = load_dyck_checkpoint(checkpoint, device, chunk_size, attention_backend)
    if model.config['attention_backend'] == 'flash_attention_2' and (device != 'cuda' or precision == 'fp32'):
        raise ValueError('FlashAttention-2 requires CUDA auto/BF16; override to eager for CPU/FP32')
    rows = evaluate_lengths(model, lengths, examples=examples, batch_size=batch_size, seed=seed,
                            device=device, precision=precision)
    training = metadata.get('protocol', {}).get('training', {})
    low, high = training.get('train_min_length'), training.get('train_max_length')
    config = model.config
    center = config['backbone']['trace_width'][config['backbone']['num_hidden_layers']//2]
    for row in rows:
        row['split'] = ('in_distribution' if low <= row['length'] <= high else 'ood') if low is not None else 'unspecified'
        row.update(model=config['family'], center_width=center, training_seed=training.get('seed'), evaluation_seed=seed)
    report = dict(format='trace-relay-dyck-evaluation-v1', task='dyck', semantics_version=SEMANTICS_VERSION,
                  diagnostic='Causal next-close type only; not a reproduction of published full-LM scores',
                  checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest,
                  training_step=metadata.get('step'), config=config, evaluation_seed=seed,
                  examples_per_length=examples, batch_size=batch_size, training_seed=training.get('seed'),
                  center_width=center, train_length_range=[low, high], resources=resource_counts(model),
                  chunk_size=chunk_size, precision=precision, rows=rows)
    write_json(output/'results.json', report)
    identity = ['model', 'center_width', 'training_seed', 'evaluation_seed', 'length', 'split',
                'bracket_types', 'max_depth', 'examples_sha256']
    metrics = ['close_accuracy', 'close_cross_entropy', 'close_count', 'correct_count']
    for name, subrows, fields in [
        ('metrics', None, ['sequence_count', 'chance_accuracy']),
        ('distance_metrics', 'distance_metrics', ['bin', 'lower', 'upper']),
        ('depth_metrics', 'depth_metrics', ['preclose_depth']),
        ('max_depth_metrics', 'max_depth_metrics', ['max_depth_reached']),
    ]:
        with (output/f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=identity+fields+metrics, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerows([{**{key:row[key] for key in identity}, **r} for r in row[subrows]] if subrows else [row])
    for row in rows:
        print(f"T={row['length']:5d} ({row['split']}): close accuracy={row['close_accuracy']:.2%}, "
              f"CE={row['close_cross_entropy']:.4f}, closes={row['close_count']}", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--examples-per-length', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261006)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--attn-implementation', choices=['eager', 'flash_attention_2'], default=None)
    p.add_argument('--chunk-size', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args(argv)
    if args.threads < 1 or args.chunk_size < 0:
        p.error('threads must be positive; chunk-size must be nonnegative')
    torch.set_num_threads(args.threads)
    return evaluate_checkpoint(args.checkpoint, args.output, lengths=args.lengths,
                               examples=args.examples_per_length, batch_size=args.batch_size, seed=args.seed,
                               device=args.device, precision=args.precision, chunk_size=args.chunk_size,
                               attention_backend=args.attn_implementation)


if __name__ == '__main__':
    main()
