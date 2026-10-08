"""Evaluate one fixed Equal Repeats checkpoint independently at every length."""
import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from .task import SEMANTICS_VERSION, _evaluation_plan, iter_evaluation_batches, validate_length
from experiment_utils import precision_context, resource_counts, write_json
from .model import load_equal_repeats_checkpoint


def validate_lengths(lengths):
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError('Lengths must be a nonempty list of unique integers')
    for length in lengths:
        validate_length(length)


@torch.no_grad()
def evaluate_length(model, length, *, examples=10000, batch_size=128, seed=20261006,
                    device='cpu', precision='auto'):
    validate_length(length)
    if type(examples) is not int or examples < 1:
        raise ValueError('Evaluation examples must be positive')
    plan, digest = _evaluation_plan(examples, length, seed)
    counts = torch.bincount(plan[2], minlength=3).tolist()
    correct, cross_entropy, confusion = 0, 0., torch.zeros(3, 3, dtype=torch.long)
    prior_mode = model.training
    model.eval()
    try:
        for batch in iter_evaluation_batches(examples, length, seed=seed, batch_size=batch_size):
            labels = batch['labels'].to(device)
            with precision_context(device, precision):
                logits = model(batch['inputs'].to(device), batch['attention_mask'].to(device))
            if logits.shape != (len(labels), 3) or not torch.isfinite(logits).all():
                raise ValueError('Expected finite [batch,3] logits')
            losses = F.cross_entropy(logits.float(), labels, reduction='none')
            predictions = logits.argmax(-1)
            correct += int((predictions == labels).sum())
            cross_entropy += float(losses.double().sum())
            confusion += torch.bincount((labels*3+predictions).cpu(), minlength=9).reshape(3, 3)
    finally:
        model.train(prior_mode)
    return dict(length=length, accuracy=correct/examples, cross_entropy=cross_entropy/examples,
                count=examples, chance_accuracy=1/3, class_counts=counts,
                majority_class_accuracy=max(counts)/examples, confusion_matrix=confusion.tolist(),
                examples_sha256=digest)


def evaluate_lengths(model, lengths, **kwargs):
    validate_lengths(lengths)
    return [evaluate_length(model, length, **kwargs) for length in lengths]


def evaluate_checkpoint(checkpoint, output, *, lengths, examples=10000, batch_size=128, seed=20261006,
                        device='cpu', precision='auto', chunk_size=0, attention_backend=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    validate_lengths(lengths)
    model, metadata, digest = load_equal_repeats_checkpoint(checkpoint, device, chunk_size, attention_backend)
    if model.config['attention_backend'] == 'flash_attention_2' and (device != 'cuda' or precision == 'fp32'):
        raise ValueError('FlashAttention-2 requires CUDA auto/BF16; override to eager for CPU/FP32')
    rows = evaluate_lengths(model, lengths, examples=examples, batch_size=batch_size, seed=seed,
                            device=device, precision=precision)
    training = metadata.get('protocol', {}).get('training', {})
    low, high = training.get('train_min_length'), training.get('train_max_length')
    center_width = model.config['backbone']['trace_width'][model.config['backbone']['num_hidden_layers']//2]
    for row in rows:
        row['split'] = ('in_distribution' if low <= row['length'] <= high else 'ood') if low is not None else 'unspecified'
        row.update(model=model.config['family'], center_width=center_width,
                   training_seed=training.get('seed'), evaluation_seed=seed)
    report = dict(format='trace-relay-equal-repeats-evaluation-v1', task='equal_repeats',
                  semantics_version=SEMANTICS_VERSION, checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=digest, training_step=metadata.get('step'), config=model.config,
                  evaluation_seed=seed, examples_per_length=examples, batch_size=batch_size,
                  training_seed=training.get('seed'), center_width=center_width,
                  train_length_range=[low, high], resources=resource_counts(model),
                  chunk_size=chunk_size, precision=precision, rows=rows)
    write_json(output/'results.json', report)
    with (output/'metrics.csv').open('w', newline='') as handle:
        fields = ['model', 'center_width', 'training_seed', 'evaluation_seed', 'length', 'split', 'accuracy', 'cross_entropy', 'count', 'chance_accuracy',
                  'majority_class_accuracy', 'examples_sha256']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"T={row['length']:5d} ({row['split']}): accuracy={row['accuracy']:.2%}, CE={row['cross_entropy']:.4f}, n={row['count']}", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--examples-per-length', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=128)
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
