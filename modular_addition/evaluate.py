"""Per-position, extrapolation and counterfactual modular-state evaluation."""
from contextlib import nullcontext
import torch
import torch.nn.functional as F

from .utils import autocast
from .task import source_summary


EVALUATION_FORMAT = 'modular_addition.evaluation.v1'


def _metric(correct, losses=None):
    count = int(correct.numel())
    successes = int(correct.sum())
    return dict(count=count, correct=successes,
                accuracy=successes / count if count else None,
                cross_entropy=float(losses.double().mean()) if count and losses is not None else None)


def score_predictions(predictions, targets, *, r_direct, losses=None, pair_count=0):
    """RF regions use one-based positions: <=R, R+1..2R, and >2R."""
    if type(r_direct) is not int or r_direct < 1:
        raise ValueError('r_direct must be a positive integer')
    if predictions.shape != targets.shape or targets.ndim != 2 or not targets.numel():
        raise ValueError('Nonempty rank-two predictions and targets of equal shape required')
    if losses is not None and losses.shape != targets.shape:
        raise ValueError('One loss per prediction is required')
    if type(pair_count) is not int or pair_count < 0 or (pair_count and len(targets) != 2 * pair_count):
        raise ValueError('Pair members must be complete and interleaved')
    correct = predictions.eq(targets)
    length = targets.shape[1]
    token = _metric(correct, losses)
    final = _metric(correct[:, -1], losses[:, -1] if losses is not None else None)
    sequence_correct = correct.all(dim=1)
    seq = dict(count=len(targets), correct=int(sequence_correct.sum()),
               accuracy=float(sequence_correct.double().mean()))
    positions = []
    for position in range(length):
        positions.append(dict(position=position + 1,
                              **_metric(correct[:, position], losses[:, position] if losses is not None else None)))
    regions = {}
    for name, start, end in (('within_rf', 0, min(r_direct, length)),
                             ('just_beyond_rf', min(r_direct, length), min(2*r_direct, length)),
                             ('far_beyond_rf', min(2*r_direct, length), length)):
        regions[name] = dict(start_position=start + 1 if start < end else None,
                            end_position=end if start < end else None,
                            **_metric(correct[:, start:end], losses[:, start:end] if losses is not None else None))
    result = dict(token=token, final=final, sequence_exact=seq,
                  token_accuracy=token['accuracy'], final_accuracy=final['accuracy'],
                  sequence_exact_accuracy=seq['accuracy'], cross_entropy=token['cross_entropy'],
                  position_wise=positions, regions=regions, r_direct=r_direct,
                  region_definition='one-based: within<=R; just_beyond=R+1..2R; far>2R')
    if pair_count:
        pair_correct = correct.reshape(pair_count, 2, length)
        pair_preds = predictions.reshape(pair_count, 2, length)
        both_final = pair_correct[:, :, -1].all(dim=1)
        both_seq = pair_correct.all(dim=2).all(dim=1)
        discriminate = pair_preds[:, 0, -1].ne(pair_preds[:, 1, -1])
        result['paired'] = dict(pair_count=pair_count,
            both_final_correct=int(both_final.sum()), both_final_accuracy=float(both_final.double().mean()),
            both_sequence_correct=int(both_seq.sum()), both_sequence_accuracy=float(both_seq.double().mean()),
            discrimination_count=int(discriminate.sum()), discrimination_accuracy=float(discriminate.double().mean()))
        result['pair_accuracy'] = result['paired']['both_final_accuracy']
    return result


@torch.inference_mode()
def evaluate_model(model, source, *, r_direct, batch_size=64, device='cpu', precision='fp32'):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    inputs, targets = source['inputs'], source['targets']
    if inputs.shape[:2] != targets.shape or targets.ndim != 2 or not targets.numel():
        raise ValueError('Malformed evaluation source')
    was_training = model.training
    model.eval()
    all_predictions, all_losses = [], []
    try:
        for start in range(0, len(targets), batch_size):
            x = inputs[start:start+batch_size].to(device)
            y = targets[start:start+batch_size].to(device)
            with autocast(device, precision):
                logits = model(x)
            if logits.shape[:-1] != y.shape or logits.shape[-1] != inputs.shape[-1]:
                raise ValueError('Model must emit one modulus-class prediction per position')
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Non-finite modular-addition logits')
            losses = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(),
                                     reduction='none').reshape_as(y)
            all_predictions.append(logits.argmax(dim=-1).cpu())
            all_losses.append(losses.cpu())
    finally:
        model.train(was_training)
    result = score_predictions(torch.cat(all_predictions), targets.cpu(), r_direct=r_direct,
                               losses=torch.cat(all_losses), pair_count=source.get('pair_count', 0))
    result['source'] = source_summary(source)
    return result


def evaluate_lengths(model, sources, **kwargs):
    rows = {}
    for length, source in sources.items():
        rows[str(length)] = dict(ordinary=evaluate_model(model, source['ordinary'], **kwargs),
            paired=evaluate_model(model, source['paired'], **kwargs) if source['paired'] is not None else None,
            paired_skipped_reason=source.get('paired_skipped_reason'))
    return dict(format=EVALUATION_FORMAT, lengths=rows)


def evaluate_checkpoint(checkpoint, output, *, lengths=(20, 40, 80, 160, 320),
                        examples=1024, paired_examples=256, seed=20261024,
                        batch_size=64, device='cpu', precision='auto',
                        disable_carry=False, attention_backend='auto', chunk_size=0,
                        disable_delta_attention=False):
    """Save an auditable fixed-weight evaluation; existing reports are preserved."""
    import csv
    import json
    from pathlib import Path
    from .model import load_checkpoint
    from .task import make_validation_sources

    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evaluation directory: {output}')
    model, state, digest = load_checkpoint(checkpoint, device=device,
        precision=precision, attention_backend=attention_backend, chunk_size=chunk_size)
    protected = state.get('protocol', {}).get('seeds', {})
    if seed in (protected.get('train'), protected.get('validation')):
        raise ValueError('Final evaluation seed must differ from training and validation seeds')
    if disable_delta_attention and not hasattr(model, 'temporary_phase_correction_scale'):
        raise ValueError('Delta attention ablation requires a TraceRelay checkpoint')
    delta_context = model.temporary_phase_correction_scale(0.) if disable_delta_attention else nullcontext()
    with delta_context, model.temporary_recurrent_carry(not disable_carry):
        effective_config = model.effective_config() if hasattr(model, 'effective_config') else model.config
        field = effective_config['direct_receptive_field']
        r_direct = field['width'] if isinstance(field, dict) else field
        sources = make_validation_sources(lengths, examples, paired_examples,
            seed, r_direct=r_direct, modulus=model.modulus)
        result = evaluate_lengths(model, sources, r_direct=r_direct, batch_size=batch_size,
                                   device=device, precision=precision)
    result.update(checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest,
                  checkpoint_step=state.get('step'), model_config=effective_config,
                  recurrent_carry=not disable_carry, evaluation_seed=seed, r_direct=r_direct)
    if hasattr(model, 'effective_config'):
        result.update(trained_model_config=model.config, delta_attention_disabled=disable_delta_attention,
            runtime_phase_correction_scale=effective_config['backbone']['phase_correction_scale'],
            phase_correction_schedule=state.get('phase_correction_schedule'))
    if 'history_ablation' in model.config:
        result['history_ablation_definition'] = model.config['history_ablation']
    output.mkdir(parents=True, exist_ok=False)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (output / 'results.json').write_text(serialized)
    (output / 'evaluation.json').write_text(serialized)
    with (output / 'position_accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['length', 'panel', 'position', 'count',
                                                    'correct', 'accuracy', 'cross_entropy'])
        writer.writeheader()
        for length, row in result['lengths'].items():
            for panel in ('ordinary', 'paired'):
                if row[panel] is not None:
                    for position in row[panel]['position_wise']:
                        writer.writerow(dict(length=length, panel=panel, **position))
    return result


def main(argv=None):
    """Evaluate one immutable checkpoint on independently seeded held-out data."""
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New output directory')
    parser.add_argument('--lengths', type=int, nargs='+', default=[20, 40, 80, 160, 320])
    parser.add_argument('--examples', type=int, default=1024)
    parser.add_argument('--paired-examples', type=int, default=256)
    parser.add_argument('--eval-seed', type=int, default=20261024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    parser.add_argument('--attention-backend', choices=['auto', 'eager', 'flash_attention_2'], default='auto')
    parser.add_argument('--chunk-size', type=int, default=0)
    parser.add_argument('--disable-carry', action='store_true')
    parser.add_argument('--disable-delta-attention', action='store_true')
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(args.checkpoint, args.output, lengths=args.lengths,
        examples=args.examples, paired_examples=args.paired_examples, seed=args.eval_seed,
        batch_size=args.batch_size, device=args.device, precision=args.precision,
        attention_backend=args.attention_backend, chunk_size=args.chunk_size,
        disable_carry=args.disable_carry, disable_delta_attention=args.disable_delta_attention)
    print(json.dumps(dict(output=str(Path(args.output).resolve()), checkpoint_step=result['checkpoint_step'],
                          lengths=args.lengths, recurrent_carry=not args.disable_carry)))
    return result


if __name__ == '__main__':
    main()
