"""Online Dyck closing-type training; only in-distribution lengths select checkpoints."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import torch

from experiment_utils import (ExperimentTracking, model_config as backbone_model_config,
                              precision_context, resource_counts, restore_rng, rng_state, write_json)
from .task import (SEMANTICS_VERSION, REFERENCE_URLS, vocabulary, generate_dyck_batch,
                       validate_length, validate_language, shifted_close_targets, close_loss)
from .evaluate import evaluate_lengths, evaluate_checkpoint, validate_lengths
from .model import (TraceRelayDyckModel, save_dyck_checkpoint,
                                              load_dyck_checkpoint)

ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path, help='Continue to a total --steps budget in a fresh output folder')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--attn-implementation', choices=['auto', 'eager', 'flash_attention_2'], default='auto')
    p.add_argument('--model', choices=['trace_relay', 'no_carry', 'swa'], default='trace_relay')
    p.add_argument('--hidden-size', type=int, default=64)
    p.add_argument('--intermediate-size', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--center-width', type=int, default=64)
    p.add_argument('--trace-widths', type=int, nargs='+', default=None)
    p.add_argument('--left-windows', type=int, nargs='+', default=[15, 15, 15])
    p.add_argument('--right-windows', type=int, nargs='+', default=[7])
    p.add_argument('--relay-strides', type=int, nargs='+', default=[8])
    p.add_argument('--skip-pairs', nargs='*', default=[])
    p.add_argument('--bracket-types', type=int, default=8)
    p.add_argument('--max-depth', type=int, default=10)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--eval-chunk-size', type=int, default=0)
    p.add_argument('--train-min-length', type=int, default=32)
    p.add_argument('--train-max-length', type=int, default=256)
    p.add_argument('--validation-lengths', type=int, nargs='+', default=[32, 64, 128, 256])
    p.add_argument('--validation-examples', type=int, default=1024)
    p.add_argument('--validation-seed', type=int, default=20261007)
    p.add_argument('--eval-lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    p.add_argument('--eval-examples', type=int, default=2000)
    p.add_argument('--eval-seed', type=int, default=20261006)
    p.add_argument('--final-evaluation', action='store_true')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--early-stop-accuracy', type=float, default=None,
                   help='Stop after consecutive ID validation means >= this threshold (default: disabled)')
    p.add_argument('--early-stop-passes', type=int, default=2)
    p.add_argument('--disable-early-stop', dest='early_stop_accuracy', action='store_const', const=None)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'], default='disabled')
    p.add_argument('--wandb-project', default='trace relay dyck')
    p.add_argument('--wandb-entity', default=None)
    return p


def model_config(args):
    cfg = copy.copy(args)
    cfg.trace_widths = args.trace_widths or [64, args.center_width, 64]
    config = backbone_model_config(cfg)
    config.update(task='dyck', vocabulary=vocabulary(args.bracket_types), semantics_version=SEMANTICS_VERSION,
                  bracket_types=args.bracket_types, max_depth=args.max_depth)
    return config


def validate_args(args):
    validate_language(args.bracket_types, args.max_depth)
    validate_length(args.train_min_length)
    validate_length(args.train_max_length)
    if args.train_min_length > args.train_max_length:
        raise ValueError('Training length range must be increasing')
    validate_lengths(args.validation_lengths)
    validate_lengths(args.eval_lengths)
    if any(not args.train_min_length <= t <= args.train_max_length for t in args.validation_lengths):
        raise ValueError('Checkpoint-selection validation lengths must be inside the training range')
    if len({args.seed, args.validation_seed, args.eval_seed}) != 3 or min(args.seed, args.validation_seed, args.eval_seed) < 0:
        raise ValueError('Training/validation/final evaluation seeds must be distinct and nonnegative')
    if min(args.steps, args.batch_size, args.eval_batch_size, args.validation_examples, args.eval_examples,
           args.eval_every, args.log_every, args.threads) < 1:
        raise ValueError('Budgets, batches, samples, intervals and threads must be positive')
    if min(args.warmup_steps, args.eval_chunk_size) < 0:
        raise ValueError('Warmup and chunk sizes must be nonnegative')
    if args.early_stop_passes < 1:
        raise ValueError('early-stop-passes must be positive')
    if args.early_stop_accuracy is not None and (
        not math.isfinite(args.early_stop_accuracy) or not 0 <= args.early_stop_accuracy <= 1
    ):
        raise ValueError('early-stop-accuracy must be finite and between 0 and 1')
    values = (args.learning_rate, args.weight_decay, args.grad_clip)
    if not all(math.isfinite(v) for v in values) or args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError('Invalid optimizer settings')


def provenance():
    sources = ['dyck/task.py', 'dyck/model.py', 'dyck/train.py',
               'dyck/evaluate.py', 'experiment_utils.py',
               'configuration_trace_relay.py', 'modeling_trace_relay.py']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    return dict(git_commit=commit, git_dirty=bool(dirty),
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                reference_urls=REFERENCE_URLS,
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


def run_training(args):
    validate_args(args)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    config = model_config(args)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    previous = None
    best_path = None
    if args.resume is None:
        model = TraceRelayDyckModel(config, args.eval_chunk_size).to(args.device)
    else:
        model, previous, digest = load_dyck_checkpoint(args.resume, args.device, args.eval_chunk_size,
                                                                config['attention_backend'])
        if dict(previous['config'], attention_backend=config['attention_backend']) != config:
            raise ValueError('Resume architecture/semantics mismatch')
        if previous.get('training_state_format') != 'dyck-training-state-v1':
            raise ValueError('Resume requires an online-training checkpoint with optimizer/generator state')
        if not previous['step'] < args.steps:
            raise ValueError('Total --steps must exceed checkpoint step')
        for key in ('seed', 'batch_size', 'train_min_length', 'train_max_length', 'validation_lengths',
                    'validation_seed', 'validation_examples', 'learning_rate', 'warmup_steps', 'weight_decay', 'grad_clip',
                    'eval_every', 'precision', 'bracket_types', 'max_depth'):
            if previous['protocol']['training'][key] != getattr(args, key):
                raise ValueError(f'Resume training setting mismatch: {key}')
        best_path = args.resume.parent/'best.pt'
        if not best_path.exists():
            raise ValueError('Resume requires the historical best.pt next to its training checkpoint')
        best_state = torch.load(best_path, map_location='cpu', weights_only=True)
        if best_state['config'] != previous['config'] or best_state['step'] != previous['best_step']:
            raise ValueError('Historical best checkpoint does not match the resumed training state')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if previous is not None:
        optimizer.load_state_dict(previous['optimizer'])
        generator.set_state(previous['generator'])
    resources = resource_counts(model)
    resume = dict(mode='fresh', start_step=0) if previous is None else dict(
        mode='full', start_step=previous['step'], checkpoint=str(args.resume.resolve()), checkpoint_sha256=digest)
    protocol = dict(format='trace-relay-dyck-protocol-v1', task='dyck',
                    semantics_version=SEMANTICS_VERSION, config=config, resources=resources, resume=resume,
                    training={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    provenance=provenance(),
                    carry_exposure=dict(first_second_delivery_token_per_layer=[2*w+1 for w in config['backbone']['relay_stride']],
                                        training_reaches_second_delivery_per_layer=[args.train_max_length >= 2*w+1 for w in config['backbone']['relay_stride']]),
                    precision='bf16' if args.device == 'cuda' and args.precision != 'fp32' else 'fp32',
                    data='Online uniform even T; row-normalized DP for uniform bounded Dyck paths; uniform opening types',
                    diagnostic='Causal next-close type only; not a reproduction of published full-LM scores',
                    target_alignment='logits at t predict close_labels at t+1; openings/padding ignored',
                    checkpoint_selection='Mean ID validation accuracy; tie-break lower mean ID CE',
                    early_stopping=dict(enabled=args.early_stop_accuracy is not None,
                                        accuracy=args.early_stop_accuracy, consecutive_passes=args.early_stop_passes,
                                        metric='mean ID validation accuracy', reset_on_failure=True,
                                        ood_used=False),
                    ood_used_for_training_control=False)
    args.output.mkdir(parents=True)
    write_json(args.output/'protocol.json', protocol)
    if previous is not None:
        shutil.copyfile(args.resume, args.output/'last.pt')
        shutil.copyfile(best_path, args.output/'best.pt')
    step = tokens = padded_tokens = close_targets = best_step = 0
    best_accuracy, best_ce = -1., float('inf')
    best_metrics = final_metrics = None
    prior_wall = 0.
    early_stop_streak = 0
    if previous is not None:
        step, tokens, padded_tokens = previous['step'], previous['tokens'], previous['padded_tokens']
        close_targets = previous['close_targets']
        best_step, best_metrics = previous['best_step'], previous['best_validation']
        final_metrics, prior_wall = previous['validation'], previous['training_wall_seconds']
        best_accuracy, best_ce = best_metrics['mean_accuracy'], best_metrics['mean_cross_entropy']
        prior_training = previous['protocol']['training']
        if (prior_training.get('early_stop_accuracy'), prior_training.get('early_stop_passes', 2)) == (
            args.early_stop_accuracy, args.early_stop_passes
        ):
            early_stop_streak = previous.get('early_stop_consecutive_passes', 0)
    start_step, start_tokens = step, tokens
    interrupted = False
    stop_reason = 'step_budget'
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with ExperimentTracking(args.output, protocol, mode=args.wandb_mode, project=args.wandb_project,
                    entity=args.wandb_entity) as tracker, (args.output/'metrics.jsonl').open('w') as handle:
        def record(row):
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            handle.flush()
            tracker.log(row)
            print(json.dumps(row, allow_nan=False), flush=True)

        def checkpoint(name):
            save_dyck_checkpoint(args.output/name, model, task='dyck',
                semantics_version=SEMANTICS_VERSION, protocol=protocol, step=step, tokens=tokens,
                padded_tokens=padded_tokens, physical_tokens=tokens, close_targets=close_targets, training_wall_seconds=prior_wall+time.perf_counter()-start,
                best_step=best_step, best_validation=best_metrics, validation=final_metrics,
                early_stop_consecutive_passes=early_stop_streak,
                training_state_format='dyck-training-state-v1', optimizer=optimizer.state_dict(),
                generator=generator.get_state(), rng=rng_state())

        try:
            if previous is not None:
                restore_rng(previous['rng'])
            for next_step in range(start_step+1, args.steps+1):
                if args.early_stop_accuracy is not None and early_stop_streak >= args.early_stop_passes:
                    stop_reason = 'validation_threshold'
                    break
                length = 2*int(torch.randint(args.train_min_length//2, args.train_max_length//2+1, (), generator=generator))
                batch = generate_dyck_batch(args.batch_size, length, bracket_types=args.bracket_types,
                                            max_depth=args.max_depth, generator=generator)
                mask = batch['attention_mask'].to(args.device)
                labels = batch['close_labels'].to(args.device)
                targets = shifted_close_targets(labels, mask)
                model.train()
                rate = args.learning_rate*min(1., next_step/max(1, args.warmup_steps))
                for group in optimizer.param_groups:
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                with precision_context(args.device, args.precision):
                    logits = model(batch['inputs'].to(args.device), mask)
                    loss = close_loss(logits, labels, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                step = next_step
                tokens += args.batch_size*length
                padded_tokens += args.batch_size*length
                valid_targets = targets != -100
                close_targets += int(valid_targets.sum())
                if step == start_step+1 or step % args.log_every == 0 or step == args.steps:
                    record(dict(phase='training', step=step, loss=float(loss.detach()),
                                close_minibatch_accuracy=float((logits[:, :-1].argmax(-1)[valid_targets] == targets[valid_targets]).float().mean()),
                                batch_close_targets=int(valid_targets.sum()), close_targets=close_targets, physical_tokens=tokens,
                                max_depth_mean=float(batch['max_depth_reached'].float().mean()),
                                max_depth_max=int(batch['max_depth_reached'].max()),
                                grad_norm=float(grad_norm), learning_rate=rate, length=length, tokens=tokens,
                                wall_seconds=prior_wall+time.perf_counter()-start))
                if step % args.eval_every == 0 or step == args.steps:
                    rows = evaluate_lengths(model, args.validation_lengths, examples=args.validation_examples,
                                            batch_size=args.eval_batch_size, seed=args.validation_seed,
                                            device=args.device, precision=args.precision)
                    accuracy = sum(r['close_accuracy'] for r in rows)/len(rows)
                    ce = sum(r['close_cross_entropy'] for r in rows)/len(rows)
                    final_metrics = dict(mean_accuracy=accuracy, mean_cross_entropy=ce, rows=rows)
                    passed = args.early_stop_accuracy is not None and accuracy >= args.early_stop_accuracy
                    early_stop_streak = early_stop_streak+1 if passed else 0
                    metrics = dict(mean_accuracy=accuracy, mean_cross_entropy=ce)
                    for row in rows:
                        metrics.update({f"length_{row['length']}/{key}":row[key] for key in ('close_accuracy', 'close_cross_entropy', 'close_count')})
                    record(dict(phase='validation', step=step, metrics=metrics,
                                early_stop_consecutive_passes=early_stop_streak,
                                tokens=tokens, wall_seconds=prior_wall+time.perf_counter()-start))
                    if (accuracy, -ce) > (best_accuracy, -best_ce):
                        best_accuracy, best_ce, best_step, best_metrics = accuracy, ce, step, final_metrics
                        checkpoint('best.pt')
                    checkpoint('last.pt')
                    if passed and early_stop_streak >= args.early_stop_passes:
                        stop_reason = 'validation_threshold'
                        record(dict(phase='early_stop', step=step, accuracy=accuracy,
                                    target_accuracy=args.early_stop_accuracy,
                                    consecutive_passes=early_stop_streak))
                        break
        except KeyboardInterrupt:
            interrupted = True
            stop_reason = 'interrupted'
            print('Interrupted; keeping the last completed validation checkpoints.', flush=True)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        result = dict(format='trace-relay-dyck-training-v1', task='dyck',
                      semantics_version=SEMANTICS_VERSION, steps=step, tokens=tokens, padded_tokens=padded_tokens,
                      physical_tokens=tokens, close_targets=close_targets,
                      training_wall_seconds=prior_wall+elapsed, segment_wall_seconds=elapsed,
                      tokens_per_second=(tokens-start_tokens)/elapsed, start_step=start_step,
                      segment_steps=step-start_step, best_step=best_step, best_validation=best_metrics,
                      last_validation=final_metrics, interrupted=interrupted, resources=resources,
                      stop_reason=stop_reason, early_stopped=stop_reason == 'validation_threshold',
                      early_stop_consecutive_passes=early_stop_streak,
                      peak_training_vram_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      ood_evaluated=False)
        write_json(args.output/'results.json', result)
        tracker.summary('training', dict(steps=step, tokens=tokens, close_targets=close_targets, best_step=best_step,
                                        best_mean_accuracy=best_accuracy, wall_seconds=prior_wall+elapsed,
                                        interrupted=interrupted, stop_reason=stop_reason,
                                        early_stop_consecutive_passes=early_stop_streak))
        if args.final_evaluation and not interrupted:
            final = evaluate_checkpoint(args.output/'best.pt', args.output/'final_eval', lengths=args.eval_lengths,
                                        examples=args.eval_examples, batch_size=args.eval_batch_size, seed=args.eval_seed,
                                        device=args.device, precision=args.precision, chunk_size=args.eval_chunk_size)
            for row in final['rows']:
                record(dict(phase='final_evaluation', step=best_step, length=row['length'], split=row['split'],
                            metrics={k:row[k] for k in ('close_accuracy', 'close_cross_entropy', 'close_count')}))
            result['ood_evaluated'] = any(row['split'] == 'ood' for row in final['rows'])
            result['final_evaluation'] = str(args.output/'final_eval/results.json')
            write_json(args.output/'results.json', result)
    return result


def main(argv=None):
    return run_training(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
