"""Aligned state tracking at fixed or uniformly sampled lengths with exact continuation."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from .model import (ModularAdditionModel, create_model, aligned_loss, build_config, estimate_flops,
                    load_checkpoint, save_checkpoint, normalize_config)
from .task import make_batch, make_validation_sources
from .evaluate import evaluate_model, evaluate_lengths
from .utils import (DEFAULT_PROJECT, DeferredSignals, Tracking, autocast, capture_rng,
                    effective_precision, provenance, restore_rng, seed_everything, stream_seed, write_json)

TRAINING_STATE_FORMAT = 'modular-addition-training-state-v1'
RESUME_SETTINGS = ('seed', 'modulus', 'sequence_length', 'train_min_length', 'batch_size', 'device', 'precision',
    'learning_rate', 'weight_decay', 'grad_clip', 'warmup_steps', 'lr_scheduler', 'min_lr_ratio',
    'validation_examples', 'validation_seed', 'validation_pair_examples', 'eval_lengths',
    'probe_examples', 'eval_every', 'length_eval_every', 'eval_batch_size', 'eval_chunk_size',
    'strict_loss_threshold', 'threads', 'paired_validation_length')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--attn-implementation', choices=['auto', 'eager', 'flash_attention_2'], default='auto')
    p.add_argument('--modulus', type=int, default=5)
    p.add_argument('--sequence-length', type=int, default=20,
                   help='Maximum training length and primary validation length')
    p.add_argument('--train-min-length', type=int, default=None,
                   help='Uniformly sample a length per batch from this minimum to --sequence-length; omitted means fixed')
    p.add_argument('--hidden-size', type=int, default=128)
    p.add_argument('--intermediate-size', type=int, default=512)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--read-windows', type=int, nargs='+', default=[5],
                   help='One shared window or one window per layer')
    p.add_argument('--write-windows', type=int, nargs='+', default=[5],
                   help='One shared window or one window per layer')
    p.add_argument('--delta-windows', '--delta-window', type=int, nargs='+', default=[0],
                   help='TraceRelay raw-delta attention: past positions plus current; 0 disables a layer')
    p.add_argument('--phase-correction-scale', type=float, default=math.pi,
                   help='TraceRelay delta-attention correction bound after tanh, in radians (0 silences it)')
    p.add_argument('--phase-correction-schedule', choices=['constant','linear_decay'], default='constant')
    p.add_argument('--phase-correction-decay-start', type=int)
    p.add_argument('--phase-correction-decay-end', type=int)
    p.add_argument('--final-delta-ablation', action='store_true')
    p.add_argument('--delta-detach-inputs', action='store_true',
                   help='Stop auxiliary Q/K/V input gradients; preserve forward correction and raw carry')
    p.add_argument('--delta-diagnostics-every', type=int, default=0,
                   help='Fixed-checkpoint ON/OFF probes and correction statistics; 0 disables')
    p.add_argument('--delta-diagnostics-lengths', type=int, nargs='+', default=[40,80])
    p.add_argument('--save-diagnostic-checkpoints', action='store_true')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--lr-scheduler', choices=['constant', 'cosine'], default='cosine')
    p.add_argument('--lr-schedule-steps', type=int)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--validation-examples', type=int, default=2000)
    p.add_argument('--validation-pair-examples', type=int, default=256)
    p.add_argument('--paired-validation-length', type=int, default=None,
                   help='Fixed paired-validation length; otherwise first evaluation length beyond the direct field')
    p.add_argument('--validation-seed', type=int, default=20261023)
    p.add_argument('--probe-examples', type=int, default=512)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--length-eval-every', type=int, default=1000)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--strict-loss-threshold', type=float, default=1e-4)
    p.add_argument('--eval-lengths', type=int, nargs='+', default=[20,40,80,160,320])
    p.add_argument('--eval-examples', type=int, default=1024)
    p.add_argument('--eval-pair-examples', type=int, default=256)
    p.add_argument('--eval-seed', type=int, default=20261024)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--eval-chunk-size', type=int, default=0)
    p.add_argument('--final-evaluation', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--wandb-mode', choices=['disabled','offline','online'], default='disabled')
    p.add_argument('--wandb-project', default=DEFAULT_PROJECT)
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-resume-id', default=None)
    return p


def architecture(a):
    return build_config(modulus=a.modulus, hidden_size=a.hidden_size, intermediate_size=a.intermediate_size,
        heads=a.heads, layers=a.layers,
        read_windows=a.read_windows[0] if len(a.read_windows)==1 else a.read_windows,
        write_windows=a.write_windows[0] if len(a.write_windows)==1 else a.write_windows,
        delta_windows=a.delta_windows[0] if len(a.delta_windows)==1 else a.delta_windows,
        phase_correction_scale=a.phase_correction_scale,
        delta_detach_inputs=a.delta_detach_inputs,
        device=a.device, precision=a.precision, attn_implementation=a.attn_implementation)


def validate_args(a):
    correction_scheduler_config(a)
    if a.delta_diagnostics_every < 0:
        raise ValueError('Diagnostic interval must be nonnegative')
    if a.delta_detach_inputs and not any(a.delta_windows):
        raise ValueError('Detached delta inputs require active TraceRelay delta attention')
    if a.delta_diagnostics_every:
        if not any(a.delta_windows):
            raise ValueError('Delta diagnostics require active TraceRelay delta attention')
        if (not a.delta_diagnostics_lengths or min(a.delta_diagnostics_lengths) < 1
                or len(set(a.delta_diagnostics_lengths)) != len(a.delta_diagnostics_lengths)
                or not set(a.delta_diagnostics_lengths).issubset(a.eval_lengths)):
            raise ValueError('Diagnostic lengths must be unique positive evaluation lengths')
    elif a.save_diagnostic_checkpoints:
        raise ValueError('Diagnostic checkpoints require delta diagnostics')
    positive = ('sequence_length','steps','batch_size','validation_examples','validation_pair_examples',
        'probe_examples','eval_every','length_eval_every','log_every','eval_examples','eval_pair_examples',
        'eval_batch_size','threads')
    if any(getattr(a, name) < 1 for name in positive) or a.modulus < 2:
        raise ValueError('Modulus >=2 and positive lengths, budgets, intervals and example counts required')
    if a.train_min_length is not None and not 1 <= a.train_min_length <= a.sequence_length:
        raise ValueError('Training minimum length must be in 1..sequence_length')
    if a.validation_pair_examples % 2 or a.eval_pair_examples % 2:
        raise ValueError('Paired example counts must be even member counts')
    if (len(set(a.eval_lengths)) != len(a.eval_lengths) or not a.eval_lengths or
            min(a.eval_lengths) < 1 or a.sequence_length not in a.eval_lengths):
        raise ValueError('Unique positive evaluation lengths must include the training length')
    if not any(n > a.sequence_length for n in a.eval_lengths):
        raise ValueError('Evaluation must include a length beyond the training length')
    if len({a.seed,a.validation_seed,a.eval_seed}) != 3 or any(not 0 <= s < 2**63 for s in (a.seed,a.validation_seed,a.eval_seed)):
        raise ValueError('Training, validation and final-test seeds must be distinct')
    if (not all(math.isfinite(v) for v in (a.learning_rate,a.weight_decay,a.grad_clip,a.min_lr_ratio,a.strict_loss_threshold))
            or a.learning_rate <= 0 or a.weight_decay < 0 or a.grad_clip <= 0
            or not 0 <= a.min_lr_ratio <= 1 or a.strict_loss_threshold <= 0
            or min(a.warmup_steps,a.eval_chunk_size) < 0):
        raise ValueError('Invalid optimization, loss threshold or chunk settings')
    if a.wandb_resume_id and (a.resume is None or a.wandb_mode != 'online'):
        raise ValueError('Online W&B continuation requires a resume checkpoint')
    config=architecture(a)
    if a.paired_validation_length is not None and (
            a.paired_validation_length not in a.eval_lengths or
            a.paired_validation_length <= config['direct_receptive_field']['width']):
        raise ValueError('Paired validation length must be an evaluation length beyond the direct receptive field')


def scheduler_config(a, previous=None):
    saved = previous.get('lr_scheduler') if previous else None
    horizon = a.lr_schedule_steps if a.lr_schedule_steps is not None else saved['schedule_steps'] if saved else a.steps
    if a.lr_scheduler == 'cosine' and horizon <= a.warmup_steps:
        raise ValueError('Cosine horizon must exceed warmup')
    result = dict(name=a.lr_scheduler, peak_lr=a.learning_rate, warmup_steps=a.warmup_steps,
                  min_lr_ratio=a.min_lr_ratio, schedule_steps=horizon)
    if saved is not None and result != saved:
        raise ValueError('Resume LR schedule mismatch')
    return result


def correction_scheduler_config(a, previous=None):
    name = a.phase_correction_schedule
    start, end = a.phase_correction_decay_start, a.phase_correction_decay_end
    result = dict(name=name, initial_scale=a.phase_correction_scale)
    if name == 'linear_decay':
        if not any(a.delta_windows):
            raise ValueError('Correction decay requires TraceRelay delta attention')
        if start is None or end is None or not 0 <= start < end:
            raise ValueError('Correction decay requires 0 <= start < end')
        result.update(decay_start_step=start, decay_end_step=end)
    elif start is not None or end is not None:
        raise ValueError('Decay endpoints require a linear_decay correction schedule')
    if previous is not None:
        saved = previous.get('phase_correction_schedule', dict(name='constant',
            initial_scale=previous['config'].get('backbone',{}).get('phase_correction_scale',math.pi)))
        if saved != result:
            raise ValueError('Resume phase correction schedule mismatch')
    return result


def correction_scale(step, schedule):
    if schedule['name'] == 'constant':
        return schedule['initial_scale']
    start, end = schedule['decay_start_step'], schedule['decay_end_step']
    return schedule['initial_scale'] * min(1.,max(0.,(end-step)/(end-start)))


def learning_rate(step, schedule):
    if step <= schedule['warmup_steps']:
        return schedule['peak_lr'] * step / max(1,schedule['warmup_steps'])
    if schedule['name'] == 'constant':
        return schedule['peak_lr']
    progress = min(1.,max(0.,(step-schedule['warmup_steps'])/(schedule['schedule_steps']-schedule['warmup_steps'])))
    return schedule['peak_lr']*(schedule['min_lr_ratio']+(1-schedule['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress)))


def settings(a):
    return {name: str(value) if isinstance(value,Path) else value for name,value in vars(a).items()}


def training_length_config(a):
    low = a.sequence_length if a.train_min_length is None else a.train_min_length
    return dict(min_length=low,max_length=a.sequence_length,
        distribution='fixed' if low==a.sequence_length else 'iid_uniform_integer_per_batch',
        unit='model input steps',version='modular-addition-length-stream-v1')


def draw_training_length(a, generator):
    low = a.sequence_length if a.train_min_length is None else a.train_min_length
    if low==a.sequence_length:
        return a.sequence_length
    return int(torch.randint(low,a.sequence_length+1,(),generator=generator))


def _source_summary(sources):
    return {length: {kind: value['fingerprint'] for kind,value in entry.items()
                     if kind in ('ordinary','paired') and value is not None}
            for length,entry in sources.items()}


def _scalars(metrics):
    result = {name:value for name,value in metrics.items() if isinstance(value,(int,float,bool))}
    for name,value in metrics.get('paired',{}).items():
        if isinstance(value,(int,float,bool)):
            result[name] = value
    for region,values in metrics.get('regions',{}).items():
        for name,value in values.items():
            if isinstance(value,(int,float,bool)):
                result[region+'_'+name] = value
    return result


def _milestones(milestones, metrics, a, step, samples, tokens, flops):
    criteria = {'token_accuracy_99':metrics['token_accuracy'] >= .99,
                'final_accuracy_99':metrics['final_accuracy'] >= .99,
                'strict_validation_ce':metrics['cross_entropy'] <= a.strict_loss_threshold}
    for name,passed in criteria.items():
        if passed and milestones.get(name) is None:
            milestones[name] = dict(step=step,samples=samples,tokens=tokens,estimated_flops=flops,
                validation_cross_entropy=metrics['cross_entropy'],token_accuracy=metrics['token_accuracy'],
                final_accuracy=metrics['final_accuracy'],criterion=('validation CE <= '+str(a.strict_loss_threshold)
                    if name=='strict_validation_ce' else 'validation accuracy >=0.99'),
                samples_definition='cumulative online samples consumed; not distinct fixed-dataset cardinality')


def run_training(a):
    validate_args(a)
    if a.output.exists():
        raise FileExistsError('Refusing existing output; resume into a fresh directory')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.set_num_threads(a.threads)
    seed_everything(a.seed)
    config = architecture(a)
    previous = None
    if a.resume is None:
        model = create_model(config,a.eval_chunk_size).to(a.device)
    else:
        model,previous,digest = load_checkpoint(a.resume,device=a.device,precision=a.precision,
            attention_backend=config['attention_backend'],chunk_size=a.eval_chunk_size)
        if config != previous['config'] or previous.get('training_state_format') != TRAINING_STATE_FORMAT:
            raise ValueError('Resume architecture or training-state mismatch')
        if a.steps <= previous['step']:
            raise ValueError('Resume budget must exceed saved step')
        for name in RESUME_SETTINGS:
            old = (previous['protocol']['training'].get(name,None)
                   if name in ('train_min_length','paired_validation_length')
                   else previous['protocol']['training'][name])
            if old != getattr(a,name):
                raise ValueError('Resume setting mismatch: '+name)
    schedule = scheduler_config(a,previous)
    phase_schedule = correction_scheduler_config(a,previous)
    parameters = model.parameters()
    optimizer = torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=a.weight_decay)
    data_generator = torch.Generator(device='cpu').manual_seed(stream_seed(a.seed,'training'))
    length_generator = torch.Generator(device='cpu').manual_seed(stream_seed(a.seed,'training_length'))
    length_counts = {}
    field = config['direct_receptive_field']
    r_direct = field['width'] if isinstance(field,dict) else field
    pair_length = (a.paired_validation_length if a.paired_validation_length is not None else
                   next((length for length in sorted(a.eval_lengths) if length > r_direct),None))
    if pair_length is None:
        raise ValueError('At least one evaluation length must exceed the direct receptive field for paired validation')
    ordinary_sources = make_validation_sources([a.sequence_length],a.validation_examples,a.validation_pair_examples,
        a.validation_seed,r_direct=r_direct,modulus=a.modulus)
    probe_sources = make_validation_sources([n for n in a.eval_lengths if n != a.sequence_length],
        a.probe_examples,a.validation_pair_examples,a.validation_seed,r_direct=r_direct,modulus=a.modulus)
    pair_source = (ordinary_sources | probe_sources)[str(pair_length)]['paired']
    source_fingerprints = dict(id=_source_summary(ordinary_sources),probes=_source_summary(probe_sources))
    diagnostic_sources = diagnostic_steps = None
    if a.delta_diagnostics_every:
        diagnostic_sources = make_validation_sources(a.delta_diagnostics_lengths,a.probe_examples,0,
            a.validation_seed,r_direct=r_direct,modulus=a.modulus)
        diagnostic_steps = sorted({s for s in [0,a.eval_every,2*a.eval_every,a.length_eval_every,a.steps,
            *range(a.delta_diagnostics_every,a.steps+1,a.delta_diagnostics_every)] if s <= a.steps})
    step = samples = tokens = flops = best_step = validation_step = 0
    last_validation = best_validation = None
    milestones = dict(token_accuracy_99=None,final_accuracy_99=None,strict_validation_ce=None)
    prior_wall = 0.
    historical_best = None
    if previous is not None:
        if source_fingerprints != previous['protocol']['source_fingerprints']:
            raise ValueError('Resume validation source mismatch')
        optimizer.load_state_dict(previous['optimizer'])
        data_generator.set_state(previous['data_rng'])
        if previous.get('length_rng') is not None:
            length_generator.set_state(previous['length_rng'])
        length_counts = dict(previous.get('length_counts',{str(a.sequence_length):previous['step']}))
        restore_rng(previous['rng'])
        step,samples,tokens,flops,best_step,validation_step = (previous[k] for k in
            ('step','samples','tokens','estimated_flops','best_step','validation_step'))
        last_validation,best_validation,milestones = (previous[k] for k in ('last_validation','best_validation','milestones'))
        prior_wall = previous.get('wall_seconds',0.)
        if best_validation is not None:
            historical_best = a.resume.parent/'best.pt'
            state = torch.load(historical_best,map_location='cpu',weights_only=True)
            if (state['step'] != best_step or normalize_config(state['config']) != config
                    or state['best_validation'] != best_validation):
                raise ValueError('Historical best checkpoint mismatch')
    resume_id = a.wandb_resume_id
    if previous is not None and previous['protocol']['training']['wandb_mode']=='online':
        tracking_path = a.resume.parent/'wandb_run.json'
        recorded = json.loads(tracking_path.read_text())['id'] if tracking_path.exists() else None
        if recorded and resume_id and recorded != resume_id:
            raise ValueError('W&B run identity mismatch')
        resume_id = resume_id or recorded
        if not resume_id or a.wandb_mode != 'online':
            raise ValueError('Online resume must continue the same W&B run')
    protocol = dict(format='modular-addition-protocol-v1',task='modular_addition',config=config,
        training=settings(a),seeds=dict(train=a.seed,validation=a.validation_seed,evaluation=a.eval_seed,
        training_stream=stream_seed(a.seed,'training'),length_stream=stream_seed(a.seed,'training_length')),
        source_fingerprints=source_fingerprints,training_lengths=training_length_config(a),
        lr_scheduler=schedule,modulus=a.modulus,sequence_length=a.sequence_length,supervision='aligned_cot',
        input_representation='one-hot integers only; no BOS/EOS/query or target feedback',
        direct_receptive_field=field,paired_validation_length=pair_length,
        selection='minimum ordinary in-distribution validation CE; OOD and paired metrics are monitor only',
        validation_control='maximum training length only; fixed budget, no early stopping',
        primary_validation_length=a.sequence_length,
        strict_loss_threshold=a.strict_loss_threshold,
        milestone_resolution_steps=a.eval_every,samples_definition='cumulative online samples consumed, not paper N*',
        reference=dict(url='https://proceedings.mlr.press/v306/ebrahimi26a.html',paper_validation_ce_threshold=1e-4,
            comparison='Initial online AdamW sweep, not fixed-dataset sample-complexity reproduction'),
        parameter_count=sum(p.numel() for p in model.parameters()),precision=effective_precision(a.device,a.precision),
        provenance=provenance(),resume=dict(mode='fresh' if previous is None else 'full',start_step=step,
            checkpoint=None if a.resume is None else str(a.resume),sha256=None if previous is None else digest))
    if phase_schedule is not None:
        model.set_phase_correction_scale(correction_scale(step,phase_schedule))
        protocol['phase_correction_schedule'] = phase_schedule
    if diagnostic_sources is not None:
        protocol['delta_diagnostics'] = dict(steps=diagnostic_steps,
            source_fingerprints=_source_summary(diagnostic_sources),
            examples=a.probe_examples,lengths=a.delta_diagnostics_lengths,
            used_for_training_control=False,save_checkpoints=a.save_diagnostic_checkpoints,
            rng_preserved=True,definition='Fixed validation examples; runtime scale versus zero at identical weights')
    a.output.mkdir(parents=True)
    write_json(a.output/'protocol.json',protocol)
    if historical_best is not None:
        shutil.copy2(historical_best,a.output/'best.pt')
    metrics_path = a.output/'metrics.jsonl'
    if previous is not None:
        original = a.resume.parent/'metrics.jsonl'
        if original.exists():
            with metrics_path.open('w') as handle:
                for line in original.open():
                    if json.loads(line)['step'] <= step:
                        handle.write(line)
    start_step,start_samples,start_tokens = step,samples,tokens
    start = time.perf_counter()
    eval_options = dict(r_direct=r_direct,batch_size=a.eval_batch_size,device=a.device,precision=a.precision)
    with Tracking(a.output,protocol,a.wandb_mode,a.wandb_project,a.wandb_entity,resume_id=resume_id) as tracking, DeferredSignals() as signals:
        def record(phase,**values):
            row=dict(phase=phase,step=step,**values)
            if phase_schedule is not None:
                row['phase_correction_scale'] = model.backbone.config.phase_correction_scale
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
            tracking.log(row)
            print(json.dumps({k:v for k,v in row.items() if k=='phase' or isinstance(v,(int,float,bool))},allow_nan=False),flush=True)

        def checkpoint(name):
            extra = dict(phase_correction_schedule=phase_schedule) if phase_schedule is not None else {}
            save_checkpoint(a.output/name,model,training_state_format=TRAINING_STATE_FORMAT,protocol=protocol,
                step=step,samples=samples,tokens=tokens,estimated_flops=flops,optimizer=optimizer.state_dict(),
                data_rng=data_generator.get_state(),length_rng=length_generator.get_state(),length_counts=length_counts,
                rng=capture_rng(),lr_scheduler=schedule,best_step=best_step,
                validation_step=validation_step,last_validation=last_validation,best_validation=best_validation,
                milestones=milestones,wall_seconds=prior_wall+time.perf_counter()-start,**extra)

        def diagnose():
            from .delta_diagnostics import evaluate_delta_diagnostics, flatten_diagnostics
            report = evaluate_delta_diagnostics(model,diagnostic_sources,
                batch_size=a.eval_batch_size,device=a.device,precision=a.precision)
            report['step'] = step
            write_json(a.output/'diagnostics'/f'step_{step:06d}.json',report)
            record('delta_diagnostics',metrics=flatten_diagnostics(report),used_for_training_control=False)
            if a.save_diagnostic_checkpoints:
                checkpoint(f'diagnostics/step_{step:06d}.pt')

        checkpoint('last.pt')
        if diagnostic_steps is not None and step in diagnostic_steps:
            diagnose()
        for update in range(step+1,a.steps+1):
            if signals.requested:
                break
            if phase_schedule is not None:
                model.set_phase_correction_scale(correction_scale(update,phase_schedule))
                if update%a.eval_every==0 or update%a.length_eval_every==0 or update==a.steps:
                    eval_options['r_direct'] = model.effective_config()['direct_receptive_field']['width']
            batch_length = draw_training_length(a,length_generator)
            batch = make_batch(a.batch_size,batch_length,data_generator,modulus=a.modulus)
            inputs,targets = batch['inputs'].to(a.device),batch['targets'].to(a.device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rate = learning_rate(update,schedule)
            for group in optimizer.param_groups:
                group['lr'] = rate
            with autocast(a.device,a.precision):
                logits=model(inputs)
                loss=aligned_loss(logits,targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite modular addition loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip,error_if_nonfinite=True)
            optimizer.step()
            step=update; samples+=a.batch_size; tokens+=a.batch_size*batch_length
            length_counts[str(batch_length)] = length_counts.get(str(batch_length),0)+1
            flops+=estimate_flops(config,a.batch_size,batch_length,training=True)
            if step==start_step+1 or step%a.log_every==0 or step==a.steps:
                correct=logits.detach().argmax(-1).eq(targets)
                record('train',cross_entropy=float(loss.detach()),token_accuracy=float(correct.float().mean()),
                    final_accuracy=float(correct[:,-1].float().mean()),sequence_exact_accuracy=float(correct.all(-1).float().mean()),
                    gradient_norm=float(grad),learning_rate=rate,samples=samples,tokens=tokens,estimated_flops=flops,
                    sequence_length=batch_length,min_length_seen=min(map(int,length_counts)),max_length_seen=max(map(int,length_counts)),
                    wall_seconds=prior_wall+time.perf_counter()-start,
                    tokens_per_second=(tokens-start_tokens)/max(1e-9,time.perf_counter()-start))
            if step%a.eval_every==0 or step==a.steps:
                last_validation=evaluate_model(model,ordinary_sources[str(a.sequence_length)]['ordinary'],**eval_options)
                paired=evaluate_model(model,pair_source,**eval_options)
                validation_step=step
                _milestones(milestones,last_validation,a,step,samples,tokens,flops)
                record('val',**_scalars(last_validation),samples=samples,tokens=tokens,
                    position_wise=last_validation['position_wise'],
                    metrics={**{'paired_'+k:v for k,v in _scalars(paired).items()},
                        **{f'length_{a.sequence_length}_ordinary_'+k:v for k,v in _scalars(last_validation).items()}})
                write_json(a.output/'latest_validation.json',dict(step=step,ordinary=last_validation,paired=paired))
                if best_validation is None or last_validation['cross_entropy']<best_validation['cross_entropy']:
                    best_validation=dict(last_validation); best_step=step; checkpoint('best.pt')
            if step%a.length_eval_every==0 or step==a.steps:
                length_results=evaluate_lengths(model,ordinary_sources | probe_sources,**eval_options)
                flat={f'length_{length}_{kind}_{k}':v for length,entry in length_results['lengths'].items()
                    for kind in ('ordinary','paired') if entry.get(kind) is not None
                    for k,v in _scalars(entry[kind]).items()}
                record('length_val',lengths=length_results['lengths'],metrics=flat,used_for_training_control=False)
                write_json(a.output/'latest_length_validation.json',dict(step=step,**length_results))
            if diagnostic_steps is not None and step in diagnostic_steps:
                diagnose()
            if signals.requested or step%a.log_every==0 or step%a.eval_every==0 or step==a.steps:
                checkpoint('last.pt')
            if signals.requested:
                break
        checkpoint('last.pt')
        result=dict(format='modular-addition-training-results-v1',steps=step,samples=samples,tokens=tokens,
            estimated_flops=flops,best_step=best_step,validation_step=validation_step,last_validation=last_validation,
            best_validation=best_validation,milestones=milestones,interrupted=bool(signals.requested),
            stop_reason='interrupted' if signals.requested else 'step_budget',signal=signals.name,
            wall_seconds=prior_wall+time.perf_counter()-start,segment_steps=step-start_step,
            parameter_count=protocol['parameter_count'],training_sequence_length=a.sequence_length,
            training_lengths=training_length_config(a),length_counts=length_counts,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else 0)
        if phase_schedule is not None:
            result.update(phase_correction_schedule=phase_schedule,
                          runtime_phase_correction_scale=model.backbone.config.phase_correction_scale)
        write_json(a.output/'results.json',result)
        tracking.summary(result)
        if a.final_evaluation and not signals.requested:
            from .evaluate import evaluate_checkpoint
            for name in ('last','best'):
                if (a.output/(name+'.pt')).exists():
                    final=evaluate_checkpoint(a.output/(name+'.pt'),a.output/('final_eval_'+name),
                        lengths=a.eval_lengths,examples=a.eval_examples,paired_examples=a.eval_pair_examples,
                        seed=a.eval_seed,device=a.device,precision=a.precision,batch_size=a.eval_batch_size,
                        chunk_size=a.eval_chunk_size)
                    record('final_'+name,checkpoint_step=final['checkpoint_step'],
                        evaluation_phase_correction_scale=final.get('runtime_phase_correction_scale'),metrics={
                        f'length_{length}_{kind}_{k}':v for length,entry in final['lengths'].items()
                        for kind in ('ordinary','paired') if entry.get(kind) is not None
                        for k,v in _scalars(entry[kind]).items()})
                    if a.final_delta_ablation:
                        delta_off=evaluate_checkpoint(a.output/(name+'.pt'),a.output/('final_eval_'+name+'_delta_off'),
                            lengths=a.eval_lengths,examples=a.eval_examples,paired_examples=a.eval_pair_examples,
                            seed=a.eval_seed,device=a.device,precision=a.precision,batch_size=a.eval_batch_size,
                            chunk_size=a.eval_chunk_size,disable_delta_attention=True)
                        record('final_'+name+'_delta_off',checkpoint_step=delta_off['checkpoint_step'],
                            evaluation_phase_correction_scale=delta_off['runtime_phase_correction_scale'],metrics={
                            f'length_{length}_{kind}_{k}':v for length,entry in delta_off['lengths'].items()
                            for kind in ('ordinary','paired') if entry.get(kind) is not None
                            for k,v in _scalars(entry[kind]).items()})
            final=evaluate_checkpoint(a.output/'last.pt',a.output/'final_eval_last_no_carry',
                lengths=a.eval_lengths,examples=a.eval_examples,paired_examples=a.eval_pair_examples,
                seed=a.eval_seed,device=a.device,precision=a.precision,batch_size=a.eval_batch_size,disable_carry=True,
                chunk_size=a.eval_chunk_size)
            record('final_last_no_carry',metrics={
                f'length_{length}_{kind}_{k}':v for length,entry in final['lengths'].items()
                for kind in ('ordinary','paired') if entry.get(kind) is not None
                for k,v in _scalars(entry[kind]).items()})
        return result


def main(argv=None):
    return run_training(build_parser().parse_args(argv))


if __name__=='__main__':
    main()
