"""Fixed-budget online Most-Freq training with ID free-generation selection and exact resume."""
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch

from .task import SEMANTICS_VERSION,SAMPLER_VERSION,SAMPLERS,validate_sampler,generate_batch,sequence_loss,shifted_targets
from .model import MostFreqModel,build_config,save_checkpoint,load_checkpoint,CHECKPOINT_FORMAT
from .evaluate import evaluate_lengths,validation_summary,evaluate_checkpoint
from .utils import (autocast,effective_precision,seed_everything,capture_rng,restore_rng,provenance,Tracking,write_json)

TRAINING_STATE_FORMAT='most-freq-training-state-v1'
# Execution identity and total budget may change; data/geometry/optimizer/ID evaluation controls cannot.
RESUME_SETTINGS=('seed','device','precision','batch_size','train_min_length','train_max_length','sampler',
 'vocab_size','min_count_gap','learning_rate','warmup_steps','weight_decay','grad_clip','validation_lengths',
 'validation_examples','validation_seed','eval_every','eval_batch_size','eval_chunk_size','threads')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    p.add_argument('--attn-implementation',choices=['auto','eager','flash_attention_2'],default='auto')
    p.add_argument('--model',choices=['trace_relay','no_carry','swa'],default='trace_relay')
    p.add_argument('--hidden-size',type=int,default=64);p.add_argument('--intermediate-size',type=int,default=256)
    p.add_argument('--heads',type=int,default=4);p.add_argument('--center-width',type=int,default=64)
    p.add_argument('--trace-widths',type=int,nargs='+',default=None)
    p.add_argument('--left-windows',type=int,nargs='+',default=[15,15,15])
    p.add_argument('--right-windows',type=int,nargs='+',default=[7,7,7])
    p.add_argument('--relay-strides',type=int,nargs='+',default=[8,8,8])
    p.add_argument('--skip-pairs',nargs='*',default=[])
    p.add_argument('--vocab-size',type=int,default=4);p.add_argument('--sampler',choices=SAMPLERS,default='separated_counts')
    p.add_argument('--min-count-gap',type=int,default=4)
    p.add_argument('--train-min-length',type=int,default=32);p.add_argument('--train-max-length',type=int,default=64)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--validation-lengths',type=int,nargs='+',default=[32,48,64])
    p.add_argument('--validation-examples',type=int,default=512);p.add_argument('--validation-seed',type=int,default=20261007)
    p.add_argument('--eval-every',type=int,default=250);p.add_argument('--eval-batch-size',type=int,default=64)
    p.add_argument('--eval-chunk-size',type=int,default=0);p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--warmup-steps',type=int,default=100)
    p.add_argument('--weight-decay',type=float,default=.01);p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--early-stop-accuracy',type=float,default=.99);p.add_argument('--early-stop-passes',type=int,default=2)
    p.add_argument('--disable-early-stop',dest='early_stop_accuracy',action='store_const',const=None)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--final-evaluation',action='store_true')
    p.add_argument('--eval-lengths',type=int,nargs='+',default=[32,48,64,96,128,256])
    p.add_argument('--eval-examples',type=int,default=1024);p.add_argument('--eval-seed',type=int,default=20261006)
    p.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    p.add_argument('--wandb-project',default='trace relay most freq');p.add_argument('--wandb-entity',default=None)
    return p


def validate_args(a):
    for length in (a.train_min_length,a.train_max_length):
        validate_sampler(a.vocab_size,a.sampler,a.min_count_gap,length)
    if a.train_min_length>a.train_max_length:raise ValueError('Training length range must be increasing')
    for panel in (a.validation_lengths,a.eval_lengths):
        if not panel or len(panel)!=len(set(panel)):raise ValueError('Length panels must be nonempty and unique')
        for length in panel:validate_sampler(a.vocab_size,a.sampler,a.min_count_gap,length)
    if any(not a.train_min_length<=t<=a.train_max_length for t in a.validation_lengths):
        raise ValueError('ID validation lengths must be within the training range')
    if len({a.seed,a.validation_seed,a.eval_seed})!=3 or not all(0<=s<2**63 for s in (a.seed,a.validation_seed,a.eval_seed)):
        raise ValueError('Train/validation/final seeds must be distinct and in 0..2**63-1')
    if min(a.steps,a.batch_size,a.validation_examples,a.eval_examples,a.eval_every,a.eval_batch_size,a.log_every,a.threads,a.early_stop_passes)<1:
        raise ValueError('Budgets, batches, intervals, examples and passes must be positive')
    if min(a.warmup_steps,a.eval_chunk_size)<0:raise ValueError('Warmup/chunk size must be nonnegative')
    if a.early_stop_accuracy is not None and (not math.isfinite(a.early_stop_accuracy) or not 0<=a.early_stop_accuracy<=1):
        raise ValueError('Early-stop threshold must be finite in [0,1]')
    if not all(math.isfinite(value) for value in (a.learning_rate,a.weight_decay,a.grad_clip)) or a.learning_rate<=0 or a.weight_decay<0 or a.grad_clip<=0:
        raise ValueError('Invalid optimizer settings')


def architecture(a):
    skips=[]
    for value in a.skip_pairs:
        try:
            parts=value.split(':')
            if len(parts)!=2:raise ValueError()
            skips.append([int(part) for part in parts])
        except ValueError:raise ValueError('skip-pairs must use source:destination') from None
    return build_config(vocab_size=a.vocab_size,model=a.model,hidden_size=a.hidden_size,
        intermediate_size=a.intermediate_size,heads=a.heads,center_width=a.center_width,trace_widths=a.trace_widths,
        left_windows=a.left_windows,right_windows=a.right_windows,relay_strides=a.relay_strides,skip_pairs=skips,
        device=a.device,precision=a.precision,attn_implementation=a.attn_implementation)


def selection_key(summary):
    return summary['mean_generation_exact_accuracy'],-summary['mean_teacher_forced_cross_entropy']


def run_training(a):
    validate_args(a);config=architecture(a)
    if a.output.exists():raise FileExistsError(f'Refusing to overwrite {a.output}')
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(a.threads);seed_everything(a.seed)
    generator=torch.Generator().manual_seed(a.seed)
    previous=None;historical_best=None;changes=[]
    if a.resume is None:
        model=MostFreqModel(config,a.eval_chunk_size).to(a.device)
    else:
        model,previous,digest=load_checkpoint(a.resume,device=a.device,precision=a.precision,chunk_size=a.eval_chunk_size)
        if previous['config']!=config:raise ValueError('Resume architecture/semantics mismatch')
        if previous.get('training_state_format')!=TRAINING_STATE_FORMAT:raise ValueError('Resume requires optimizer/RNG training state')
        if a.steps<previous['protocol']['training']['steps'] or a.steps<=previous['step']:
            raise ValueError('Resume budget must not shrink and must exceed the saved update')
        for key in RESUME_SETTINGS:
            if previous['protocol']['training'][key]!=getattr(a,key):raise ValueError(f'Resume setting mismatch: {key}')
        historical_best=a.resume.parent/'best.pt'
        if not historical_best.exists():raise ValueError('Historical best.pt is required beside resume checkpoint')
        best=torch.load(historical_best,map_location='cpu',weights_only=True)
        if best.get('format')!=CHECKPOINT_FORMAT or best['config']!=config or best['step']!=previous['best_step'] or best['best_validation']!=previous['best_validation']:
            raise ValueError('Historical best checkpoint mismatch')
        changes=list(previous['protocol'].get('resume_changes',[]))
        for key in ('steps','early_stop_accuracy','early_stop_passes'):
            old=previous['protocol']['training'][key]
            if old!=getattr(a,key):changes.append(dict(step=previous['step'],setting=key,old=old,new=getattr(a,key)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    step=source_tokens=physical_tokens=supervised_targets=best_step=streak=0
    best_summary=last_summary=None;previous_wall=0.
    if previous is not None:
        optimizer.load_state_dict(previous['optimizer']);generator.set_state(previous['training_rng'])
        step=previous['step'];source_tokens=previous['source_tokens'];physical_tokens=previous['physical_tokens']
        supervised_targets=previous['supervised_targets'];best_step=previous['best_step']
        best_summary=previous['best_validation'];last_summary=previous['validation'];previous_wall=previous['wall_seconds']
        old=previous['protocol']['training']
        if (old['early_stop_accuracy'],old['early_stop_passes'])==(a.early_stop_accuracy,a.early_stop_passes):
            streak=previous['early_stop_streak']
    protocol=dict(format='trace-relay-most-freq-protocol-v1',task='most_freq',semantics_version=SEMANTICS_VERSION,
        config=config,sampler=dict(name=a.sampler,min_count_gap=a.min_count_gap,version=SAMPLER_VERSION),
        training={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},provenance=provenance(),
        precision=effective_precision(a.device,a.precision),parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        resume=dict(mode='fresh',start_step=0) if previous is None else dict(mode='full',start_step=step,checkpoint=str(a.resume.resolve()),sha256=digest),
        resume_changes=changes,checkpoint_selection='Mean per-length ID generation exact, tie-break lower mean TF CE',
        early_stopping=dict(metric='mean ID generation exact',threshold=a.early_stop_accuracy,consecutive=a.early_stop_passes,
                            same_validation_set=True,ood_used=False),
        adaptation='Causal Most-Freq diagnostic: original architecture/data/supervision/scores are not reproduced',
        sampler_uniform_count_combinations=False if a.sampler=='separated_counts' else None,
        ood_used_for_training_control=False)
    a.output.mkdir(parents=True);write_json(a.output/'protocol.json',protocol)
    if previous is not None:
        shutil.copyfile(a.resume,a.output/'last.pt');shutil.copyfile(historical_best,a.output/'best.pt')
    global_rng=capture_rng() if previous is None else previous['global_rng']
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter();start_step=step;start_tokens=physical_tokens
    interrupted=False;stop_reason='step_budget'
    with Tracking(a.output,protocol,a.wandb_mode,a.wandb_project,a.wandb_entity) as tracking,(a.output/'metrics.jsonl').open('w') as log:
        restore_rng(global_rng)
        def record(row):
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();tracking.log(row)
            print(json.dumps(row,allow_nan=False),flush=True)
        def checkpoint(filename):
            save_checkpoint(a.output/filename,model,protocol=protocol,training_state_format=TRAINING_STATE_FORMAT,
                optimizer=optimizer.state_dict(),training_rng=generator.get_state(),global_rng=capture_rng(),step=step,
                source_tokens=source_tokens,physical_tokens=physical_tokens,supervised_targets=supervised_targets,
                best_step=best_step,best_validation=best_summary,validation=last_summary,early_stop_streak=streak,
                wall_seconds=previous_wall+time.perf_counter()-start)
        try:
            for update in range(step+1,a.steps+1):
                if a.early_stop_accuracy is not None and streak>=a.early_stop_passes:
                    stop_reason='validation_threshold';break
                length=int(torch.randint(a.train_min_length,a.train_max_length+1,(),generator=generator))
                batch=generate_batch(a.batch_size,length,vocab_size=a.vocab_size,sampler=a.sampler,
                                      min_count_gap=a.min_count_gap,generator=generator)
                inputs=batch['inputs'].to(a.device);mask=batch['attention_mask'].to(a.device)
                labels=batch['target_labels'].to(a.device)
                lr=a.learning_rate*min(1.,update/max(1,a.warmup_steps))
                for group in optimizer.param_groups:group['lr']=lr
                model.train();optimizer.zero_grad(set_to_none=True)
                with autocast(a.device,a.precision):
                    logits=model(inputs,mask);loss=sequence_loss(logits,labels,mask)
                if not loss.isfinite():raise FloatingPointError('Non-finite loss')
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip,error_if_nonfinite=True)
                optimizer.step();step=update
                targets=shifted_targets(labels,mask);valid=targets.ne(-100)
                source_tokens+=a.batch_size*length;physical_tokens+=int(mask.sum());supervised_targets+=int(valid.sum())
                if step==start_step+1 or step%a.log_every==0 or step==a.steps:
                    hit=logits[:,:-1].argmax(-1).eq(targets)
                    record(dict(phase='train',step=step,loss=float(loss.detach()),
                        teacher_forced_minibatch_token_accuracy=float(hit[valid].float().mean()),
                        teacher_forced_minibatch_exact_accuracy=float((hit|~valid).all(-1).float().mean()),
                        gradient_norm=float(grad_norm),learning_rate=lr,source_length=length,
                        source_tokens=source_tokens,physical_tokens=physical_tokens,supervised_targets=supervised_targets,
                        wall_seconds=previous_wall+time.perf_counter()-start))
                if step%a.eval_every==0 or step==a.steps:
                    rows=evaluate_lengths(model,a.validation_lengths,examples=a.validation_examples,
                        batch_size=a.eval_batch_size,seed=a.validation_seed,sampler=a.sampler,min_count_gap=a.min_count_gap,
                        device=a.device,precision=a.precision)
                    last_summary=validation_summary(rows)
                    score=last_summary['mean_generation_exact_accuracy']
                    passed=a.early_stop_accuracy is not None and score>=a.early_stop_accuracy
                    streak=streak+1 if passed else 0
                    per_length={f"length_{row['length']}/{name}":row[name] for row in rows for name in (
                        'generation_exact_accuracy','teacher_forced_token_accuracy','teacher_forced_exact_accuracy','teacher_forced_cross_entropy')}
                    record(dict(phase='val',step=step,metrics=per_length,**{k:v for k,v in last_summary.items() if k!='rows'},
                                early_stop_streak=streak,wall_seconds=previous_wall+time.perf_counter()-start))
                    if best_summary is None or selection_key(last_summary)>selection_key(best_summary):
                        best_summary=last_summary;best_step=step;checkpoint('best.pt')
                    checkpoint('last.pt')
                    if passed and streak>=a.early_stop_passes:
                        stop_reason='validation_threshold';break
        except KeyboardInterrupt:
            interrupted=True;stop_reason='interrupted'
            # Keep the last complete validation state, never a partially sampled/update state.
            print('Interrupted; sweep must not launch another run.',flush=True)
        if a.device=='cuda':torch.cuda.synchronize()
        wall=time.perf_counter()-start
        result=dict(format='trace-relay-most-freq-training-v1',steps=step,best_step=best_step,
            best_validation=best_summary,last_validation=last_summary,early_stop_streak=streak,
            source_tokens=source_tokens,physical_tokens=physical_tokens,supervised_targets=supervised_targets,
            wall_seconds=previous_wall+wall,segment_wall_seconds=wall,segment_steps=step-start_step,
            measured_physical_tokens_per_second=(physical_tokens-start_tokens)/wall,
            parameter_count=protocol['parameter_count'],trainable_parameters=protocol['trainable_parameters'],
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
            stop_reason=stop_reason,interrupted=interrupted,ood_evaluated=False)
        write_json(a.output/'results.json',result);tracking.summary(result)
        if a.final_evaluation and not interrupted:
            try:
                final=evaluate_checkpoint(a.output/'best.pt',a.output/'final_eval',lengths=a.eval_lengths,
                    examples=a.eval_examples,batch_size=a.eval_batch_size,seed=a.eval_seed,device=a.device,
                    precision=a.precision,chunk_size=a.eval_chunk_size)
                for row in final['rows']:
                    record(dict(phase='final',step=best_step,length=row['length'],
                        metrics={f"length_{row['length']}/{name}":row[name] for name in (
                            'generation_exact_accuracy','generation_top1_accuracy','generation_top2_prefix_accuracy')}))
                result['ood_evaluated']=any(row['length_ood'] or row['distribution_shift'] for row in final['rows'])
                result['final_evaluation']=str(a.output/'final_eval/results.json')
            except KeyboardInterrupt:
                result.update(interrupted=True,stop_reason='interrupted_final_evaluation')
            write_json(a.output/'results.json',result)
    return result


def main(argv=None):return run_training(build_parser().parse_args(argv))


if __name__=='__main__':main()
