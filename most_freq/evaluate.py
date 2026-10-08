"""Source-only generation evaluation, deterministic source plans and shortcut baselines."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .task import (SEMANTICS_VERSION,SAMPLER_VERSION,SAMPLERS,evaluation_plan,materialize,oracle,
                   validate_sampler,shifted_targets)
from .model import load_checkpoint
from .utils import autocast,effective_precision,write_json,write_csv

METRIC_NUMERATORS = {
 'generation_exact_accuracy':('exact_correct','sequence_count'),
 'generation_top1_accuracy':('top1_correct','sequence_count'),
 'generation_top2_prefix_accuracy':('top2_correct','sequence_count'),
 'generation_symbol_accuracy':('symbol_correct','symbol_count'),
 'generation_set_exact_accuracy':('set_correct','sequence_count'),
 'generation_unique_valid_accuracy':('unique_valid','sequence_count'),
 'generation_termination_accuracy':('terminated_count','sequence_count')}


def score_predictions(predictions, targets, vocab_size):
    """Compare raw class sequences. No sorting, deduplication or forced EOS correction."""
    if len(predictions)!=len(targets) or not predictions:raise ValueError('Nonempty matching sequence lists required')
    total=dict(sequence_count=len(targets),symbol_count=0,symbol_correct=0,exact_correct=0,
               top1_correct=0,top2_correct=0,set_correct=0,unique_valid=0,terminated_count=0)
    for predicted,target in zip(predictions,targets):
        if not target or target[-1]!=vocab_size:raise ValueError('Gold target must end with EOS')
        gold=target[:-1]
        if not gold:raise ValueError('Gold source must contain at least one symbol')
        ended=vocab_size in predicted
        # Only the actual first-EOS prefix counts as output, matching the decoder's termination rule.
        output=predicted[:predicted.index(vocab_size)+1] if ended else predicted
        ranking=output[:-1] if ended else output
        total['symbol_count']+=len(gold)
        total['symbol_correct']+=sum(index<len(ranking) and value==ranking[index] for index,value in enumerate(gold))
        total['exact_correct']+=int(ended and output==target)
        total['top1_correct']+=int(bool(ranking) and ranking[0]==gold[0])
        width=min(2,len(gold))
        total['top2_correct']+=int(len(ranking)>=width and ranking[:width]==gold[:width])
        total['set_correct']+=int(len(ranking)==len(gold) and len(set(ranking))==len(ranking) and set(ranking)==set(gold))
        total['unique_valid']+=int(bool(ranking) and len(ranking)<=len(gold) and len(set(ranking))==len(ranking)
                                   and all(0<=value<vocab_size and value in gold for value in ranking))
        total['terminated_count']+=int(ended)
    return total


def metric_values(total):
    return {**total,**{name:total[numerator]/total[denominator] for name,(numerator,denominator) in METRIC_NUMERATORS.items()}}


def baseline_outputs(source, vocab_size, suffix_lengths=(16,32)):
    validate_sampler(vocab_size,'iid_uniform',1,source.shape[1])
    full=oracle(source,vocab_size)
    # Each baseline computes its own allowed information. Suffix does not consult full's metadata.
    first=[]
    for counts,positions in zip(full['counts'],full['first_positions']):
        present=counts.gt(0).nonzero().flatten().tolist()
        first.append(sorted(present,key=lambda symbol:int(positions[symbol]))+[vocab_size])
    result={'fixed_symbol_order':[list(range(vocab_size))+[vocab_size] for _ in source],
            'first_appearance':first}
    for width in suffix_lengths:
        if type(width) is not int or width<1:raise ValueError('Suffix widths must be positive integers')
        suffix=source[:,-width:]
        result[f'suffix_frequency_{width}']=[rank+[vocab_size] for rank in oracle(suffix,vocab_size)['rankings']]
    return result


def _accumulate(total,values):
    for key,value in values.items():total[key]=total.get(key,0)+value


@torch.no_grad()
def evaluate_length(model,length,*,examples=512,batch_size=64,seed=20261007,sampler='separated_counts',
                    min_count_gap=4,device='cpu',precision='fp32',suffix_lengths=(16,32),example_limit=0):
    if type(batch_size) is not int or batch_size<1 or type(example_limit) is not int or example_limit<0:
        raise ValueError('Positive batch size and nonnegative example limit required')
    v=model.vocab_size
    plan,fingerprint=evaluation_plan(examples,length,seed,vocab_size=v,sampler=sampler,min_count_gap=min_count_gap)
    generated_total={};baseline_totals={};tf_correct=tf_exact=tf_count=0;losses=[]
    disagreements=0;disagreement_margins=[];examples_saved=[];saved_success=saved_failure=0
    previous_mode=model.training;model.eval()
    try:
        for start in range(0,examples,batch_size):
            batch=materialize(plan[start:start+batch_size],v)
            mask=batch['attention_mask'].to(device);labels=batch['target_labels'].to(device)
            targets=shifted_targets(labels,mask)
            with autocast(device,precision):
                scores=model(batch['inputs'].to(device),mask)[:,:-1].float()
                output=model.generate(batch['source'],prefill_chunk_size=model.eval_chunk_size)
            if not scores.isfinite().all():raise FloatingPointError('Non-finite teacher-forced logits')
            valid=targets.ne(-100);tf_prediction=scores.argmax(-1)
            correct=tf_prediction.eq(targets)|~valid
            exact=correct.all(-1).cpu()
            tf_correct+=int((tf_prediction.eq(targets)&valid).sum());tf_count+=int(valid.sum());tf_exact+=int(exact.sum())
            losses.extend(F.cross_entropy(scores[valid],targets[valid],reduction='none').double().cpu().tolist())
            predictions=[row[:int(count)].tolist() for row,count in zip(output['tokens'].cpu(),output['lengths'].cpu())]
            gold=[row[mask].tolist() for row,mask in zip(batch['target_classes'],batch['target_mask'])]
            _accumulate(generated_total,score_predictions(predictions,gold,v))
            free_exact=torch.tensor([pred==target for pred,target in zip(predictions,gold)])
            different=exact!=free_exact
            disagreements+=int(different.sum())
            if different.any():
                margin=scores.topk(2,-1).values.diff(dim=-1).abs().squeeze(-1)
                for row in different.nonzero().flatten().tolist():
                    disagreement_margins.append(float(margin[row][valid[row]].min()))
            for name,baseline in baseline_outputs(batch['source'],v,suffix_lengths).items():
                _accumulate(baseline_totals.setdefault(name,{}),score_predictions(baseline,gold,v))
            for index,(pred,target) in enumerate(zip(predictions,gold)):
                success=pred==target
                # Bound successes and failures independently; metadata never goes back into model inputs.
                if (saved_success if success else saved_failure)>=example_limit:continue
                examples_saved.append(dict(source=batch['source'][index].tolist(),counts=batch['counts'][index].tolist(),
                    first_positions=batch['first_positions'][index].tolist(),gold_ranking=target[:-1],
                    generated_classes=pred,generated_ranking=pred[:-1] if pred and pred[-1]==v else pred,
                    terminated=bool(output['terminated'][index]),exact=success))
                if success:saved_success+=1
                else:saved_failure+=1
    finally:model.train(previous_mode)
    return dict(length=length,source_length=length,vocab_size=v,sampler=sampler,min_count_gap=min_count_gap,
        sampler_version=SAMPLER_VERSION,examples_sha256=fingerprint,**metric_values(generated_total),
        teacher_forced_token_accuracy=tf_correct/tf_count,teacher_forced_exact_accuracy=tf_exact/examples,
        teacher_forced_cross_entropy=math.fsum(losses)/tf_count,teacher_forced_cross_entropy_sum=math.fsum(losses),
        teacher_forced_correct_count=tf_correct,teacher_forced_target_count=tf_count,teacher_forced_exact_count=tf_exact,
        tf_generation_exact_disagreement_count=disagreements,
        minimum_teacher_forced_margin_on_exact_disagreement=min(disagreement_margins) if disagreement_margins else None,
        physical_token_count=examples*(length+1)+tf_count,
        baselines={name:metric_values(total) for name,total in baseline_totals.items()},
        suffix_oracle_cases={f'suffix_frequency_{width}':width>=length for width in suffix_lengths},saved_examples=examples_saved)


def evaluate_lengths(model,lengths,**kwargs):
    if not lengths or len(set(lengths))!=len(lengths):raise ValueError('Evaluation lengths must be nonempty and unique')
    return [evaluate_length(model,length,**kwargs) for length in lengths]


def validation_summary(rows):
    return dict(mean_generation_exact_accuracy=sum(r['generation_exact_accuracy'] for r in rows)/len(rows),
        mean_teacher_forced_token_accuracy=sum(r['teacher_forced_token_accuracy'] for r in rows)/len(rows),
        mean_teacher_forced_cross_entropy=sum(r['teacher_forced_cross_entropy'] for r in rows)/len(rows),rows=rows)


def evaluate_checkpoint(checkpoint,output,*,lengths=(32,48,64,96,128,256),examples=1024,batch_size=64,
                        seed=20261006,device='cpu',precision='auto',attention_backend=None,chunk_size=0,
                        sampler=None,min_count_gap=None,suffix_lengths=(16,32),example_limit=8):
    output=Path(output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    model,checkpoint_data,digest=load_checkpoint(checkpoint,device=device,precision=precision,
                                                attention_backend=attention_backend,chunk_size=chunk_size)
    protocol=checkpoint_data.get('protocol',{})
    training=protocol.get('training',{})
    saved_sampler=protocol.get('sampler',{})
    chosen=sampler if sampler is not None else saved_sampler.get('name','separated_counts')
    gap=min_count_gap if min_count_gap is not None else saved_sampler.get('min_count_gap',4)
    rows=evaluate_lengths(model,lengths,examples=examples,batch_size=batch_size,seed=seed,sampler=chosen,
        min_count_gap=gap,device=device,precision=precision,suffix_lengths=suffix_lengths,example_limit=example_limit)
    distribution_shift=chosen!=saved_sampler.get('name',chosen) or (chosen=='separated_counts' and gap!=saved_sampler.get('min_count_gap',gap))
    center=model.config['backbone']['trace_width'][model.config['backbone']['num_hidden_layers']//2]
    for row in rows:
        within=training.get('train_min_length',1)<=row['length']<=training.get('train_max_length',0)
        row.update(model=model.config['family'],center_width=center,training_seed=training.get('seed'),evaluation_seed=seed,
            split='distribution_shift' if distribution_shift else ('in_distribution' if within else 'length_ood'),
            distribution_shift=distribution_shift,length_ood=not within)
    report=dict(format='trace-relay-most-freq-evaluation-v1',semantics_version=SEMANTICS_VERSION,
        adaptation='Causal Most-Freq diagnostic; original architecture/data/supervision/scores are not reproduced',
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest,training_step=checkpoint_data.get('step'),
        config=model.config,training_protocol=training,provenance=protocol.get('provenance'),
        evaluation_seed=seed,examples_per_length=examples,batch_size=batch_size,chunk_size=chunk_size,
        sampler=dict(name=chosen,min_count_gap=gap,version=SAMPLER_VERSION),
        precision=effective_precision(device,precision),rows=rows)
    write_json(output/'results.json',report)
    write_csv(output/'metrics.csv',[{k:v for k,v in row.items() if k not in ('baselines','suffix_oracle_cases','saved_examples')} for row in rows])
    identity=('model','center_width','training_seed','evaluation_seed','length','sampler','min_count_gap','split','examples_sha256')
    write_csv(output/'baseline_metrics.csv',[{**{k:row[k] for k in identity},'baseline':name,**metrics,
        'full_source_oracle':row['suffix_oracle_cases'].get(name,False)} for row in rows for name,metrics in row['baselines'].items()])
    with (output/'examples.jsonl').open('w') as handle:
        for row in rows:
            for example in row['saved_examples']:handle.write(json.dumps(dict(length=row['length'],sampler=row['sampler'],**example))+'\n')
    for row in rows:
        print(f"T={row['length']} {row['split']}: generation exact={row['generation_exact_accuracy']:.2%}, top1={row['generation_top1_accuracy']:.2%}",flush=True)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',default=[32,48,64,96,128,256])
    parser.add_argument('--examples-per-length',type=int,default=1024);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    parser.add_argument('--attn-implementation',choices=['eager','flash_attention_2'],default=None)
    parser.add_argument('--chunk-size',type=int,default=0);parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--sampler',choices=SAMPLERS,default=None);parser.add_argument('--min-count-gap',type=int,default=None)
    parser.add_argument('--suffix-lengths',type=int,nargs='+',default=[16,32]);parser.add_argument('--example-limit',type=int,default=8)
    args=parser.parse_args(argv)
    if args.threads<1 or args.chunk_size<0:parser.error('threads must be positive and chunk-size nonnegative')
    torch.set_num_threads(args.threads)
    return evaluate_checkpoint(args.checkpoint,args.output,lengths=args.lengths,examples=args.examples_per_length,
        batch_size=args.batch_size,seed=args.seed,device=args.device,precision=args.precision,
        attention_backend=args.attn_implementation,chunk_size=args.chunk_size,sampler=args.sampler,
        min_count_gap=args.min_count_gap,suffix_lengths=args.suffix_lengths,example_limit=args.example_limit)


if __name__=='__main__':main()
