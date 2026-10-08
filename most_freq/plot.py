"""Optional raw-seed and mean±sample-SD plots with strict protocol matching."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics

from .utils import write_json,write_csv


METRICS=('generation_exact_accuracy','generation_top1_accuracy','generation_top2_prefix_accuracy')
IGNORED_TRAINING=('output','resume','seed','wandb_mode','wandb_project','wandb_entity','log_every',
                  'threads','final_evaluation','eval_seed','eval_examples','eval_lengths')


def profile(config,training,*,pair=False):
    config=copy.deepcopy(config);training=copy.deepcopy(training)
    for key in IGNORED_TRAINING:training.pop(key,None)
    if pair:
        config.pop('family',None);config['backbone'].pop('recurrent_carry',None);training.pop('model',None)
    return dict(config=config,training=training)


def stats(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def aggregate(paths):
    validation=[];raw=[];baseline_raw=[];records={};profiles={};eval_profiles={}
    validation_ids=set()
    for path in map(Path,paths):
        report_paths=[]
        if path.is_dir():
            protocol=json.loads((path/'protocol.json').read_text())
            config,training=protocol['config'],protocol['training']
            family=config['family'];center=config['backbone']['trace_width'][config['backbone']['num_hidden_layers']//2]
            cohort=(family,center)
            current=profile(config,training)
            if cohort in profiles and profiles[cohort]!=current:raise ValueError('Different training protocols/geometry cannot be pooled')
            profiles[cohort]=current
            for line in (path/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['phase']!='val':continue
                identity=cohort+(training['seed'],row['step'])
                if identity in validation_ids:raise ValueError('Duplicate validation run/seed/step')
                validation_ids.add(identity)
                validation.append(dict(model=family,center_width=center,training_seed=training['seed'],
                    step=row['step'],metric='mean_generation_exact_accuracy',value=row['mean_generation_exact_accuracy']))
            if (path/'final_eval/results.json').exists():report_paths.append(path/'final_eval/results.json')
        else:report_paths.append(path)
        for report_path in report_paths:
            report=json.loads(report_path.read_text())
            if report.get('format')!='trace-relay-most-freq-evaluation-v1':raise ValueError('Expected Most-Freq evaluation')
            for row in report['rows']:
                identity=(row['model'],row['center_width'],row['training_seed'],row['sampler'],row['min_count_gap'],row['length'])
                if identity in records:raise ValueError('Duplicate evaluation model/seed/distribution/length')
                cohort=identity[:2]+identity[3:5]
                current=dict(profile=profile(report['config'],report['training_protocol']),
                             evaluation_seed=report['evaluation_seed'],precision=report['precision'],examples=report['examples_per_length'],
                             sampler=report['sampler'])
                if cohort in eval_profiles and eval_profiles[cohort]!=current:raise ValueError('Different sampler/margin/geometry/budget/evaluation protocols cannot be pooled')
                eval_profiles[cohort]=current;records[identity]=(row,report)
                common={key:row[key] for key in ('model','center_width','training_seed','sampler','min_count_gap','length','split','examples_sha256')}
                for metric in METRICS:raw.append(dict(**common,metric=metric,value=row[metric]))
                for baseline,values in row['baselines'].items():
                    baseline_raw.append(dict(**common,baseline=baseline,metric='generation_exact_accuracy',value=values['generation_exact_accuracy']))
    groups=defaultdict(list);fingerprints={};base_groups=defaultdict(list);val_groups=defaultdict(list)
    for row in raw:
        key=(row['model'],row['center_width'],row['sampler'],row['min_count_gap'],row['length'],row['metric'])
        if key in fingerprints and fingerprints[key]!=row['examples_sha256']:raise ValueError('Evaluation fingerprints differ across training seeds')
        fingerprints[key]=row['examples_sha256'];groups[key].append(row['value'])
    for row in baseline_raw:
        key=(row['model'],row['center_width'],row['sampler'],row['min_count_gap'],row['length'],row['baseline'])
        base_groups[key].append(row['value'])
    for row in validation:val_groups[(row['model'],row['center_width'],row['step'])].append(row['value'])
    summary=[dict(zip(('model','center_width','sampler','min_count_gap','length','metric'),key),**stats(values)) for key,values in sorted(groups.items())]
    baseline_summary=[dict(zip(('model','center_width','sampler','min_count_gap','length','baseline'),key),**stats(values)) for key,values in sorted(base_groups.items())]
    val_summary=[dict(model=key[0],center_width=key[1],step=key[2],**stats(values)) for key,values in sorted(val_groups.items())]
    gaps=[]
    for identity,(row,report) in records.items():
        if identity[0]!='trace_relay':continue
        other=records.get(('no_carry',*identity[1:]))
        if other is None:continue
        baseline,other_report=other
        if profile(report['config'],report['training_protocol'],pair=True)!=profile(other_report['config'],other_report['training_protocol'],pair=True):
            raise ValueError('Carry gap requires identical training protocol/budget and geometry')
        if row['examples_sha256']!=baseline['examples_sha256'] or report['precision']!=other_report['precision']:
            raise ValueError('Carry gap requires matched evaluation sources/precision')
        if report['config']['backbone']['recurrent_carry'] is not True or other_report['config']['backbone']['recurrent_carry'] is not False:
            raise ValueError('Carry flags disagree with model families')
        gaps.append(dict(center_width=identity[1],training_seed=identity[2],sampler=identity[3],min_count_gap=identity[4],
                         length=identity[5],gap=row['generation_exact_accuracy']-baseline['generation_exact_accuracy']))
    return dict(validation_raw=validation,validation_summary=val_summary,raw=raw,summary=summary,
                baseline_raw=baseline_raw,baseline_summary=baseline_summary,carry_gaps=gaps)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('runs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():p.error('Use a fresh plot output directory')
    data=aggregate(args.runs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True);write_json(args.output/'results.json',data)
    for name,rows in data.items():write_csv(args.output/f'{name}.csv',rows)
    def draw(ax,rows,x,label):
        rows=sorted(rows,key=lambda r:r[x])
        line,=ax.plot([r[x] for r in rows],[r['mean'] for r in rows],'o-',label=f'{label}, n={min(r["n"] for r in rows)}')
        for r in rows:
            if r['sample_sd'] is not None:ax.errorbar(r[x],r['mean'],yerr=r['sample_sd'],color=line.get_color(),capsize=3)
        return line.get_color()
    if data['validation_summary']:
        fig,ax=plt.subplots(figsize=(8,5));groups=defaultdict(list)
        for r in data['validation_summary']:groups[(r['model'],r['center_width'])].append(r)
        for (model,center),rows in groups.items():
            color=draw(ax,rows,'step',f'{model} center{center}')
            raw=[r for r in data['validation_raw'] if r['model']==model and r['center_width']==center]
            ax.scatter([r['step'] for r in raw],[r['value'] for r in raw],s=12,alpha=.4,color=color)
        ax.set(xlabel='Optimizer steps',ylabel='Mean ID generation exact');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(args.output/'id_validation.png',dpi=160);plt.close(fig)
    for metric in METRICS:
        if not data['summary']:continue
        fig,ax=plt.subplots(figsize=(8,5));groups=defaultdict(list)
        for r in data['summary']:
            if r['metric']==metric:groups[(r['model'],r['center_width'],r['sampler'],r['min_count_gap'])].append(r)
        for (model,center,sampler,gap),rows in groups.items():
            color=draw(ax,rows,'length',f'{model} c{center} {sampler} gap{gap}')
            raw=[r for r in data['raw'] if (r['model'],r['center_width'],r['sampler'],r['min_count_gap'],r['metric'])==(model,center,sampler,gap,metric)]
            ax.scatter([r['length'] for r in raw],[r['value'] for r in raw],s=12,alpha=.4,color=color)
        ax.set(xlabel='Source length',ylabel=metric);ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(args.output/f'{metric}.png',dpi=160);plt.close(fig)
    # One panel per neural family/center/distribution avoids accidental mixing of shortcut baselines.
    panels=defaultdict(list)
    for row in data['baseline_summary']:panels[(row['model'],row['center_width'],row['sampler'],row['min_count_gap'])].append(row)
    for key,rows in panels.items():
        fig,ax=plt.subplots(figsize=(8,5))
        model,center,sampler,gap=key
        neural=[r for r in data['summary'] if (r['model'],r['center_width'],r['sampler'],r['min_count_gap'],r['metric'])==(*key,'generation_exact_accuracy')]
        draw(ax,neural,'length',f'{model} center{center}')
        for name in sorted({r['baseline'] for r in rows}):draw(ax,[r for r in rows if r['baseline']==name],'length',name)
        ax.set(xlabel='Source length',ylabel='Generation exact',title=f'{sampler} gap={gap}')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(args.output/f'baseline_comparison-{model}-center{center}-{sampler}-gap{gap}.png',dpi=160);plt.close(fig)
    return data


if __name__=='__main__':main()
