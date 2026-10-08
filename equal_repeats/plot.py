"""Optional length curves and paired carry-minus-no-carry gaps; training needs no plotting dependency."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def aggregate(paths):
    records = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get('format') != 'trace-relay-equal-repeats-evaluation-v1':
            raise ValueError(f'Expected an Equal Repeats evaluation: {path}')
        for row in report['rows']:
            key = (row['model'], row['center_width'], row['training_seed'], row['evaluation_seed'], row['length'])
            if key in records:
                raise ValueError(f'Duplicate evaluated model/seed/length: {key}')
            records[key] = (row, report)
    groups = defaultdict(list)
    for (family, center, _, eval_seed, length), (row, _) in records.items():
        groups[(family, center, eval_seed, length)].append(row['accuracy'])
    summary = [dict(model=family, center_width=center, evaluation_seed=seed, length=length,
                    mean_accuracy=statistics.mean(values), std_accuracy=statistics.stdev(values) if len(values)>1 else 0.,
                    training_seeds=len(values)) for (family,center,seed,length),values in sorted(groups.items())]
    gaps = defaultdict(list)
    for (family, center, train_seed, eval_seed, length), (row, report) in records.items():
        if family != 'trace_relay':
            continue
        other = records.get(('no_carry', center, train_seed, eval_seed, length))
        if other is None:
            continue
        baseline, other_report = other
        if row['examples_sha256'] != baseline['examples_sha256']:
            raise ValueError('Carry gaps require exactly the same evaluated examples')
        a,b = report['config']['backbone'],other_report['config']['backbone']
        for setting in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'num_attention_heads',
                        'trace_width', 'left_window', 'right_window', 'relay_stride', 'skip_pairs', 'relay_enabled'):
            if a[setting] != b[setting]:
                raise ValueError(f'Carry comparison profile differs: {setting}')
        if report['train_length_range'] != other_report['train_length_range']:
            raise ValueError('Carry comparison training length ranges differ')
        gaps[(center,eval_seed,length)].append(row['accuracy']-baseline['accuracy'])
    gap_rows = [dict(center_width=center, evaluation_seed=seed, length=length,
                     mean_gap=statistics.mean(values), std_gap=statistics.stdev(values) if len(values)>1 else 0.,
                     paired_training_seeds=len(values)) for (center,seed,length),values in sorted(gaps.items())]
    return summary, gap_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, nargs='+', help='One or more final_eval/results.json files')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Use a fresh output directory')
    summary,gaps = aggregate(args.results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True)
    for name,rows in [('summary.csv', summary),('carry_gap.csv', gaps)]:
        if rows:
            with (args.output/name).open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for rows,field,std_field,filename,ylabel in [
        (summary,'mean_accuracy','std_accuracy','accuracy_vs_length','Accuracy'),
        (gaps,'mean_gap','std_gap','carry_gap_vs_length','Carry − no-carry accuracy')]:
        if not rows:
            continue
        grouped=defaultdict(list)
        for row in rows:
            label=f"{row['model']} / center {row['center_width']}" if field=='mean_accuracy' else f"center {row['center_width']}"
            grouped[(label,row['evaluation_seed'])].append(row)
        fig,ax=plt.subplots(figsize=(8,5))
        for (label,seed),values in sorted(grouped.items()):
            values.sort(key=lambda r:r['length'])
            x=[r['length'] for r in values];y=[r[field] for r in values]
            ax.errorbar(x,y,yerr=[r[std_field] for r in values],marker='o',capsize=3,label=f'{label}, eval {seed}')
        ax.axhline(1/3 if field=='mean_accuracy' else 0,color='gray',linestyle='--',linewidth=1)
        ax.set_xscale('log',base=2)
        ax.set_xlabel('Sequence length');ax.set_ylabel(ylabel)
        if field=='mean_accuracy':
            ax.set_ylim(0,1.02)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout()
        for extension in ('png','pdf','svg'):
            fig.savefig(args.output/f'{filename}.{extension}',dpi=160)
        plt.close(fig)
    return summary,gaps


if __name__=='__main__':
    main()
