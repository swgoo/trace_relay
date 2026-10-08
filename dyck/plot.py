"""Optional Dyck length/distance/depth plots; matching-distance curves keep lengths separate."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def _profile(report):
    backbone = dict(report['config']['backbone'])
    backbone.pop('recurrent_carry', None)
    return dict(backbone=backbone, train_length_range=report['train_length_range'],
                semantics_version=report['semantics_version'])


def aggregate(paths):
    records = {}
    profiles = {}
    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get('format') != 'trace-relay-dyck-evaluation-v1':
            raise ValueError(f'Expected a Dyck evaluation: {path}')
        for row in report['rows']:
            identity = (row['model'], row['center_width'], row['training_seed'], row['evaluation_seed'],
                        row['bracket_types'], row['max_depth'], row['length'])
            if identity in records:
                raise ValueError(f'Duplicate evaluated model/seed/length: {identity}')
            cohort = identity[:2] + identity[3:6]
            profile = _profile(report)
            if cohort in profiles and profiles[cohort] != profile:
                raise ValueError('Cannot aggregate training seeds with different geometry/protocol')
            profiles[cohort] = profile
            records[identity] = (row, report)
    groups = {kind:defaultdict(list) for kind in ('length', 'distance', 'depth', 'length_gap', 'distance_gap')}
    # Keep bins with zero count, represented by None rather than fake 0% accuracy.
    def add(kind, common, row, field=None):
        extra = () if field is None else (row[field],)
        groups[kind][common+extra].append(row['close_accuracy'])

    for identity, (row, report) in records.items():
        family, center, train_seed, eval_seed, k, m, length = identity
        common = (family, center, eval_seed, k, m, length)
        add('length', common, row)
        for bin_row in row['distance_metrics']:
            add('distance', common, bin_row, 'bin')
        for bin_row in row['depth_metrics']:
            add('depth', common, bin_row, 'preclose_depth')
        if family != 'trace_relay':
            continue
        other = records.get(('no_carry', *identity[1:]))
        if other is None:
            continue  # No pairing across training seeds; no fabricated baseline.
        baseline, other_report = other
        if row['examples_sha256'] != baseline['examples_sha256']:
            raise ValueError('Carry gaps require identical evaluation-example fingerprints')
        if _profile(report) != _profile(other_report):
            raise ValueError('Carry comparison geometry/protocol differs')
        if report['config']['backbone']['recurrent_carry'] is not True or other_report['config']['backbone']['recurrent_carry'] is not False:
            raise ValueError('Carry comparison flags are inconsistent')
        gap_common = ('carry_minus_no_carry', center, eval_seed, k, m, length)
        add('length_gap', gap_common, dict(close_accuracy=row['close_accuracy']-baseline['close_accuracy']))
        other_bins = {r['bin']:r for r in baseline['distance_metrics']}
        if set(other_bins) != {r['bin'] for r in row['distance_metrics']}:
            raise ValueError('Carry comparison distance bins differ')
        for r in row['distance_metrics']:
            b = other_bins[r['bin']]
            if r['close_count'] != b['close_count']:
                raise ValueError('Carry comparison distance counts differ')
            gap = r['close_accuracy']-b['close_accuracy'] if r['close_count'] else None
            add('distance_gap', gap_common, dict(bin=r['bin'], close_accuracy=gap), 'bin')
    result = {}
    for kind, grouped in groups.items():
        rows = []
        for identity, values in sorted(grouped.items()):
            valid = [v for v in values if v is not None]
            row = dict(zip(('model', 'center_width', 'evaluation_seed', 'bracket_types', 'max_depth', 'length'), identity[:6]))
            if len(identity) > 6:
                row['bin' if 'distance' in kind else 'preclose_depth'] = identity[6]
            row.update(mean_accuracy=statistics.mean(valid) if valid else None,
                       std_accuracy=statistics.stdev(valid) if len(valid)>1 else (0. if valid else None),
                       training_seeds=len(values), nonempty_seeds=len(valid))
            rows.append(row)
        result[kind] = rows
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path, nargs='+')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Use a fresh output directory')
    result = aggregate(args.results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True)
    for kind, rows in result.items():
        if not rows:
            continue
        with (args.output/f'{kind}_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        panels = defaultdict(list)
        for row in rows:
            panel = (row['evaluation_seed'], row['bracket_types'], row['max_depth'])
            if 'length' not in kind:
                panel += (row['length'],)
            panels[panel].append(row)
        for panel, values in panels.items():
            x_field = 'length' if 'length' in kind else ('bin' if 'distance' in kind else 'preclose_depth')
            labels = sorted({r[x_field] for r in values}, key=lambda v:int(str(v).split('..')[0].lstrip('>')))
            grouped = defaultdict(list)
            for row in values:
                grouped[(row['model'], row['center_width'])].append(row)
            fig, ax = plt.subplots(figsize=(9, 5))
            for (model, center), line in sorted(grouped.items()):
                line.sort(key=lambda r:labels.index(r[x_field]))
                x = [labels.index(r[x_field]) if x_field == 'bin' else r[x_field] for r in line]
                y = [r['mean_accuracy'] if r['mean_accuracy'] is not None else float('nan') for r in line]
                sd = [r['std_accuracy'] if r['std_accuracy'] is not None else float('nan') for r in line]
                ax.errorbar(x, y, yerr=sd, marker='o', capsize=3, label=f'{model} / center {center}')
            is_gap = kind.endswith('_gap')
            ax.axhline(0 if is_gap else 1/panel[1], color='gray', linestyle='--', linewidth=1)
            if x_field == 'length':
                ax.set_xscale('log', base=2)
                ax.set_xticks(labels, labels=[str(x) for x in labels])
            elif x_field == 'bin':
                ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha='right')
            if not is_gap:
                ax.set_ylim(0, 1.02)
            ax.set_xlabel({'length':'Sequence length', 'bin':'Matching opener distance', 'preclose_depth':'Pre-close stack depth'}[x_field])
            ax.set_ylabel('Carry - no-carry accuracy' if is_gap else 'Closing-type accuracy')
            ax.set_title(f'k={panel[1]}, m={panel[2]}, eval seed={panel[0]}' + (f', T={panel[3]}' if len(panel)>3 else ''))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.tight_layout()
            stem = f'{kind}-k{panel[1]}-m{panel[2]}-eval{panel[0]}' + (f'-T{panel[3]}' if len(panel)>3 else '')
            for extension in ('png', 'pdf', 'svg'):
                fig.savefig(args.output/f'{stem}.{extension}', dpi=160)
            plt.close(fig)
    (args.output/'aggregation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    main()
