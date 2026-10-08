"""Print/run the 9 carry-only Dyck runs sequentially; no overwrite or OOD selection."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--centers', type=int, nargs='+', choices=[64, 32, 16], default=[64, 32, 16])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--models', nargs='+', choices=['trace_relay', 'no_carry', 'swa'], default=['trace_relay'])
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs')
    p.add_argument('--python', default=str(ROOT/'.venv/bin/python'))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--early-stop-accuracy', type=float, default=.99)
    p.add_argument('--early-stop-passes', type=int, default=2)
    p.add_argument('--disable-early-stop', action='store_true')
    p.add_argument('--wait-pid', type=int, help='Wait for an existing training process before the sequential sweep')
    p.add_argument('extra', nargs=argparse.REMAINDER, help='Shared trainer overrides after --')
    args = p.parse_args(argv)
    if min(args.seeds) < 0 or any(len(set(values)) != len(values) for values in (args.centers, args.seeds, args.models)):
        p.error('Axes must be unique and seeds nonnegative')
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    protected = {'--output', '--center-width', '--trace-widths', '--seed', '--model', '--resume',
                 '--early-stop-accuracy', '--early-stop-passes', '--disable-early-stop'}
    if any(word.split('=')[0] in protected for word in extra):
        p.error('Use sweep axes to set output/model/center/seed; resume is not a sweep option')
    if args.wait_pid is not None and args.wait_pid < 1:
        p.error('wait-pid must be positive')
    jobs = []
    # Keep explicit model variants adjacent when requested; default is carry only.
    for seed in args.seeds:
        for center in args.centers:
            for family in args.models:
                name = f'dyck-{family}-center{center}-seed{seed}'
                output = args.output_root/name
                log = output.with_suffix('.log')
                if output.exists() or log.exists():
                    p.error(f'Refusing existing output: {output}')
                command = [args.python, '-u', '-m', 'dyck.train',
                           '--output', str(output), '--device', args.device, '--model', family,
                           '--center-width', str(center), '--seed', str(seed), '--final-evaluation',
                           '--wandb-mode', 'online']
                if not args.disable_early_stop:
                    command += ['--early-stop-accuracy', str(args.early_stop_accuracy),
                                '--early-stop-passes', str(args.early_stop_passes)]
                command += extra
                jobs.append((command, output, log))
    for command, _, _ in jobs:
        print(shlex.join(command), flush=True)
    if not args.execute:
        print(f'{len(jobs)} commands printed; no training started. Add --execute to run.')
        return jobs
    if args.wait_pid is not None:
        import time
        proc = Path(f'/proc/{args.wait_pid}/stat')
        def process_identity():
            try:
                fields = proc.read_text().rsplit(')', 1)[1].split()
                return fields[19] if fields[0] != 'Z' else None
            except FileNotFoundError:
                return None
        identity = process_identity()
        print(f'Waiting for PID {args.wait_pid} before starting {len(jobs)} jobs.', flush=True)
        while identity is not None and process_identity() == identity:
            time.sleep(5)
    for command, output, log in jobs:
        if output.exists() or log.exists():
            raise FileExistsError(f'Refusing existing output: {output}')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as handle:
            subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        if json.loads((output/'results.json').read_text())['interrupted']:
            raise SystemExit('Training was interrupted; sweep stopped before launching another job.')
    print(f'{len(jobs)} sequential jobs completed.', flush=True)
    return jobs


if __name__ == '__main__':
    main()
