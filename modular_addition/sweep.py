"""Preview or execute independent modular-addition seeds in parallel."""
import argparse
from pathlib import Path
import shlex
import sys

from .execution import execute_jobs
from .train import architecture, build_parser, validate_args, scheduler_config
from .utils import DEFAULT_PROJECT


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/modular-addition-phase1'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--parallel',type=int,default=3)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    parser.add_argument('--wandb-project',default=DEFAULT_PROJECT)
    parser.add_argument('--wandb-entity',default=None)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--matched-initialization',action='store_true',
                        help='Match shared baseline weights and training RNG for delta comparisons')
    parser.add_argument('trainer_args',nargs=argparse.REMAINDER)
    a=parser.parse_args(argv)
    if a.parallel<1 or len(set(a.seeds))!=len(a.seeds) or (a.execute and a.dry_run):
        parser.error('Positive parallelism, unique seeds, and one execution mode required')
    extra=a.trainer_args[1:] if a.trainer_args[:1]==['--'] else a.trainer_args
    reserved={'--output','--resume','--seed','--device','--wandb-mode','--wandb-project','--wandb-entity','--wandb-resume-id'}
    if any(value.split('=')[0] in reserved for value in extra):
        parser.error('Set seed/output/device/tracking via sweep options; sweep cannot resume')
    jobs=[]
    for seed in a.seeds:
        output=a.output_root/f'modular-addition-seed{seed}'
        log=output.with_suffix('.log')
        if output.exists() or log.exists():
            parser.error('Refusing existing output/log: '+str(output))
        options=['--output',str(output),'--seed',str(seed),'--device',a.device,
            '--wandb-mode',a.wandb_mode,'--wandb-project',a.wandb_project,'--final-evaluation']
        if a.wandb_entity:
            options+=['--wandb-entity',a.wandb_entity]
        options+=extra
        args=build_parser().parse_args(options)
        try:
            validate_args(args); architecture(args); scheduler_config(args)
        except ValueError as error:
            parser.error(str(error))
        module = 'delta_matched_train' if a.matched_initialization else 'train'
        jobs.append(dict(output=output,log=log,command=[a.python,'-u','-m',f'{__package__}.{module}',*options]))
    for job in jobs:
        print(shlex.join(job['command']),flush=True)
    if a.execute:
        execute_jobs(jobs,a.parallel)
    else:
        print(f'{len(jobs)} commands; parallel={a.parallel}; no jobs launched.',flush=True)
    return jobs


if __name__=='__main__':
    main()
