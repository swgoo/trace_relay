"""Validated sequential Most-Freq sweeps. Commands-only unless --execute is explicit."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .train import build_parser,validate_args,architecture


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--centers',type=int,nargs='+',choices=[64,32,16],default=[64,32,16])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--models',nargs='+',choices=['trace_relay','no_carry','swa'],default=['trace_relay'])
    p.add_argument('--output-root',type=Path,default=Path('outputs/most-freq-sweep'))
    p.add_argument('--python',default=sys.executable);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--execute',action='store_true')
    p.add_argument('trainer_args',nargs=argparse.REMAINDER,help='Shared trainer options after --')
    a=p.parse_args(argv)
    if any(len(values)!=len(set(values)) for values in (a.centers,a.seeds,a.models)) or any(not 0<=seed<2**63 for seed in a.seeds):
        p.error('Sweep axes must be unique; seeds must be in 0..2**63-1')
    extra=a.trainer_args[1:] if a.trainer_args[:1]==['--'] else a.trainer_args
    reserved={'--output','--resume','--model','--center-width','--trace-widths','--seed'}
    if any(arg.split('=')[0] in reserved for arg in extra):p.error('Use sweep axes; output/resume/trace-width overrides are not allowed')
    jobs=[]
    for seed in a.seeds:
        for center in a.centers:
            for family in a.models:
                output=a.output_root/f'most-freq-{family}-center{center}-seed{seed}'
                log=output.with_suffix('.log')
                if output.exists() or log.exists():p.error(f'Refusing existing output/log: {output}')
                args=['--output',str(output),'--device',a.device,'--center-width',str(center),
                      '--seed',str(seed),'--model',family,'--final-evaluation','--wandb-mode','online',*extra]
                options=build_parser().parse_args(args);validate_args(options);architecture(options)
                command=[a.python,'-u','-m','most_freq.train',*args]
                jobs.append(dict(command=command,output=output,log=log))
    for job in jobs:print(shlex.join(job['command']),flush=True)
    if not a.execute:
        print(f'{len(jobs)} commands; no jobs launched.',flush=True)
        return jobs
    for job in jobs:
        if job['output'].exists() or job['log'].exists():raise FileExistsError(f"Refusing existing output/log: {job['output']}")
        job['log'].parent.mkdir(parents=True,exist_ok=True)
        with job['log'].open('x') as handle:
            subprocess.run(job['command'],stdout=handle,stderr=subprocess.STDOUT,check=True)
        report=json.loads((job['output']/'results.json').read_text())
        if report.get('interrupted') or report.get('stop_reason') not in ('step_budget','validation_threshold'):
            raise RuntimeError('Run interrupted/failed; stopping before the next job')
    return jobs


if __name__=='__main__':main()
