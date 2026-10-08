"""Most-Freq-only I/O, precision, reproducibility and optional experiment logging."""
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def choose_backend(device, precision, requested):
    if device not in ('cpu', 'cuda') or precision not in ('auto', 'fp32', 'bf16'):
        raise ValueError('Unsupported device/precision')
    if precision == 'bf16' and device != 'cuda':
        raise ValueError('BF16 execution requires CUDA')
    if requested not in ('auto', 'eager', 'flash_attention_2'):
        raise ValueError('Unsupported attention backend')
    backend = requested
    if backend == 'auto':
        backend = 'flash_attention_2' if device == 'cuda' and precision != 'fp32' and importlib.util.find_spec('flash_attn') else 'eager'
    if backend == 'flash_attention_2':
        if device != 'cuda' or precision == 'fp32':
            raise ValueError('FlashAttention-2 requires CUDA and BF16/auto precision')
        if importlib.util.find_spec('flash_attn') is None:
            raise ValueError('FlashAttention-2 package is unavailable')
    return backend


def effective_precision(device, precision):
    return 'bf16' if device == 'cuda' and precision != 'fp32' else 'fp32'


def autocast(device, precision):
    if precision == 'bf16' and device != 'cuda':
        raise ValueError('BF16 execution requires CUDA')
    return torch.autocast('cuda', dtype=torch.bfloat16) if effective_precision(device, precision) == 'bf16' else contextlib.nullcontext()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    result = dict(python=random.getstate(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def provenance():
    core = [ROOT/'configuration_trace_relay.py', ROOT/'modeling_trace_relay.py']
    sources = sorted((ROOT/'most_freq').glob('*.py'))+core
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    return dict(git_commit=commit, git_dirty=bool(status),
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


class Tracking:
    """JSON logs always work; importing W&B is deferred until explicitly enabled."""
    def __init__(self, directory, protocol, mode='disabled', project='trace relay most freq', entity=None):
        self.directory, self.protocol = Path(directory), protocol
        self.mode, self.project, self.entity = mode, project, entity
        self.run = None

    def __enter__(self):
        if self.mode != 'disabled':
            try:
                import wandb
            except ImportError:
                if self.mode == 'online':
                    raise RuntimeError('Online tracking requires the optional wandb package') from None
                print('W&B unavailable; offline metrics remain in local JSON files.', flush=True)
                return self
            self.run = wandb.init(project=self.project, entity=self.entity, name=self.directory.name,
                                  dir=str(self.directory), mode=self.mode, config=self.protocol,
                                  settings=wandb.Settings(disable_git=True, console='off'))
            self.run.define_metric('step')
            self.run.define_metric('*', step_metric='step')
            write_json(self.directory/'wandb_run.json', dict(id=self.run.id, url=self.run.url, project=self.project))
        return self

    def log(self, row):
        if self.run is None:
            return
        self.run.log({'step':row['step'], **{f"{row['phase']}/{key}":value for key,value in row.items()
            if key not in ('step', 'phase') and isinstance(value, (int, float, bool))},
            **{f"{row['phase']}/{key}":value for key,value in row.get('metrics', {}).items()}})

    def summary(self, result):
        if self.run is not None:
            self.run.summary.update(result)

    def __exit__(self, exc_type, exc, tb):
        if self.run is not None:
            self.run.finish(exit_code=int(exc_type is not None))
