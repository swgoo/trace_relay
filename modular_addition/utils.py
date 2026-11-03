"""Standalone modular-addition I/O, precision, reproducibility and optional logging."""
import contextlib
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import signal
import threading

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = 'trace_relay_modular_addition'


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
    sources = sorted((ROOT/'modular_addition').glob('*.py'))+core
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(git_commit=commit, git_dirty=dirty,
                source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                torch_version=str(torch.__version__), cuda_version=torch.version.cuda)


class Tracking:
    """JSON logs always work; importing W&B is deferred until explicitly enabled."""
    def __init__(self, directory, protocol, mode='disabled', project=DEFAULT_PROJECT, entity=None,
                 resume_id=None):
        self.directory, self.protocol = Path(directory), protocol
        self.mode, self.project, self.entity = mode, project, entity
        self.resume_id = resume_id
        if resume_id is not None and mode != 'online':
            raise ValueError('W&B run continuation requires online tracking')
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
                                  id=self.resume_id, resume='must' if self.resume_id else None,
                                  allow_val_change=bool(self.resume_id),
                                  settings=wandb.Settings(disable_git=True, console='off'))
            self.run.define_metric('step')
            self.run.define_metric('*', step_metric='step')
            write_json(self.directory/'wandb_run.json', dict(id=self.run.id, url=self.run.url, project=self.project))
            if self.resume_id:
                self.run.summary.update(dict(continuation_start_step=self.protocol['resume']['start_step'],
                                             continuation_target_steps=self.protocol['training']['steps']))
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


def stream_seed(seed, namespace):
    return int.from_bytes(hashlib.sha256(f'modular-addition-v1:{seed}:{namespace}'.encode()).digest()[:8], 'little') % 2**63


class DeferredSignals(contextlib.AbstractContextManager):
    """Save a resumable optimizer state after the current update on cancellation."""
    def __init__(self):
        self.requested, self.name, self.previous = False, None, {}

    def _request(self, number, frame):
        self.requested, self.name = True, signal.Signals(number).name

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                self.previous[number] = signal.signal(number, self._request)
        return self

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
