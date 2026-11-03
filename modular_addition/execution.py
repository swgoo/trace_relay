"""Bounded parallel execution for independent modular-addition training runs."""
import json
import signal
import subprocess
import threading
import time


def execute_jobs(jobs, parallel):
    """Keep a bounded queue; signal all active trainers and stop scheduling on any failure."""
    if type(parallel) is not int or parallel < 1:
        raise ValueError('parallel must be a positive integer')
    active, pending, previous_handlers = [], iter(jobs), {}
    stop_requested, failed, exhausted = False, None, False

    def stop_children(number=signal.SIGTERM):
        for entry in active:
            if entry['process'].poll() is None:
                try:
                    entry['process'].send_signal(number)
                except ProcessLookupError:
                    pass

    def relay_signal(number, frame):
        nonlocal stop_requested
        stop_requested = True
        stop_children(number)

    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, relay_signal)
    try:
        while active or not exhausted:
            while not exhausted and not stop_requested and failed is None and len(active) < parallel:
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                if job['output'].exists() or job['log'].exists():
                    raise FileExistsError(f"Refusing existing output/log: {job['output']}")
                job['log'].parent.mkdir(parents=True, exist_ok=True)
                handle = job['log'].open('x')
                try:
                    process = subprocess.Popen(job['command'], stdout=handle, stderr=subprocess.STDOUT)
                except BaseException:
                    handle.close()
                    raise
                active.append(dict(job=job, process=process, handle=handle))
                print(f"Started {job['output'].name} (pid {process.pid})", flush=True)
            for entry in list(active):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['handle'].close()
                active.remove(entry)
                job = entry['job']
                if code and failed is None and not stop_requested:
                    failed = subprocess.CalledProcessError(code, job['command'])
                if not code and failed is None and not stop_requested:
                    report_path = job['output'] / 'results.json'
                    try:
                        report = json.loads(report_path.read_text())
                        if report.get('interrupted') or report.get('stop_reason') != 'step_budget':
                            failed = RuntimeError(f"Run did not finish its fixed budget: {job['output']}")
                    except (OSError, ValueError) as error:
                        failed = RuntimeError(f"Missing/invalid run report {report_path}: {error}")
                print(f"Finished {job['output'].name} (exit {code})", flush=True)
            if stop_requested or failed is not None:
                exhausted = True
                stop_children()
            if active:
                time.sleep(.1)
        if failed is not None:
            raise failed
        if stop_requested:
            print('Sweep interrupted; active runs saved their stop checkpoints; no further runs started.', flush=True)
        return not stop_requested
    finally:
        # Also protect against launch failures and unexpected parent exceptions.
        stop_children()
        for entry in active:
            entry['process'].wait()
            entry['handle'].close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
