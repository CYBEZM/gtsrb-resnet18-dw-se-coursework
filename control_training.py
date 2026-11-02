"""Pause/resume only this project's queue; saved checkpoints remain intact."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import psutil

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'runs'


def owned(process, script):
    try:
        return (Path(process.cwd()).resolve() == ROOT and
                script in [Path(a).name for a in process.cmdline()])
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False


def queue_processes():
    return [p for p in psutil.process_iter() if owned(p, 'run_experiments.py')]


def control(action):
    RUNS.mkdir(exist_ok=True)
    queues = queue_processes()
    trainers = [p for p in psutil.process_iter() if owned(p, 'train.py')]
    if action == 'status':
        print(json.dumps({'queue_pids': [p.pid for p in queues],
                          'training_pids': [p.pid for p in trainers]}, indent=2))
        return
    if action == 'pause':
        # Stop the scheduler first so it cannot launch another training process.
        targets = []
        for queue in queues:
            for child in queue.children(recursive=True):
                if any(owned(child, name) for name in ('train.py', 'eval.py', 'profile_model.py',
                                                       'make_tables.py', 'build_report.py',
                                                       'audit_results.py', 'audit_summary.py', 'make_examples.py')):
                    targets.append(child)
            queue.terminate()
        psutil.wait_procs(queues, timeout=10)
        for child in {p.pid: p for p in targets + trainers}.values():
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(targets + trainers, timeout=10)
        if alive:
            raise RuntimeError('Some task processes did not exit; inspect before resuming')
        value = {'stage': 'paused', 'utc': datetime.now(timezone.utc).isoformat(),
                 'note': 'Resume from last atomically saved epoch; incomplete epoch will repeat.'}
        temp = RUNS / 'queue_status.json.tmp'
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temp, RUNS / 'queue_status.json')
        print('Paused. Checkpoints preserved; GPU memory will be released.')
        return
    if queues or trainers:
        raise RuntimeError('A queue or trainer is already running; refusing duplicate launch')
    # Recover the actual data location from existing run metadata.
    references = sorted(RUNS.glob('*/*/data_reference.json'))
    if not references:
        raise RuntimeError('No previous run metadata; use documented first-run command')
    data_root = json.loads(references[0].read_text(encoding='utf-8'))['data_root']
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP |
             subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
    with (RUNS / 'queue.stdout.log').open('ab') as stdout, (RUNS / 'queue.stderr.log').open('ab') as stderr:
        child = subprocess.Popen([sys.executable, '-u', 'run_experiments.py', '--data-root', data_root],
                                  cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                  creationflags=flags, start_new_session=os.name != 'nt')
    (RUNS / 'queue.pid').write_text(str(child.pid), encoding='ascii')
    print(f'Resumed queue PID={child.pid}. Completed runs are skipped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'pause', 'resume'])
    control(parser.parse_args().action)
