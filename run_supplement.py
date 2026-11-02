"""Separate exploratory E runs; never change the original A-D protocol/results."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from src.utils import load_config, read_json, save_json, sha256

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'protocols/supplement_E.json'


def freeze():
    parent = ROOT / 'protocol_frozen.json'
    original = read_json(parent)
    candidate = {key: value for key, value in original.items()
                 if key in ('manifest_sha256', 'corruption_protocol', 'source_sha256',
                            'selection', 'evaluation', 'seeds')}
    candidate.update(version=4, configs={'E': load_config(ROOT / 'configs/E.yaml')},
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=sha256(parent), test_results_already_observed=True,
        scope='Post-test exploratory E: DW only in stage3/4, no attention; all seeds retained',
        question='Does keeping standard convolutions in stage1/2 change accuracy-complexity tradeoff?',
        comparisons=['E-A', 'E-B'], no_test_tuning=True)
    if PLAN.exists():
        saved = read_json(PLAN)
        for key in candidate:
            if key != 'frozen_utc' and saved[key] != candidate[key]:
                raise ValueError(f'Supplement protocol differs in {key}')
    else:
        save_json(PLAN, candidate)
    return read_json(PLAN)


def main(data_root):
    plan = freeze()
    for source, expected in plan['source_sha256'].items():
        if sha256(ROOT / source) != expected:
            raise ValueError(f'Frozen source changed: {source}')
    status_path = ROOT / 'runs/supplement_status.json'
    def stage(name, **kwargs):
        save_json(status_path, dict(stage=name, utc=datetime.now(timezone.utc).isoformat(), **kwargs))
    def command(*args):
        print('RUN', ' '.join(map(str, args)), flush=True)
        subprocess.run([sys.executable, '-u', *map(str, args)], cwd=ROOT, check=True)
    try:
        for seed in plan['seeds']:
            stage('training', variant='E', seed=seed)
            command('train.py', '--config', 'configs/E.yaml', '--seed', seed,
                    '--data-root', data_root, '--output', f'runs/E/{seed}', '--resume')
        for seed in plan['seeds']:
            stage('evaluation', variant='E', seed=seed)
            command('eval.py', '--run', f'runs/E/{seed}', '--data-root', data_root,
                    '--protocol', PLAN)
        stage('profiling', variant='E')
        command('-c', 'from profile_model import profile; from src.utils import save_json; '
                'save_json("report/supplement/profile_E.json",profile("E","runs/E/11/best.ckpt"))')
        stage('analysis')
        command('analyze_results.py')
        stage('finalizing')
        command('finalize_improvements.py','--data-root',data_root)
        stage('complete')
    except BaseException as error:
        stage('failed', detail=str(error), error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    if args.freeze_only:
        freeze()
        print(PLAN)
    else:
        main(args.data_root)
