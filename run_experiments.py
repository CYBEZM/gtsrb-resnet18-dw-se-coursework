"""Sequential, resumable experiment queue for a single GPU."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils import load_config, read_json, save_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--protocol', type=Path, default=Path('protocol_frozen.json'))
    args = parser.parse_args()
    if not args.protocol.exists():
        raise RuntimeError('Run and inspect pilot/validation examples, then freeze_protocol.py first')
    frozen = read_json(args.protocol)
    for source, expected in frozen['source_sha256'].items():
        if sha256(source) != expected:
            raise ValueError(f'Source changed after freeze: {source}')
    for variant in 'ABCD':
        if load_config(f'configs/{variant}.yaml') != frozen['configs'][variant]:
            raise ValueError(f'Configuration changed after freeze: {variant}')
    args.runs_root.mkdir(parents=True, exist_ok=True)
    status_path = args.runs_root / 'queue_status.json'
    def status(stage, **kwargs):
        save_json(status_path, dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **kwargs))
    def command(cmd):
        print('RUN', ' '.join(map(str, cmd)), flush=True)
        subprocess.run([sys.executable, '-u', *map(str, cmd)], check=True)
    try:
        # Each process releases all GPU allocations before the next run.
        for seed in frozen['seeds']:
            for variant in 'ABDC':
                run = args.runs_root / variant / str(seed)
                status('training', variant=variant, seed=seed)
                command(['train.py', '--config', f'configs/{variant}.yaml', '--seed', seed,
                         '--data-root', args.data_root, '--output', run, '--resume'])
        for seed in frozen['seeds']:
            for variant in 'ABCD':
                status('evaluation', variant=variant, seed=seed)
                command(['eval.py', '--run', args.runs_root / variant / str(seed),
                         '--data-root', args.data_root, '--protocol', args.protocol])
        for variant in 'ABCD':
            status('profiling', variant=variant)
            command(['profile_model.py', '--variant', variant, '--checkpoint', args.runs_root / variant / str(frozen['seeds'][0]) / 'best.ckpt',
                     '--output', f'report/profiles/{variant}.json'])
        status('tables')
        command(['make_tables.py', '--runs-root', args.runs_root, '--protocol', args.protocol])
        status('report_draft')
        command(['build_report.py', '--runs-root', args.runs_root, '--protocol', args.protocol])
        status('complete')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, detail=str(error))
        raise


if __name__ == '__main__':
    main()
