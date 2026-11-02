"""Sequential, resumable experiment queue for a single GPU."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils import load_config, read_json, save_json, sha256
from experiment_protocol import evaluation_protocol_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--protocol', type=Path, default=Path('protocol_frozen.json'))
    parser.add_argument('--plan-only', action='store_true')
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
    for seed in frozen['seeds']:
        evaluation_protocol_path(args.protocol, seed)
    if args.plan_only:
        pending = [(variant, seed) for seed in frozen['seeds'] for variant in 'ABDC'
                   if not (args.runs_root / variant / str(seed) / 'complete.json').exists()]
        print('Pending training runs:', pending, flush=True)
        print('Planned evaluation conditions:', 16 * 4 * len(frozen['seeds']), flush=True)
        return
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
                         '--data-root', args.data_root,
                         '--protocol', evaluation_protocol_path(args.protocol, seed)])
        for variant in 'ABCD':
            profile_path = Path(f'report/profiles/{variant}.json')
            if profile_path.exists():
                profile = read_json(profile_path)
                if profile['checkpoint_sha256'] != sha256(args.runs_root / variant / str(frozen['seeds'][0]) / 'best.ckpt'):
                    raise ValueError('Existing hardware profile checkpoint differs')
                print('Reuse validated GPU profile:', profile_path, flush=True)
                continue
            status('profiling', variant=variant)
            command(['profile_model.py', '--variant', variant, '--checkpoint', args.runs_root / variant / str(frozen['seeds'][0]) / 'best.ckpt',
                     '--output', f'report/profiles/{variant}.json'])
        status('tables')
        command(['make_tables.py', '--runs-root', args.runs_root, '--protocol', args.protocol])
        status('summary_audit')
        command(['audit_summary.py', '--protocol', args.protocol])
        status('audit')
        command(['audit_results.py', '--runs-root', args.runs_root, '--protocol', args.protocol])
        status('examples')
        command(['make_examples.py', '--runs-root', args.runs_root, '--data-root', args.data_root])
        status('report_draft')
        command(['build_report.py', '--runs-root', args.runs_root, '--protocol', args.protocol])
        status('complete')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, detail=str(error))
        raise


if __name__ == '__main__':
    main()
