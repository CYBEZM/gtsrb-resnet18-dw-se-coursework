"""Freeze the final evaluation protocol BEFORE inspecting test predictions."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.data import PROTOCOL, build_cache
from src.utils import load_config, read_json, save_json, sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--output', type=Path, default=Path('protocol_frozen.json'))
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    build_cache(args.data_root)
    candidate = dict(version=2, frozen_utc=datetime.now(timezone.utc).isoformat(),
              manifest_sha256=sha256(args.data_root / 'cache48/manifest.json'),
              configs={v: load_config(f'configs/{v}.yaml') for v in 'ABCD'},
              seeds=args.seeds or (read_json(args.output)['seeds'] if args.output.exists() else [11, 22, 33]),
              corruption_protocol=PROTOCOL,
              source_sha256={p: sha256(p) for p in ['src/models.py', 'src/data.py', 'src/utils.py', 'train.py', 'eval.py']},
              selection='max validation macro-F1, then min loss, then earliest epoch',
              evaluation='Clean and 15 prespecified corruptions, all planned models/seeds; no test tuning')
    if args.output.exists():
        frozen = read_json(args.output)
        # JSON round-trip normalizes tuple-valued preprocessing constants.
        import json
        candidate = json.loads(json.dumps(candidate))
        for key in ('manifest_sha256', 'configs', 'seeds', 'corruption_protocol',
                    'source_sha256', 'selection', 'evaluation'):
            if candidate[key] != frozen[key]:
                raise ValueError(f'Existing frozen protocol differs in {key}; refusing overwrite')
        print('Existing frozen protocol validated:', args.output)
    else:
        save_json(args.output, candidate)
    print(args.output)
