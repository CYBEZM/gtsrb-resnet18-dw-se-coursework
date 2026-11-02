"""Package verified results using explicit allowlists, without data or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def public_files(root=ROOT):
    files = set()
    patterns = ['*.py', '*.md', 'requirements*.txt', 'LICENSE', '.gitignore',
                '.gitattributes', 'protocol_frozen.json', 'weights_manifest.json', 'examples/*.png',
                'examples/*.json', 'examples/*.md', 'configs/*.yaml',
                'src/*.py', 'tests/*.py', 'protocols/*.json', 'validation/*.json']
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    for path in (root / 'report').rglob('*'):
        if (path.is_file() and (path.suffix in {'.json', '.csv', '.md', '.txt', '.png', '.svg', '.pdf'}
                or (path.parent.name=='external' and path.suffix=='.npz'))
                and not path.name.startswith('smoke_')):
            files.add(path)
    variants='ABCD'+('E' if (root/'report/supplement/audit.json').exists() else '')
    for variant in variants:
        for seed in (11, 22, 33):
            run = root / 'runs' / variant / str(seed)
            for name in ('config.json', 'environment.json', 'data_reference.json',
                         'complexity.json', 'complete.json', 'log.csv',
                         'test_metrics.json', 'robustness_metrics.json'):
                path = run / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                files.add(path)
    return sorted(files)


def check_ready():
    protocol = ROOT / 'protocol_frozen.json'
    for name in ('final_audit.json', 'summary_audit.json'):
        audit = json.loads((ROOT / 'report' / name).read_text(encoding='utf-8'))
        if audit['status'] != 'PASS' or audit['protocol_sha256'] != digest(protocol):
            raise ValueError(f'Current protocol has not passed {name}')
    result = json.loads((ROOT / 'report/final_audit.json').read_text(encoding='utf-8'))
    if result['verified_conditions'] != 192:
        raise ValueError('Expected all 192 evaluated conditions')
    summary = json.loads((ROOT / 'report/tables/summary.json').read_text(encoding='utf-8'))
    if summary['seeds'] != [11, 22, 33] or len(summary['per_seed']) != 12:
        raise ValueError('Expected all twelve runs')
    for source in summary['source_files']:
        if digest(ROOT / source['path']) != source['sha256']:
            raise ValueError(f'Summary input changed: {source["path"]}')
    for name in ('deep_analysis/analysis.json','external/audit.json','supplement/audit.json'):
        path=ROOT/'report'/name
        if not path.exists():
            continue
        audit=json.loads(path.read_text(encoding='utf-8'))
        if audit['status']!='PASS':
            raise ValueError(f'Supplemental audit failed: {name}')
        if 'summary_sha256' in audit and digest(path.parent/'summary.json')!=audit['summary_sha256']:
            raise ValueError(f'Supplemental summary changed: {name}')
        for source in audit.get('source_files',[]):
            if digest(ROOT/source['path'])!=source['sha256']:
                raise ValueError(f'Supplement input changed: {source["path"]}')
    independent=ROOT/'report/deep_analysis/independent_audit.json'
    if independent.exists():
        audit=json.loads(independent.read_text(encoding='utf-8'))
        if audit['status']!='PASS' or audit['analysis_sha256']!=digest(independent.parent/'analysis.json'):
            raise ValueError('Independent per-class audit is stale')


def package(output, include_weights=False):
    check_ready()
    output = Path(output).resolve()
    files = public_files()
    if include_weights:
        variants='ABCD'+('E' if (ROOT/'report/supplement/audit.json').exists() else '')
        for variant in variants:
            for seed in (11, 22, 33):
                run = ROOT / 'runs' / variant / str(seed)
                files.extend([run / 'best.ckpt', run / 'source_snapshot.zip'])
                files.extend(sorted((run / 'predictions').glob('*.npz')))
                if len(list((run / 'predictions').glob('*.npz'))) != 16:
                    raise ValueError(f'Incomplete predictions: {run}')
        files.extend(sorted((ROOT/'report/external').glob('*.npz')))
    files=sorted(set(files))
    manifest = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                     sha256=digest(p)) for p in sorted(files)]
    note = ('This archive contains code, all twelve run metrics, and the measured report.\n'
            'No dataset or credentials are included. See README.md for installation and data download.\n'
            'All mean/SD values use seeds 11,22,33; sample SD uses ddof=1.\n')
    note += ('Best checkpoints, source snapshots and 192 prediction arrays are included.\n'
             'Run audit_results.py and audit_summary.py from the extracted project folder.\n'
             if include_weights else
             'Weights and prediction arrays are in the separate full evidence archive.\n'
             'audit_summary.py works with this archive; audit_results.py also needs that full archive.\n')
    if (ROOT/'report/supplement/audit.json').exists():
        note+='E is a separately frozen post-test exploratory study (3 seeds, 48 conditions).\n'
    if (ROOT/'report/external/audit.json').exists():
        note+='External GTSDB crop classification is supplementary; physical-sign independence is unproven.\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(files):
            archive.write(path, 'traffic_sign_cnn/' + path.relative_to(ROOT).as_posix())
        archive.writestr('traffic_sign_cnn/PACKAGE_MANIFEST.json',
                         json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr('traffic_sign_cnn/PACKAGE_README.txt', note)
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'Bad archive entry: {bad}')
    temporary.replace(output)
    result = dict(path=str(output), bytes=output.stat().st_size, sha256=digest(output),
                  files=len(manifest), includes_weights_predictions=include_weights)
    output.with_suffix('.manifest.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-weights', action='store_true')
    args = parser.parse_args()
    package(args.output, args.include_weights)
