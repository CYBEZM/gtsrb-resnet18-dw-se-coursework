"""Record the user-requested repeat extension after the seed-11 evaluation."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from experiment_protocol import FIXED_FIELDS, evaluation_protocol_path
from src.utils import read_json, save_json, sha256


def extend(protocol_path, runs_root, report_root):
    protocol_path, runs_root, report_root = map(Path, (protocol_path, runs_root, report_root))
    previous = read_json(protocol_path)
    if previous['seeds'] == [11, 22, 33] and previous.get('version') == 3:
        for seed in previous['seeds']:
            evaluation_protocol_path(protocol_path, seed)
        print('Extension already recorded; no files changed')
        return
    if previous['seeds'] != [11]:
        raise ValueError('Expected the completed single-seed protocol')
    for source, expected in previous['source_sha256'].items():
        if sha256(source) != expected:
            raise ValueError(f'Frozen source changed: {source}')
    archive = Path('archives/single_seed_11')
    if archive.exists():
        raise FileExistsError('Extension archive already exists; inspect before retrying')
    inherited_files = []
    for variant in 'ABCD':
        run = runs_root / variant / '11'
        complete = read_json(run / 'complete.json')
        assert complete['checkpoint_sha256'] == sha256(run / 'best.ckpt')
        assert len(read_json(run / 'robustness_metrics.json')) == 16
        for path in sorted(run.rglob('*')):
            if path.is_file():
                inherited_files.append(dict(path=path.as_posix(), sha256=sha256(path)))
    old_hash = sha256(protocol_path)
    parent_path = Path('protocols/protocol_v2_single_seed.json')
    if parent_path.exists() and sha256(parent_path) != old_hash:
        raise ValueError('Existing protocol archive differs')
    archive.mkdir(parents=True)
    shutil.copytree(report_root, archive / 'report')
    old_docx = Path('../大作业报告_交通标志识别.docx')
    if old_docx.exists():
        shutil.copy2(old_docx, archive / old_docx.name)
    parent_path.parent.mkdir(exist_ok=True)
    shutil.copy2(protocol_path, parent_path)
    expanded = copy.deepcopy(previous)
    expanded.update(version=3, seeds=[11, 22, 33],
        amended_utc=datetime.now(timezone.utc).isoformat(),
        previous_protocol_path=parent_path.as_posix(), previous_protocol_sha256=old_hash,
        amendment_reason='User requested all four variants repeat with seeds 22 and 33 after seeing seed-11 test results. No architecture, split, training, selection or corruption changes; include all seeds, never select by test accuracy.',
        test_results_already_observed=True,
        inherited_evaluations={'11': dict(path=parent_path.as_posix(), sha256=old_hash)},
        aggregation=dict(seeds=[11, 22, 33], standard_deviation_ddof=1,
                         no_best_seed_selection=True, no_automatic_significance_claims=True))
    assert all(previous[field] == expanded[field] for field in FIXED_FIELDS)
    save_json(archive / 'preserved_run_files.json', inherited_files)
    save_json(protocol_path, expanded)
    save_json('validation/three_seed_extension.json', dict(
        amended_utc=expanded['amended_utc'], reason=expanded['amendment_reason'],
        previous_protocol_sha256=old_hash, active_protocol_sha256=sha256(protocol_path),
        archive=archive.as_posix(), preserved_files=archive.joinpath('preserved_run_files.json').as_posix(),
        new_runs=[dict(variant=v, seed=s, epochs=60) for s in (22, 33) for v in 'ABCD'],
        final_runs=12, final_evaluation_conditions=192,
        unchanged_fields=list(FIXED_FIELDS)))
    print('Recorded extension: 8 new runs; seed-11 files preserved byte for byte')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='protocol_frozen.json')
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--report-root', default='report')
    args = parser.parse_args()
    extend(args.protocol, args.runs_root, args.report_root)
