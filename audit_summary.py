"""Independently check aggregate means and sample SD with Python statistics."""
import argparse
import math
from pathlib import Path
import statistics

from src.utils import read_json, save_json, sha256


def audit(summary_path, protocol_path):
    summary = read_json(summary_path)
    protocol = read_json(protocol_path)
    assert summary['protocol_sha256'] == sha256(protocol_path)
    assert summary['seeds'] == protocol['seeds']
    seeds = protocol['seeds']
    assert len(summary['per_seed']) == 4 * len(seeds)
    def check(values, mean, std):
        assert math.isclose(statistics.mean(values), mean, rel_tol=0, abs_tol=1e-12)
        expected_std = statistics.stdev(values) if len(values) > 1 else None
        assert (std is None and expected_std is None) or (
            std is not None and expected_std is not None and
            math.isclose(std, expected_std, rel_tol=0, abs_tol=1e-12))
    clean = {}
    for variant in 'ABCD':
        rows = [r for r in summary['per_seed'] if r['variant'] == variant]
        assert [r['seed'] for r in rows] == seeds
        model = next(r for r in summary['models'] if r['variant'] == variant)
        clean[variant] = {r['seed']: read_json(r['source']) for r in rows}
        for field in ('accuracy', 'macro_f1'):
            check([clean[variant][s][field] for s in seeds], model[field + '_mean'], model[field + '_std'])
        for row in (r for r in summary['robustness'] if r['variant'] == variant):
            originals = [read_json(Path(r['source']).parent / 'robustness_metrics.json')[f'{row["kind"]}_{row["severity"]}'] for r in rows]
            for field in ('accuracy', 'macro_f1'):
                check([r[field] for r in originals], row[field + '_mean'], row[field + '_std'])
    for row in summary['contrasts']:
        if row['contrast'] == 'SE interaction':
            values = [100 * ((clean['D'][s]['accuracy'] - clean['B'][s]['accuracy']) -
                            (clean['C'][s]['accuracy'] - clean['A'][s]['accuracy'])) for s in seeds]
        else:
            plus, minus = {'DW effect': ('B', 'A'), 'SE standard': ('C', 'A'),
                           'SE depthwise': ('D', 'B')}[row['contrast']]
            values = [100 * (clean[plus][s]['accuracy'] - clean[minus][s]['accuracy']) for s in seeds]
        check(values, row['mean_accuracy_difference_pp'], row['std_accuracy_difference_pp'])
    return dict(status='PASS', seeds=seeds, standard_deviation_ddof=1,
                protocol_sha256=sha256(protocol_path), summary_sha256=sha256(summary_path),
                checks=['clean and corruption Accuracy/Macro-F1 means and sample SD',
                        'paired contrasts and interaction means and sample SD', 'all seeds included'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default='report/tables/summary.json')
    parser.add_argument('--protocol', default='protocol_frozen.json')
    parser.add_argument('--output', default='report/summary_audit.json')
    args = parser.parse_args()
    save_json(args.output, audit(args.summary, args.protocol))
    print('PASS: aggregate means and sample standard deviations independently verified')
