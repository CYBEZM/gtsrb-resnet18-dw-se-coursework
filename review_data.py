"""One measured data view for the integrated report, figures and repository overview.

No training/evaluation code or historical results are rewritten here.
"""
import csv
from pathlib import Path
from statistics import mean, stdev

from src.utils import read_json, save_json, sha256

ROOT = Path(__file__).resolve().parent
VARIANTS = 'ABCDE'
SEEDS = (11, 22, 33)
NAMES = dict(A='ResNet-18', B='全阶段 DW', C='ResNet-18 + SE',
             D='全阶段 DW + SE', E='stage3/4 DW')
KINDS = dict(brightness='亮度降低', motion='运动模糊', noise='高斯噪声',
             jpeg='JPEG压缩', resolution='降低分辨率')


def csv_rows(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def pm(avg, sd, scale=100, digits=2):
    return f'{avg*scale:.{digits}f} ± {sd*scale:.{digits}f}'


def collect():
    original = read_json(ROOT / 'report/tables/summary.json')
    extra = read_json(ROOT / 'report/supplement/summary.json')
    models, runs, profiles, complexity, sources, robustness = {}, {}, {}, {}, set(), []
    for variant in VARIANTS:
        samples = []
        for seed in SEEDS:
            run = ROOT / f'runs/{variant}/{seed}'
            metric = read_json(run / 'test_metrics.json')
            complete = read_json(run / 'complete.json')
            logs = csv_rows(f'runs/{variant}/{seed}/log.csv')
            assert len(logs) == 60 and metric['samples'] == 12630
            runs[variant, seed] = metric
            samples.append(metric)
            for filename in ('test_metrics.json', 'complete.json', 'complexity.json',
                             'log.csv', 'robustness_metrics.json'):
                sources.add(f'runs/{variant}/{seed}/{filename}')
        model = dict(variant=variant)
        for key in ('accuracy', 'macro_f1'):
            model[key+'_mean'] = mean(r[key] for r in samples)
            model[key+'_sd'] = stdev(r[key] for r in samples)
        complexity[variant] = read_json(ROOT/f'runs/{variant}/11/complexity.json')
        model.update(parameters=complexity[variant]['parameters'],
                     conv_linear_macs=complexity[variant]['conv_linear_macs'])
        profile_path = f'report/supplement/contemporary_profiles/{variant}.json'
        sources.add(profile_path)
        profiles[variant] = read_json(ROOT/profile_path)
        assert profiles[variant]['checkpoint_sha256'] == runs[variant, 11]['checkpoint_sha256']
        models[variant] = model
    for row in original['models']:
        for key in ('accuracy_mean', 'macro_f1_mean'):
            assert abs(row[key]-models[row['variant']][key]) < 1e-12
    for row in original['robustness']:
        robustness.append(dict(variant=row['variant'], kind=row['kind'], severity=row['severity'],
            accuracy_mean=row['accuracy_mean'], accuracy_sd=row['accuracy_std']))
    for row in extra['robustness']:
        kind, severity = row['condition'].rsplit('_', 1)
        if kind == 'clean':
            assert abs(models['E']['accuracy_mean']-row['accuracy_mean']) < 1e-12
            continue
        robustness.append(dict(variant='E', kind=kind, severity=int(severity),
            accuracy_mean=row['accuracy_mean'], accuracy_sd=row['accuracy_sample_sd']))
    assert len(robustness) == 75
    sources.update(['report/tables/summary.json', 'report/supplement/summary.json',
                    'report/external/summary.json', 'report/data/dataset_summary.json',
                    'report/deep_analysis/paired_errors.csv',
                    'report/deep_analysis/paired_class_summary.csv',
                    'report/deep_analysis/largest_validation_drops.csv'])
    contrasts = []
    for label, positive, negative in [('B-A','B','A'),('C-A','C','A'),('D-B','D','B'),
                                      ('E-A','E','A'),('E-B','E','B')]:
        values = [100*(runs[positive,s]['accuracy']-runs[negative,s]['accuracy']) for s in SEEDS]
        contrasts.append(dict(contrast=label, mean_pp=mean(values), sd_pp=stdev(values),
                              seed_differences_pp=values))
    return dict(models=models, runs=runs, profiles=profiles, complexity=complexity,
                robustness=robustness, contrasts=contrasts, source_paths=sorted(sources))


def write_summary(data):
    output = dict(models=list(data['models'].values()), profiles=data['profiles'],
        robustness=data['robustness'], contrasts=data['contrasts'], seeds=list(SEEDS),
        training_runs=15, epochs=900, gtsrb_conditions=240,
        timing_source='report/supplement/contemporary_profiles/{variant}.json',
        sources=[dict(path=p, sha256=sha256(ROOT/p)) for p in data['source_paths']])
    save_json(ROOT/'report/integrated/summary.json', output)
    return output
