"""Generate all quantitative tables/figures directly from recorded outputs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.utils import read_json, save_json, sha256
from experiment_protocol import evaluation_protocol_path

COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    fig.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def make_confusion(runs_root, output, seed):
    """Render fixed-seed recorded confusion matrices with a white zero level."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, variant in zip(axes, ['A', 'D']):
        cm = np.array(read_json(Path(runs_root) / variant / str(seed) / 'test_metrics.json')['confusion_matrix'])
        norm = cm / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap='Greys')
        ax.set(title=f'{variant}, seed {seed}', xlabel='Predicted class ID', ylabel='True class ID')
    fig.colorbar(im, ax=list(axes), shrink=0.75, label='Row-normalized frequency')
    save_figure(fig, Path(output) / 'figures/confusion')


def sample_std(values):
    """One run has no estimable across-seed SD: serialize null, never zero/NaN."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def make(runs_root, output, protocol_path='protocol_frozen.json'):
    runs_root, output = Path(runs_root), Path(output)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    seeds = protocol['seeds']
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Protocol must specify unique training seeds')
    n_seeds = len(seeds)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    runs, per_seed, summary, robust_rows, sources = {}, [], [], [], []
    for v in 'ABCD':
        runs[v] = []
        for seed in seeds:
            path = runs_root / v / str(seed)
            evaluation_protocol = evaluation_protocol_path(protocol_path, seed)
            clean = read_json(path / 'test_metrics.json')
            robust = read_json(path / 'robustness_metrics.json')
            if len(robust) != 16:
                raise ValueError(f'Incomplete conditions: {path}')
            complete = read_json(path / 'complete.json')
            if clean['checkpoint_sha256'] != complete['checkpoint_sha256']:
                raise ValueError('Checkpoint provenance mismatch')
            for metrics in [clean, *robust.values()]:
                if (metrics['protocol_sha256'] != sha256(evaluation_protocol) or
                        metrics['checkpoint_sha256'] != complete['checkpoint_sha256'] or
                        metrics['seed'] != seed or metrics['variant'] != v):
                    raise ValueError(f'Evaluation provenance mismatch: {path}')
            with (path / 'log.csv').open(newline='', encoding='utf-8') as f:
                history = [{k: float(val) for k, val in r.items()} for r in csv.DictReader(f)]
            run = dict(path=path, clean=clean, robust=robust, complete=complete, history=history)
            runs[v].append(run)
            per_seed.append(dict(variant=v, seed=seed, accuracy=clean['accuracy'], macro_f1=clean['macro_f1'],
                                 correct=clean['correct'], samples=clean['samples'], best_epoch=complete['best_epoch'] + 1,
                                 source=(path / 'test_metrics.json').as_posix()))
            for filename in ('test_metrics.json', 'robustness_metrics.json', 'complexity.json', 'complete.json', 'log.csv'):
                p = path / filename
                sources.append(dict(path=p.as_posix(), sha256=sha256(p)))
        acc = np.array([r['clean']['accuracy'] for r in runs[v]])
        f1 = np.array([r['clean']['macro_f1'] for r in runs[v]])
        cmp = read_json(runs_root / v / str(seeds[0]) / 'complexity.json')
        profile = read_json(output / 'profiles' / f'{v}.json')
        sources.append(dict(path=(output / 'profiles' / f'{v}.json').as_posix(),
                            sha256=sha256(output / 'profiles' / f'{v}.json')))
        summary.append(dict(variant=v, n_seeds=n_seeds, accuracy_mean=float(acc.mean()), accuracy_std=sample_std(acc),
                            macro_f1_mean=float(f1.mean()), macro_f1_std=sample_std(f1),
                            parameters=cmp['parameters'], conv_linear_macs=cmp['conv_linear_macs'],
                            latency_median_ms=profile['latency_median_ms'], latency_p95_ms=profile['latency_p95_ms'],
                            throughput_images_s=profile['throughput_images_s'], weight_file_bytes=profile['weight_file_bytes']))
        for kind in ('brightness', 'motion', 'noise', 'jpeg', 'resolution'):
            for level in (1, 2, 3):
                vals = [r['robust'][f'{kind}_{level}'] for r in runs[v]]
                accuracies = np.array([r['accuracy'] for r in vals])
                scores = np.array([r['macro_f1'] for r in vals])
                robust_rows.append(dict(variant=v, kind=kind, severity=level, value=vals[0]['value'],
                    accuracy_mean=float(accuracies.mean()), accuracy_std=sample_std(accuracies),
                    macro_f1_mean=float(scores.mean()), macro_f1_std=sample_std(scores),
                    accuracy_drop_pp=float(100 * (acc - accuracies).mean())))
    write_csv(tables / 'per_seed.csv', per_seed)
    write_csv(tables / 'summary.csv', summary)
    write_csv(tables / 'robustness.csv', robust_rows)
    contrasts = []
    for name, plus, minus in [('DW effect', 'B', 'A'), ('SE standard', 'C', 'A'), ('SE depthwise', 'D', 'B')]:
        differences = np.array([100 * (p['clean']['accuracy'] - m['clean']['accuracy']) for p, m in zip(runs[plus], runs[minus])])
        contrasts.append(dict(contrast=name, mean_accuracy_difference_pp=float(differences.mean()),
                              std_accuracy_difference_pp=sample_std(differences),
                              **{f'seed{seed}_pp': float(value) for seed, value in zip(seeds, differences)}))
    interaction = [100 * ((runs['D'][i]['clean']['accuracy'] - runs['B'][i]['clean']['accuracy']) -
                         (runs['C'][i]['clean']['accuracy'] - runs['A'][i]['clean']['accuracy'])) for i in range(n_seeds)]
    contrasts.append(dict(contrast='SE interaction', mean_accuracy_difference_pp=float(np.mean(interaction)),
                          std_accuracy_difference_pp=sample_std(interaction),
                          **{f'seed{seed}_pp': float(value) for seed, value in zip(seeds, interaction)}))
    write_csv(tables / 'contrasts.csv', contrasts)
    sources.append(dict(path=Path(protocol_path).as_posix(), sha256=sha256(protocol_path)))
    for reference in protocol.get('inherited_evaluations', {}).values():
        sources.append(dict(path=reference['path'], sha256=reference['sha256']))
    save_json(tables / 'summary.json', dict(models=summary, per_seed=per_seed, robustness=robust_rows,
              seeds=seeds, protocol_sha256=sha256(protocol_path),
              contrasts=contrasts, source_files=sources,
              note='Raw fractions retained; single-run SD is null, no error bars; multiple-run SD uses ddof=1'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    for v, color in zip('ABCD', COLORS):
        histories = runs[v]
        for ax, field, label in zip(axes, ['train_loss', 'val_macro_f1'], ['Training loss', 'Validation Macro-F1']):
            curves = np.array([[r[field] for r in run['history']] for run in histories])
            if field == 'val_macro_f1':
                curves *= 100
            mean = curves.mean(0)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, label=v, color=color)
            if n_seeds > 1:
                std = curves.std(0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.14)
            ax.set(xlabel='Epoch', ylabel=label)
            ax.grid(alpha=0.2)
    axes[0].legend(title='Variant')
    save_figure(fig, figures / 'learning_curves')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
    for ax, kind in zip(axes.flat, ('brightness', 'motion', 'noise', 'jpeg', 'resolution')):
        for v, color in zip('ABCD', COLORS):
            base = next(r for r in summary if r['variant'] == v)
            rows = [r for r in robust_rows if r['variant'] == v and r['kind'] == kind]
            mean = np.array([base['accuracy_mean']] + [r['accuracy_mean'] for r in rows]) * 100
            ax.plot(range(4), mean, marker='o', label=v, color=color)
            if n_seeds > 1:
                std = np.array([base['accuracy_std']] + [r['accuracy_std'] for r in rows]) * 100
                ax.fill_between(range(4), mean - std, mean + std, color=color, alpha=0.14)
        ax.set(title=kind, xlabel='Severity (0 = clean)', ylabel='Accuracy (%)', xticks=[0, 1, 2, 3])
        ax.grid(alpha=0.2)
    axes.flat[0].legend(ncol=4)
    axes.flat[5].axis('off')
    seed_note = (f'Single training seed {seeds[0]}\nNo across-seed SD estimated' if n_seeds == 1 else
                 f'{n_seeds} training seeds\nMean and sample SD')
    axes.flat[5].text(0.05, 0.8, seed_note + '\nFixed track-grouped split\nShared corrupted inputs\nSeverity scales differ by corruption', va='top', linespacing=1.8)
    save_figure(fig, figures / 'robustness')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for r, color in zip(summary, COLORS):
        for ax, field, xlabel, divisor in [(axes[0], 'parameters', 'Parameters (million)', 1e6),
                                            (axes[1], 'conv_linear_macs', 'Conv/Linear MACs (million)', 1e6)]:
            x, y = r[field] / divisor, r['accuracy_mean'] * 100
            if r['accuracy_std'] is None:
                ax.plot(x, y, 'o', color=color)
            else:
                ax.errorbar(x, y, yerr=r['accuracy_std'] * 100, fmt='o', capsize=4, color=color)
            ax.annotate(r['variant'], (x, y), xytext=(6, 5), textcoords='offset points')
            ax.set(xlabel=xlabel, ylabel='Clean accuracy (%)')
            ax.grid(alpha=0.2)
    save_figure(fig, figures / 'tradeoffs')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    for ax, field, ylabel in zip(axes, ['latency_median_ms', 'latency_p95_ms', 'throughput_images_s'],
                                ['Batch-1 median latency (ms)', 'Batch-1 P95 latency (ms)', 'Batch-32 throughput (images/s)']):
        ax.bar(list('ABCD'), [r[field] for r in summary], color=COLORS)
        ax.set(ylabel=ylabel)
        ax.grid(axis='y', alpha=0.2)
    save_figure(fig, figures / 'hardware')
    # Fixed seed selected before testing, not the seed with best test performance.
    make_confusion(runs_root, output, seeds[0])
    print('Generated', tables, figures, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--output', default='report')
    parser.add_argument('--protocol', default='protocol_frozen.json')
    parser.add_argument('--confusion-only', action='store_true', help='Render only the recorded confusion matrices')
    args = parser.parse_args()
    if args.confusion_only:
        make_confusion(args.runs_root,args.output,read_json(args.protocol)['seeds'][0])
    else:
        make(args.runs_root, args.output, args.protocol)
