"""Generate all quantitative tables/figures directly from recorded outputs."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.utils import read_json, save_json, sha256

COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def make(runs_root, output):
    runs_root, output = Path(runs_root), Path(output)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    runs, per_seed, summary, robust_rows, sources = {}, [], [], [], []
    for v in 'ABCD':
        runs[v] = []
        for seed in (11, 22, 33):
            path = runs_root / v / str(seed)
            clean = read_json(path / 'test_metrics.json')
            robust = read_json(path / 'robustness_metrics.json')
            if len(robust) != 16:
                raise ValueError(f'Incomplete conditions: {path}')
            complete = read_json(path / 'complete.json')
            if clean['checkpoint_sha256'] != complete['checkpoint_sha256']:
                raise ValueError('Checkpoint provenance mismatch')
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
        cmp = read_json(runs_root / v / '11/complexity.json')
        profile = read_json(output / 'profiles' / f'{v}.json')
        summary.append(dict(variant=v, n_seeds=3, accuracy_mean=float(acc.mean()), accuracy_std=float(acc.std(ddof=1)),
                            macro_f1_mean=float(f1.mean()), macro_f1_std=float(f1.std(ddof=1)),
                            parameters=cmp['parameters'], conv_linear_macs=cmp['conv_linear_macs'],
                            latency_median_ms=profile['latency_median_ms'], latency_p95_ms=profile['latency_p95_ms'],
                            throughput_images_s=profile['throughput_images_s'], weight_file_bytes=profile['weight_file_bytes']))
        for kind in ('brightness', 'motion', 'noise', 'jpeg', 'resolution'):
            for level in (1, 2, 3):
                vals = [r['robust'][f'{kind}_{level}'] for r in runs[v]]
                accuracies = np.array([r['accuracy'] for r in vals])
                scores = np.array([r['macro_f1'] for r in vals])
                robust_rows.append(dict(variant=v, kind=kind, severity=level, value=vals[0]['value'],
                    accuracy_mean=float(accuracies.mean()), accuracy_std=float(accuracies.std(ddof=1)),
                    macro_f1_mean=float(scores.mean()), macro_f1_std=float(scores.std(ddof=1)),
                    accuracy_drop_pp=float(100 * (acc - accuracies).mean())))
    write_csv(tables / 'per_seed.csv', per_seed)
    write_csv(tables / 'summary.csv', summary)
    write_csv(tables / 'robustness.csv', robust_rows)
    contrasts = []
    for name, plus, minus in [('DW effect', 'B', 'A'), ('SE standard', 'C', 'A'), ('SE depthwise', 'D', 'B')]:
        differences = np.array([100 * (p['clean']['accuracy'] - m['clean']['accuracy']) for p, m in zip(runs[plus], runs[minus])])
        contrasts.append(dict(contrast=name, mean_accuracy_difference_pp=float(differences.mean()),
                              std_accuracy_difference_pp=float(differences.std(ddof=1)),
                              seed11_pp=float(differences[0]), seed22_pp=float(differences[1]), seed33_pp=float(differences[2])))
    interaction = [100 * ((runs['D'][i]['clean']['accuracy'] - runs['B'][i]['clean']['accuracy']) -
                         (runs['C'][i]['clean']['accuracy'] - runs['A'][i]['clean']['accuracy'])) for i in range(3)]
    contrasts.append(dict(contrast='SE interaction', mean_accuracy_difference_pp=float(np.mean(interaction)),
                          std_accuracy_difference_pp=float(np.std(interaction, ddof=1)),
                          seed11_pp=interaction[0], seed22_pp=interaction[1], seed33_pp=interaction[2]))
    write_csv(tables / 'contrasts.csv', contrasts)
    save_json(tables / 'summary.json', dict(models=summary, per_seed=per_seed, robustness=robust_rows,
              contrasts=contrasts, source_files=sources, note='Raw fractions retained; standard deviation uses ddof=1'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    for v, color in zip('ABCD', COLORS):
        histories = runs[v]
        for ax, field, label in zip(axes, ['train_loss', 'val_macro_f1'], ['Training loss', 'Validation Macro-F1']):
            curves = np.array([[r[field] for r in run['history']] for run in histories])
            if field == 'val_macro_f1':
                curves *= 100
            mean, std = curves.mean(0), curves.std(0, ddof=1)
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, label=v, color=color)
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
            std = np.array([base['accuracy_std']] + [r['accuracy_std'] for r in rows]) * 100
            ax.plot(range(4), mean, marker='o', label=v, color=color)
            ax.fill_between(range(4), mean - std, mean + std, color=color, alpha=0.14)
        ax.set(title=kind, xlabel='Severity (0 = clean)', ylabel='Accuracy (%)', xticks=[0, 1, 2, 3])
        ax.grid(alpha=0.2)
    axes.flat[0].legend(ncol=4)
    axes.flat[5].axis('off')
    axes.flat[5].text(0.05, 0.8, 'Three training seeds\nMean and sample SD\nFixed track-grouped split\nShared corrupted inputs\nSeverity scales differ by corruption', va='top', linespacing=1.8)
    save_figure(fig, figures / 'robustness')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for r, color in zip(summary, COLORS):
        for ax, field, xlabel, divisor in [(axes[0], 'parameters', 'Parameters (million)', 1e6),
                                            (axes[1], 'conv_linear_macs', 'Conv/Linear MACs (million)', 1e6)]:
            x, y = r[field] / divisor, r['accuracy_mean'] * 100
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    for ax, v in zip(axes, ['A', 'D']):
        cm = np.array(runs[v][0]['clean']['confusion_matrix'])
        norm = cm / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
        ax.set(title=f'{v}, seed 11', xlabel='Predicted class ID', ylabel='True class ID')
    fig.colorbar(im, ax=list(axes), shrink=0.75, label='Row-normalized frequency')
    save_figure(fig, figures / 'confusion')
    print('Generated', tables, figures, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--output', default='report')
    args = parser.parse_args()
    make(args.runs_root, args.output)
