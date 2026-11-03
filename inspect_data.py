"""Inspect only train/validation examples before freezing the test protocol."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data import PROTOCOL, corrupt
from src.utils import read_json, save_json, sha256


def inspect(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'cache48/manifest.json'
    m = read_json(manifest_path)
    arrays = np.load(root / 'cache48/train_images.npy', mmap_mode='r')
    val = m['val_indices']
    examples = [next(i for i in val if m['train_rows'][i]['label'] == label) for label in (1, 13, 25, 38)]
    kinds = ('brightness', 'motion', 'noise', 'jpeg', 'resolution')
    fig, axes = plt.subplots(4, 6, figsize=(10, 7), layout='constrained')
    for row, i in enumerate(examples):
        sample = m['train_rows'][i]
        a = arrays[i]
        images = [a] + [corrupt(a, kind, 2, sample['path']) for kind in kinds]
        for col, (image, title) in enumerate(zip(images, ('clean', *kinds))):
            axes[row, col].imshow(image.transpose(1, 2, 0))
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)
            if col == 0:
                axes[row, col].text(-0.05, 0.5, f'class {sample["label"]}', rotation=90,
                                   transform=axes[row, col].transAxes, ha='right', va='center')
    fig.savefig(output / 'validation_corruptions.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.2), layout='constrained')
    x = np.arange(43)
    ax.bar(x, [r['train'] for r in m['per_class']], label='Train', color='#0072B2')
    ax.bar(x, [r['val'] for r in m['per_class']], bottom=[r['train'] for r in m['per_class']],
           label='Validation', color='#E69F00')
    ax.set(xlabel='Class ID', ylabel='Images', xticks=np.arange(0, 43, 3))
    ax.legend()
    fig.savefig(output / 'class_distribution.png', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    with (output / 'split_counts.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, list(m['per_class'][0]))
        writer.writeheader()
        writer.writerows(m['per_class'])
    summary = dict(train_images=len(m['train_indices']), val_images=len(val),
                   supplied_training_images=len(m['train_rows']), test_images=len(m['test_rows']),
                   train_tracks=m['train_tracks'], val_tracks=m['val_tracks'], track_overlap=m['track_overlap'],
                   min_width=min(r['width'] for r in m['train_rows']), max_width=max(r['width'] for r in m['train_rows']),
                   min_height=min(r['height'] for r in m['train_rows']), max_height=max(r['height'] for r in m['train_rows']),
                   manifest_sha256=sha256(manifest_path), visual_examples=[m['train_rows'][i] for i in examples],
                   dataset_version='GTSRB-Training_fixed.zip (torchvision version); not the 39,209-image final training release',
                   protocol=PROTOCOL)
    save_json(output / 'dataset_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--output', type=Path, default=Path('report/data'))
    args = parser.parse_args()
    inspect(args.data_root, args.output)
