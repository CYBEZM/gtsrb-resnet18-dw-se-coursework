"""Meaningful implementation checks; creates no reported accuracy claims."""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import augment, build_cache, corrupt
from src.models import TrafficResNet, complexity
from src.utils import metrics_from_confusion, save_json


def verify(data_root=None):
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checks = {'device': device, 'models': {}}
    for variant in 'ABCD':
        model = TrafficResNet(variant).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        x = torch.rand(4, 3, 48, 48, device=device)
        generator = torch.Generator(device=device).manual_seed(123)
        a = augment(x, generator)
        b = augment(x, torch.Generator(device=device).manual_seed(123))
        assert torch.equal(a, b)
        before = model.fc.weight.detach().clone()
        y = torch.tensor([0, 1, 2, 42], device=device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            loss = torch.nn.functional.cross_entropy(model(a), y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, model.fc.weight)
        checks['models'][variant] = complexity(model)
        checks['models'][variant]['optimizer_step'] = 'passed'
        del model, optimizer
    cm = np.zeros((43, 43), dtype=int)
    cm[0, 0], cm[0, 1], cm[1, 1] = 1, 1, 1
    result = metrics_from_confusion(cm)
    assert np.isclose(result['accuracy'], 2 / 3)
    assert np.isclose(result['macro_f1'], (2 / 3 + 2 / 3) / 43)
    array = np.arange(3 * 48 * 48, dtype=np.uint8).reshape(3, 48, 48)
    for kind in ('brightness', 'motion', 'noise', 'jpeg', 'resolution'):
        for severity in (1, 2, 3):
            a = corrupt(array, kind, severity, 'sample')
            assert np.array_equal(a, corrupt(array, kind, severity, 'sample'))
            assert a.shape == array.shape and a.dtype == np.uint8
    checks['metrics_and_corruptions'] = 'passed'
    if data_root:
        manifest = build_cache(data_root)
        a, b = manifest['train_indices'], manifest['val_indices']
        rows = manifest['train_rows']
        assert not ({rows[i]['track'] for i in a} & {rows[i]['track'] for i in b})
        assert set(rows[i]['label'] for i in a) == set(rows[i]['label'] for i in b) == set(range(43))
        checks['dataset'] = dict(train=len(a), validation=len(b), test=len(manifest['test_rows']),
                                  train_tracks=manifest['train_tracks'], val_tracks=manifest['val_tracks'], overlap=0)
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', default='report/verification.json')
    args = parser.parse_args()
    results = verify(args.data_root)
    save_json(args.output, results)
    print(results)
