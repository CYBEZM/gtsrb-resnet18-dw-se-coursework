"""Evaluate frozen checkpoints. Never selects epochs or changes training."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.data import PROTOCOL, corrupt, normalize
from src.models import TrafficResNet
from src.utils import metrics_from_confusion, read_json, save_json, seed_all, sha256


def condition_cache(root, manifest, kind, level):
    folder = Path(root) / 'cache48'
    clean_path = folder / 'test_images.npy'
    if kind == 'clean':
        return clean_path
    path = folder / f'test_{kind}_{level}.npy'
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        meta = read_json(metadata)
        if (meta['protocol'] == read_json_protocol() and
                meta['clean_sha256'] == manifest['cache_sha256']['test_images.npy'] and
                meta['sha256'] == sha256(path)):
            return path
        raise ValueError('Existing corruption cache does not match frozen protocol')
    clean = np.load(clean_path, mmap_mode='r')
    temporary = path.with_suffix('.tmp.npy')
    out = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=clean.shape)
    for i, row in enumerate(manifest['test_rows']):
        out[i] = corrupt(clean[i], kind, level, row['path'])
    out.flush()
    del out
    temporary.replace(path)
    save_json(metadata, dict(kind=kind, severity=level, protocol=read_json_protocol(),
              clean_sha256=manifest['cache_sha256']['test_images.npy'], sha256=sha256(path)))
    return path


def read_json_protocol():
    # Normalize tuples to JSON lists before comparing with serialized protocol.
    import json
    return json.loads(json.dumps(PROTOCOL))


@torch.inference_mode()
def evaluate(model, images, labels, batch_size=256):
    device = next(model.parameters()).device
    cm = torch.zeros(1849, dtype=torch.long, device=device)
    predictions, probabilities = [], []
    for begin in range(0, len(images), batch_size):
        x = torch.as_tensor(np.array(images[begin:begin + batch_size]), device=device)
        x = normalize(x.float() / 255).contiguous(memory_format=torch.channels_last)
        logits = model(x)
        probs = logits.softmax(1)
        pred = probs.argmax(1)
        y = labels[begin:begin + batch_size]
        cm += torch.bincount(43 * y + pred, minlength=1849)
        predictions.append(pred.cpu().numpy())
        probabilities.append(probs.cpu().numpy())
    return (metrics_from_confusion(cm.reshape(43, 43).cpu().numpy()),
            np.concatenate(predictions), np.concatenate(probabilities))


def evaluate_run(run, data_root, frozen_path):
    torch.set_num_threads(4)
    seed_all(2026)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this experiment protocol')
    run, data_root = Path(run), Path(data_root)
    frozen = read_json(frozen_path)
    manifest_path = data_root / 'cache48/manifest.json'
    if sha256(manifest_path) != frozen['manifest_sha256'] or frozen['corruption_protocol'] != read_json_protocol():
        raise ValueError('Dataset or corruption protocol changed after freeze')
    for source, expected in frozen['source_sha256'].items():
        if sha256(source) != expected:
            raise ValueError(f'Frozen source changed: {source}')
    cfg = read_json(run / 'config.json')
    recorded_source = {r['path']: r['sha256'] for r in read_json(run / 'environment.json')['source_files']}
    for source in ('src/models.py', 'src/data.py', 'train.py'):
        if recorded_source.get(source) != frozen['source_sha256'][source]:
            raise ValueError(f'Run used source different from frozen protocol: {source}')
    variant, seed = cfg['variant'], cfg['seed']
    if seed not in frozen['seeds'] or {k: v for k, v in cfg.items() if k != 'seed'} != frozen['configs'][variant]:
        raise ValueError('Run is not included in the frozen experiment plan')
    complete = read_json(run / 'complete.json')
    checkpoint_hash = sha256(run / 'best.ckpt')
    if complete['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Checkpoint hash mismatch')
    checkpoint = torch.load(run / 'best.ckpt', map_location='cpu', weights_only=False)
    if checkpoint['manifest_sha256'] != frozen['manifest_sha256']:
        raise ValueError('Checkpoint trained with a different dataset manifest')
    model = TrafficResNet(variant).cuda().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(checkpoint['model'])
    manifest = read_json(manifest_path)
    labels_np = np.array([r['label'] for r in manifest['test_rows']], dtype=np.int64)
    labels = torch.as_tensor(labels_np, device='cuda')
    prediction_dir = run / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    sample_ids = np.array([r['path'] for r in manifest['test_rows']])
    conditions = [('clean', 0)] + [(kind, level) for kind in ('brightness', 'motion', 'noise', 'jpeg', 'resolution') for level in (1, 2, 3)]
    result_path = run / 'robustness_metrics.json'
    results = read_json(result_path) if result_path.exists() else {}
    for kind, level in conditions:
        key = f'{kind}_{level}'
        if key in results:
            old = results[key]
            prediction_file = run / old['prediction_file']
            if (old['checkpoint_sha256'] != checkpoint_hash or
                    old['protocol_sha256'] != sha256(frozen_path) or
                    old['prediction_sha256'] != sha256(prediction_file)):
                raise ValueError('Existing evaluation result mismatch')
            continue
        path = condition_cache(data_root, manifest, kind, level)
        arrays = np.load(path, mmap_mode='r')
        metrics, predicted, probabilities = evaluate(model, arrays, labels)
        dest = prediction_dir / f'{key}.npz'
        np.savez_compressed(dest, sample_ids=sample_ids, labels=labels_np, predicted=predicted, probabilities=probabilities)
        metrics.update(dict(variant=variant, seed=seed, kind=kind, severity=level,
                            value=None if kind == 'clean' else PROTOCOL[kind][level - 1],
                            checkpoint_sha256=checkpoint_hash, checkpoint_epoch=checkpoint['epoch'],
                            protocol_sha256=sha256(frozen_path), manifest_sha256=frozen['manifest_sha256'],
                            input_cache_sha256=sha256(path), inference_precision='FP32, TF32 disabled',
                            prediction_file=dest.relative_to(run).as_posix(), prediction_sha256=sha256(dest)))
        results[key] = metrics
        save_json(result_path, results)
        if kind == 'clean':
            save_json(run / 'test_metrics.json', metrics)
        print(variant, seed, key, 'accuracy', round(metrics['accuracy'], 6), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--protocol', type=Path, default=Path('protocol_frozen.json'))
    args = parser.parse_args()
    evaluate_run(args.run, args.data_root, args.protocol)
