"""Grouped GTSRB split, canonical image cache, deterministic corruptions."""
import csv
import hashlib
import io
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .utils import read_json, save_json, sha256

SIZE = 48
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
PROTOCOL = {
    'version': 'gtsrb48-v1', 'size': SIZE, 'padding': 'edge', 'resize': 'PIL bilinear',
    'normalization_mean': NORMALIZE_MEAN, 'normalization_std': NORMALIZE_STD,
    'brightness': [0.7, 0.4, 0.2], 'motion': [3, 5, 9],
    'noise': [0.03, 0.06, 0.12], 'jpeg': [50, 25, 10], 'resolution': [32, 24, 12],
    'motion_angle_degrees': 0, 'noise_units': 'RGB values scaled to [0,1]',
    'order': 'edge pad -> resize 48 -> corruption -> tensor -> normalization',
}


def canonical_image(image):
    image = image.convert('RGB')
    a = np.asarray(image)
    h, w = a.shape[:2]
    delta_h, delta_w = max(h, w) - h, max(h, w) - w
    padded = np.pad(a, ((delta_h // 2, delta_h - delta_h // 2),
                        (delta_w // 2, delta_w - delta_w // 2), (0, 0)), mode='edge')
    return Image.fromarray(padded).resize((SIZE, SIZE), Image.Resampling.BILINEAR)


def scan(root):
    root = Path(root)
    candidates = [root / 'GTSRB/Training', root / 'GTSRB/Final_Training/Images']
    training = next((p for p in candidates if p.is_dir()), None)
    if training is None:
        raise FileNotFoundError('Training archive not extracted; run prepare_data.py')
    train_rows = []
    for class_dir in sorted(p for p in training.iterdir() if p.is_dir() and p.name.isdigit()):
        label = int(class_dir.name)
        annotation = class_dir / f'GT-{label:05d}.csv'
        with annotation.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f, delimiter=';'):
                path = class_dir / row['Filename']
                if int(row['ClassId']) != label:
                    raise ValueError(f'Bad train annotation: {path}')
                train_rows.append(dict(path=path.relative_to(root).as_posix(), label=label,
                                       track=f'{label:05d}/{path.stem.split("_")[0]}',
                                       width=int(row['Width']), height=int(row['Height']),
                                       roi=[int(row[k]) for k in ('Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')]))
    test_dir = root / 'GTSRB/Final_Test/Images'
    with (root / 'GT-final_test.csv').open(newline='', encoding='utf-8-sig') as f:
        test_rows = [dict(path=(test_dir / r['Filename']).relative_to(root).as_posix(),
                          label=int(r['ClassId']), width=int(r['Width']), height=int(r['Height']),
                          roi=[int(r[k]) for k in ('Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')])
                     for r in csv.DictReader(f, delimiter=';')]
    if len({r['path'] for r in train_rows + test_rows}) != len(train_rows + test_rows):
        raise ValueError('Duplicate sample paths')
    if set(r['label'] for r in train_rows) != set(range(43)):
        raise ValueError('Expected all 43 classes')
    for row in train_rows + test_rows:
        if not 0 <= row['label'] < 43:
            raise ValueError('Missing image or invalid label')
    return train_rows, test_rows


def build_cache(root, split_seed=2026, val_fraction=0.2):
    root = Path(root)
    cache = root / 'cache48'
    cache.mkdir(exist_ok=True)
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['split_seed'] != split_seed or manifest['val_fraction'] != val_fraction:
            raise ValueError('Existing split differs; use a new data directory to change it')
        for name in ('train_images.npy', 'test_images.npy'):
            if sha256(cache / name) != manifest['cache_sha256'][name]:
                raise ValueError(f'Cache integrity mismatch: {name}')
        return manifest
    train_rows, test_rows = scan(root)
    rng = np.random.default_rng(split_seed)
    train_idx, val_idx, summary = [], [], []
    for label in range(43):
        indices = [i for i, r in enumerate(train_rows) if r['label'] == label]
        tracks = sorted({train_rows[i]['track'] for i in indices})
        if len(tracks) < 2:
            raise ValueError(f'Class {label} has insufficient tracks')
        chosen = set(rng.permutation(tracks)[:max(1, min(len(tracks) - 1, round(val_fraction * len(tracks))))])
        a = [i for i in indices if train_rows[i]['track'] not in chosen]
        b = [i for i in indices if train_rows[i]['track'] in chosen]
        train_idx.extend(a)
        val_idx.extend(b)
        summary.append(dict(class_id=label, train=len(a), val=len(b), total_tracks=len(tracks), val_tracks=len(chosen)))
    train_tracks = {train_rows[i]['track'] for i in train_idx}
    val_tracks = {train_rows[i]['track'] for i in val_idx}
    assert not train_tracks & val_tracks
    assert sorted(train_idx + val_idx) == list(range(len(train_rows)))
    image_hash = {}
    def decode(row):
        path = root / row['path']
        content = path.read_bytes()  # Missing files fail here, without duplicate stat/open calls.
        with Image.open(io.BytesIO(content)) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError(f'CSV/image dimension mismatch: {path}')
            x1, y1, x2, y2 = row['roi']
            if not (0 <= x1 <= x2 < image.width and 0 <= y1 <= y2 < image.height):
                raise ValueError(f'Invalid ROI: {path}')
            pixels = np.asarray(canonical_image(image)).transpose(2, 0, 1).copy()
        return content, pixels
    for name, rows in [('train', train_rows), ('test', test_rows)]:
        arr = np.lib.format.open_memmap(cache / f'{name}_images.npy', mode='w+', dtype=np.uint8,
                                        shape=(len(rows), 3, SIZE, SIZE))
        raw_digest = hashlib.sha256()
        # Ordered map preserves the exact manifest/content-hash order while
        # overlapping Windows file IO. Only the main thread writes the cache.
        with ThreadPoolExecutor(max_workers=8) as workers:
            for i, (content, pixels) in enumerate(workers.map(decode, rows)):
                raw_digest.update(rows[i]['path'].encode('utf-8'))
                raw_digest.update(content)
                arr[i] = pixels
                if (i + 1) % 10000 == 0:
                    print('Cached', name, i + 1, flush=True)
        arr.flush()
        del arr
        image_hash[name] = raw_digest.hexdigest()
    manifest = dict(schema=1, split_seed=split_seed, val_fraction=val_fraction,
                    train_indices=sorted(train_idx), val_indices=sorted(val_idx),
                    train_rows=train_rows, test_rows=test_rows, per_class=summary,
                    train_tracks=len(train_tracks), val_tracks=len(val_tracks), track_overlap=0,
                    raw_content_sha256=image_hash, preprocessing=PROTOCOL,
                    cache_sha256={n: sha256(cache / n) for n in ('train_images.npy', 'test_images.npy')})
    save_json(manifest_path, manifest)
    return manifest


def normalize(x):
    # Fixed affine normalization, independent of validation/test statistics.
    return (x - 0.5) / 0.5


def augment(x, generator):
    """Per-image affine jitter + brightness, GPU-batched with fixed RNG stream."""
    import torch
    import torch.nn.functional as F
    n = x.shape[0]
    u = torch.rand((n, 5), device=x.device, generator=generator)
    angle = (u[:, 0] * 20 - 10) * (math.pi / 180)
    scale = 0.95 + 0.1 * u[:, 1]
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = torch.cos(angle) / scale
    theta[:, 1, 1] = theta[:, 0, 0]
    theta[:, 0, 1] = -torch.sin(angle) / scale
    theta[:, 1, 0] = -theta[:, 0, 1]
    theta[:, :, 2] = u[:, 2:4] * 0.2 - 0.1  # normalized coordinates: +/- 5% image width/height
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return (x * (0.8 + 0.4 * u[:, 4, None, None, None])).clamp(0, 1)


def corrupt(array, kind, level, sample_id):
    image = Image.fromarray(array.transpose(1, 2, 0))
    value = PROTOCOL[kind][level - 1]
    if kind == 'brightness':
        image = ImageEnhance.Brightness(image).enhance(value)
    elif kind == 'motion':
        # Horizontal motion with edge padding, explicit odd-length line kernel.
        a = np.asarray(image).astype(np.float32)
        radius = value // 2
        padded = np.pad(a, ((0, 0), (radius, radius), (0, 0)), mode='edge')
        out = sum(padded[:, j:j + SIZE] for j in range(value)) / value
        image = Image.fromarray(np.rint(out).clip(0, 255).astype(np.uint8))
    elif kind == 'noise':
        token = f'{PROTOCOL["version"]}/{sample_id}/{kind}/{level}'.encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little')
        noise = np.random.default_rng(seed).normal(0, value * 255, (SIZE, SIZE, 3))
        image = Image.fromarray(np.rint(np.asarray(image).astype(float) + noise).clip(0, 255).astype(np.uint8))
    elif kind == 'jpeg':
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=value, subsampling=2, optimize=False)
        stream.seek(0)
        with Image.open(stream) as decoded:
            image = decoded.convert('RGB').copy()
    elif kind == 'resolution':
        image = image.resize((value, value), Image.Resampling.BILINEAR).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    else:
        raise ValueError(kind)
    return np.asarray(image).transpose(2, 0, 1).copy()
