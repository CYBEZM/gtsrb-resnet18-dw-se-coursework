"""Download the public GTSRB archives used by torchvision; verify before extraction."""
import argparse
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests

BASE = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
ARCHIVES = {
    'GTSRB-Training_fixed.zip': '513f3c79a4c5141765e10e952eaa2478',
    'GTSRB_Final_Test_Images.zip': 'c7e4e6327067d32654124b0fe9e82185',
    'GTSRB_Final_Test_GT.zip': 'fe31e9c9270bbcd7b84b7f21a9d9d9e5',
}


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root, mirror=False):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in ARCHIVES.items():
        if mirror and name == 'GTSRB_Final_Test_GT.zip':
            continue
        url = (f'https://hf-mirror.com/datasets/Zzyy2000/GTSRB/resolve/main/{name}' if mirror else BASE + name)
        path = root / name
        if not path.exists() or digest(path, 'md5') != expected:
            temp = path.with_suffix('.zip.part')
            print('Downloading', name, flush=True)
            for attempt in range(8):
                size = temp.stat().st_size if temp.exists() else 0
                try:
                    with requests.get(url, headers={'Range': f'bytes={size}-'}, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            size = 0
                        elif not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
                            raise RuntimeError('Unexpected download byte range')
                        last = time.monotonic()
                        with temp.open('ab' if size else 'wb') as f:
                            for block in response.iter_content(1024 * 1024):
                                f.write(block)
                                size += len(block)
                                if time.monotonic() - last > 20:
                                    print(name, round(size / 1e6, 1), 'MB received', flush=True)
                                    last = time.monotonic()
                    break
                except requests.RequestException as exc:
                    print('Retry', attempt + 1, name, type(exc).__name__, flush=True)
                    if attempt == 7:
                        raise
            if digest(temp, 'md5') != expected:
                raise RuntimeError(f'MD5 mismatch: {name}; preserving partial file for inspection')
            temp.replace(path)
        marker = root / (name + '.extracted')
        if not marker.exists():
            with zipfile.ZipFile(path) as z:
                for entry in z.infolist():
                    # Reject traversal syntactically without tens of thousands of
                    # costly Windows filesystem resolve calls before extraction.
                    member = PurePosixPath(entry.filename.replace('\\', '/'))
                    if member.is_absolute() or '..' in member.parts or ':' in entry.filename:
                        raise RuntimeError('Unsafe archive path')
                z.extractall(root)
            marker.write_text(expected, encoding='utf-8')
        records.append(dict(name=name, url=url, original_url=BASE + name, md5=expected,
                            sha256=digest(path, 'sha256'), bytes=path.stat().st_size))
        print('Verified and extracted', name, flush=True)
    if mirror:
        # Public byte-preserving archive mirrors; annotation mirror retains official
        # test indices, image dimensions and ROI. Only suffix/path is converted.
        url = 'https://hf-mirror.com/datasets/bazyl/GTSRB/resolve/main/Test.csv'
        annotation = root / 'mirror_Test.csv'
        expected_sha = '7d67912235e4570d19fb6bc96e9b227b4d89e4cd2ed47e14b8e8d976e1eb82b1'
        if not annotation.exists():
            response = requests.get(url, timeout=(30, 120))
            response.raise_for_status()
            annotation.write_bytes(response.content)
        if digest(annotation, 'sha256') != expected_sha:
            raise RuntimeError('Annotation mirror SHA256 mismatch')
        fields = ['Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId']
        with annotation.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        ids = [int(Path(r['Path']).stem) for r in rows]
        if sorted(ids) != list(range(12630)):
            raise ValueError('Annotation mirror has unexpected test indices')
        with (root / 'GT-final_test.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fields, delimiter=';')
            writer.writeheader()
            for r in rows:
                record = {k: r[k] for k in fields if k != 'Filename'}
                record['Filename'] = f'{int(Path(r["Path"]).stem):05d}.ppm'
                writer.writerow(record)
        records.append(dict(name=annotation.name, url=url, sha256=expected_sha,
                            conversion='Path Test/NNNNN.png -> Filename NNNNN.ppm; other columns unchanged',
                            converted_sha256=digest(root / 'GT-final_test.csv', 'sha256')))
    (root / 'download_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/gtsrb'))
    parser.add_argument('--mirror', action='store_true', help='Use hash-verified public archive/annotation mirrors')
    args = parser.parse_args()
    prepare(args.data_root, args.mirror)
