"""Fetch a fixed annotated GTSDB subset with HTTP ranges and ZIP CRC checks.

Only data is read from the pinned archive. Dataset images are not redistributed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

import requests

REVISION = '0d1c1a77fce322ba23b57a5e3261fae934f5afe3'
URL = f'https://huggingface.co/datasets/GlowBond/GTSDB/resolve/{REVISION}/archive.zip?download=true'
SIZE = 1729369023
ARCHIVE_SHA256 = '6144b38827acf3989bd7cd000a424c52cfd46a49da07e586439e25cf52995547'


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position+offset if whence == 1 else SIZE+offset)
        if self.position < 0: raise ValueError('Negative archive offset')
        return self.position

    def read(self, size=-1):
        if size < 0: size = SIZE-self.position
        size = min(size, SIZE-self.position)
        if not size: return b''
        if size > 16*1024*1024: raise ValueError('Refusing unexpectedly large range')
        begin, end = self.position, self.position+size-1
        response = requests.get(URL+f'&range={begin}-{end}',
            headers={'Range': f'bytes={begin}-{end}'}, stream=True, timeout=45)
        try:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {begin}-{end}/{SIZE}':
                raise RuntimeError(f'Range not honored: {response.status_code} {response.headers.get("Content-Range")}')
            data = response.content
            if len(data) != size: raise ValueError('Incomplete HTTP range')
            self.position += size
            return data
        finally:
            response.close()


def read_entry(name):
    with zipfile.ZipFile(RemoteZip()) as archive:
        return archive.read(name)  # zipfile independently checks member CRC32.


def inspect(destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteZip()) as archive:
        names = archive.namelist()
        documents = [n for n in names if n.endswith(('ReadMe.txt','gt.txt'))]
        for index, name in enumerate(documents):
            data = archive.read(name)
            path = destination / f'{index}_{Path(name).name}'
            path.write_bytes(data)
            print(name, len(data), data[:4400].decode('utf-8',errors='replace'), flush=True)
        (destination/'archive_listing.json').write_text(json.dumps(names,indent=2),encoding='utf-8')


def prepare(destination, count=100):
    destination.mkdir(parents=True, exist_ok=True)
    gt_name = 'TrainIJCNN2013/TrainIJCNN2013/gt.txt'
    raw_gt = read_entry(gt_name)
    rows = [line.strip().split(';') for line in raw_gt.decode('utf-8-sig').splitlines() if line.strip()]
    scenes = sorted({r[0] for r in rows})[:count]
    selected = [r for r in rows if r[0] in scenes]
    selection = dict(source='GTSDB annotated training road scenes, no model training on this dataset',
        mirror_url=URL, revision=REVISION, archive_bytes=SIZE,
        publisher_archive_sha256=ARCHIVE_SHA256,
        archive_hash_verified_locally=False,
        integrity='Pinned mirror revision; HTTPS Content-Range, ZIP member CRC32, downloaded member SHA256',
        selection_rule=f'First {count} lexicographically sorted annotated scene filenames; all their annotations',
        scene_filenames=scenes, annotations=selected,
        ground_truth_member=gt_name, ground_truth_sha256=hashlib.sha256(raw_gt).hexdigest(),
        limitations='External dataset identity does not guarantee physical-sign independence from GTSRB; '
                    'exact canonical image duplicates will be excluded before inference; no real-road detector is evaluated')
    plan_path = destination/'selection_frozen.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != selection:
        raise ValueError('Existing external selection differs')
    plan_path.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'gt.txt').write_bytes(raw_gt)
    prefix = 'TrainIJCNN2013/TrainIJCNN2013/'
    def fetch(name):
        path = destination/'scenes'/name
        if not path.exists():
            data = read_entry(prefix+name)
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(data); temporary.replace(path)
        data = path.read_bytes()
        return dict(member=prefix+name, path=path.relative_to(destination).as_posix(),
                    bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        files=[]
        for row in pool.map(fetch,scenes):
            files.append(row)
            if len(files)%10==0: print(f'External scenes downloaded: {len(files)}/{len(scenes)}',flush=True)
    (destination/'download_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print('External subset downloaded:',len(scenes),'scenes;',len(selected),'signs',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('../../work/gtsdb_external'))
    parser.add_argument('--inspect-only',action='store_true')
    parser.add_argument('--count',type=int,default=100)
    args=parser.parse_args()
    inspect(args.output) if args.inspect_only else prepare(args.output,args.count)
