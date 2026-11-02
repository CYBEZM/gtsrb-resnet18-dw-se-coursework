"""Download the fixed project release and verify each file before extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from urllib.request import urlopen,Request
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def install(all_seeds=False,output='.',archive=None):
    manifest=json.loads(Path(__file__).with_name('weights_manifest.json').read_text(encoding='utf-8'))
    name='weights_all_seeds.zip' if all_seeds else 'weights_seed11.zip'
    entry=next(r for r in manifest['archives'] if r['name']==name)
    output=Path(output).resolve()
    if all((output/r['path']).is_file() and digest(output/r['path'])==r['sha256'] for r in entry['files']):
        print('All requested weights already verified.');return
    with tempfile.TemporaryDirectory(prefix='gtsrb_weights_') as temp:
        source=Path(archive) if archive else Path(temp)/name
        if not archive:
            url=f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/{name}'
            print('Downloading',url,flush=True)
            with urlopen(Request(url,headers={'User-Agent':'gtsrb-coursework-weights'}),timeout=90) as response,source.open('wb') as f:
                for block in iter(lambda:response.read(1024*1024),b''):f.write(block)
        assert source.stat().st_size==entry['bytes'] and digest(source)==entry['sha256'],'Archive hash mismatch'
        expected={r['path']:r for r in entry['files']}
        with zipfile.ZipFile(source) as z:
            assert len(z.namelist())==len(expected) and set(z.namelist())==set(expected)
            for name,row in expected.items():
                relative=PurePosixPath(name)
                assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
                destination=(output/name).resolve()
                assert destination.is_relative_to(output)
                if destination.exists():
                    assert digest(destination)==row['sha256'],f'Refusing to replace different file: {destination}'
                extracted=Path(temp)/Path(name).name
                with z.open(name) as f,extracted.open('wb') as g:
                    for block in iter(lambda:f.read(1024*1024),b''):g.write(block)
                assert digest(extracted)==row['sha256'] and extracted.stat().st_size==row['bytes']
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    import shutil
                    shutil.copyfile(extracted,destination)
                print('Verified',name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-seeds',action='store_true')
    p.add_argument('--output',default='.');p.add_argument('--archive',help='Verify/install an already downloaded ZIP')
    a=p.parse_args();install(a.all_seeds,a.output,a.archive)
