"""Create release archives from audited A-D checkpoints, excluding optimizers/data."""
from pathlib import Path
import argparse
import zipfile
from src.utils import read_json,save_json,sha256


def package(output):
    from package_results import check_ready
    check_ready()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest=dict(repository='CYBEZM/gtsrb-resnet18-dw-se-coursework',
                  tag='v1.0.0-three-seed',archives=[])
    for name,seeds in [('weights_seed11.zip',[11]),('weights_all_seeds.zip',[11,22,33])]:
        dest=output/name
        records=[]
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for v in 'ABCD':
                for seed in seeds:
                    source=Path(f'runs/{v}/{seed}/best.ckpt')
                    complete=read_json(source.parent/'complete.json')
                    assert sha256(source)==complete['checkpoint_sha256']
                    archive.write(source,source.as_posix())
                    records.append(dict(path=source.as_posix(),sha256=sha256(source),bytes=source.stat().st_size))
        with zipfile.ZipFile(dest) as archive:
            assert archive.testzip() is None
        manifest['archives'].append(dict(name=name,sha256=sha256(dest),bytes=dest.stat().st_size,files=records))
    save_json('weights_manifest.json',manifest)
    save_json(output/'weights_manifest.json',manifest)
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='../../work/release_weights')
    package(p.parse_args().output)
