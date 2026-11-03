"""Create release archives from audited A-D checkpoints, excluding optimizers/data."""
from pathlib import Path
import argparse
import zipfile
from src.utils import read_json,save_json,sha256


def package(output,supplement=False):
    from package_results import check_ready
    check_ready()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if supplement:
        audit=read_json('report/supplement/audit.json')
        assert audit['status']=='PASS' and audit['verified_conditions']==48
    manifest=dict(repository='CYBEZM/gtsrb-resnet18-dw-se-coursework',
                  tag='v1.1.0-supplement-E' if supplement else 'v1.0.0-three-seed',archives=[])
    archives=([('weights_E_all_seeds.zip',[11,22,33])] if supplement else
              [('weights_seed11.zip',[11]),('weights_all_seeds.zip',[11,22,33])])
    for name,seeds in archives:
        dest=output/name
        records=[]
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for v in ('E' if supplement else 'ABCD'):
                for seed in seeds:
                    source=Path(f'runs/{v}/{seed}/best.ckpt')
                    complete=read_json(source.parent/'complete.json')
                    assert sha256(source)==complete['checkpoint_sha256']
                    archive.write(source,source.as_posix())
                    records.append(dict(path=source.as_posix(),sha256=sha256(source),bytes=source.stat().st_size))
        with zipfile.ZipFile(dest) as archive:
            assert archive.testzip() is None
        manifest['archives'].append(dict(name=name,sha256=sha256(dest),bytes=dest.stat().st_size,files=records))
    manifest_name='weights_supplement_manifest.json' if supplement else 'weights_manifest.json'
    save_json(manifest_name,manifest)
    save_json(output/manifest_name,manifest)
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='../../work/release_weights')
    p.add_argument('--supplement',action='store_true')
    a=p.parse_args();package(a.output,a.supplement)
