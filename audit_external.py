"""Independently audit external predictions and preserve source/permission evidence."""
import argparse
from pathlib import Path
import shutil
import statistics
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.utils import read_json, save_json, sha256


def audit(external_root):
    out = Path('report/external')
    external_root = Path(external_root)
    selection = read_json(external_root/'selection_frozen.json')
    downloads = read_json(external_root/'download_manifest.json')
    for row in downloads:
        assert sha256(external_root/row['path']) == row['sha256']
    documents = []
    for filename in ('1_ReadMe.txt', 'gt.txt', 'selection_frozen.json', 'download_manifest.json'):
        source = external_root/filename
        target = out/('source_'+filename)
        shutil.copyfile(source,target)
        documents.append(dict(path=target.as_posix(),sha256=sha256(target)))
    text = (external_root/'1_ReadMe.txt').read_text(encoding='utf-8')
    permission = 'You are free to use this data package for any purpose you like.'
    assert permission in text
    source_manifest = dict(dataset='German Traffic Sign Detection Benchmark (GTSDB)',
        official_url='https://benchmark.ini.rub.de/gtsdb_dataset.html',
        mirror_url=selection['mirror_url'], revision=selection['revision'],
        permission_quote=permission, permission_document='source_1_ReadMe.txt',
        attribution='Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum; GTSDB authors',
        archive_sha256_claimed_by_publisher=selection['publisher_archive_sha256'],
        full_archive_hash_verified_locally=False, downloaded_scenes=downloads,
        documents=documents, integrity=selection['integrity'])
    save_json(out/'source_manifest.json',source_manifest)
    plan=read_json(out/'input_protocol.json')
    summary=read_json(out/'summary.json')
    ids=np.array([r['sample_id'] for r in plan['samples']])
    labels=np.array([r['class_id'] for r in plan['samples']])
    checked=[]
    for result in summary['results']:
        key=f"{result['variant']}_{result['seed']}"
        path=out/(key+'.npz')
        assert sha256(path)==result['prediction_sha256']
        assert sha256(out/'input_protocol.json')==result['input_protocol_sha256']
        checkpoint=Path(f"runs/{result['variant']}/{result['seed']}/best.ckpt")
        assert sha256(checkpoint)==result['checkpoint_sha256']
        with np.load(path) as raw:
            assert np.array_equal(raw['sample_ids'],ids)
            assert np.array_equal(raw['labels'],labels)
            probs,pred=raw['probabilities'],raw['predicted']
            assert probs.shape==(len(ids),43) and np.isfinite(probs).all()
            assert (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-5)
            assert np.array_equal(probs.argmax(1),pred)
            accuracy=accuracy_score(labels,pred)
            f1=f1_score(labels,pred,labels=sorted(set(labels)),average='macro',zero_division=0)
            cm=confusion_matrix(labels,pred,labels=list(range(43)))
            top3=float(np.any(np.argsort(probs,axis=1)[:,-3:]==labels[:,None],axis=1).mean())
            assert np.isclose(accuracy,result['accuracy'],atol=1e-14,rtol=0)
            assert np.isclose(f1,result['macro_f1_present_classes'],atol=1e-14,rtol=0)
            assert np.isclose(top3,result['top3_accuracy'],atol=1e-14,rtol=0)
            assert np.array_equal(cm,result['confusion_matrix'])
            assert int(np.sum(pred==labels))==result['correct']
        checked.append(dict(path=path.as_posix(),sha256=sha256(path)))
    for row in summary['models']:
        matches=[r for r in summary['results'] if r['variant']==row['variant']]
        assert sorted(r['seed'] for r in matches)==[11,22,33]
        for metric in ('accuracy','macro_f1_present_classes','top3_accuracy'):
            values=[r[metric] for r in matches]
            assert np.isclose(row[metric+'_mean'],statistics.mean(values),atol=1e-14,rtol=0)
            assert np.isclose(row[metric+'_sample_sd'],statistics.stdev(values),atol=1e-14,rtol=0)
    audit=dict(status='PASS',models=len(checked),samples=len(labels),
        exact_duplicates_excluded=len(plan['excluded_exact_duplicates']),
        summary_sha256=sha256(out/'summary.json'),input_protocol_sha256=sha256(out/'input_protocol.json'),
        source_manifest_sha256=sha256(out/'source_manifest.json'),prediction_files=checked,
        checks=['sklearn accuracy/present-class F1/confusion','top3','probabilities',
                'sample alignment','checkpoint/input hashes','means/sample SD','source scene SHA256'])
    save_json(out/'audit.json',audit)
    print('PASS external:',len(checked),'models,',len(labels),'samples')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--external-root',default='../../work/gtsdb_external')
    audit(parser.parse_args().external_root)
