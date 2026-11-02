"""Verify per-class and paired-error tables against independent sklearn metrics."""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.utils import read_json,save_json,sha256


def main():
    out=Path('report/deep_analysis')
    def read(name):
        with (out/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    source=read_json(out/'analysis.json')
    assert source['status']=='PASS'
    for row in source['source_files']:assert sha256(row['path'])==row['sha256']
    perclass=read('per_class_per_seed.csv')
    cache={}
    for variant in 'ABCD':
        for seed in (11,22,33):
            path=Path(f'runs/{variant}/{seed}')
            result=read_json(path/'test_metrics.json')
            with np.load(path/result['prediction_file']) as raw:
                truth,pred=raw['labels'],raw['predicted']
                cache[variant,seed]=(truth.copy(),pred.copy())
                values=precision_recall_fscore_support(truth,pred,labels=list(range(43)),zero_division=0)
                rows=[r for r in perclass if r['variant']==variant and int(r['seed'])==seed]
                assert len(rows)==43
                for r in rows:
                    cls=int(r['class_id'])
                    for index,key in enumerate(('precision','recall','f1','support')):
                        assert np.isclose(float(r[key]),values[index][cls],rtol=0,atol=1e-14)
    pairs=read('paired_errors.csv')
    for r in pairs:
        v,ref=r['comparison'].split('-');seed=int(r['seed'])
        truth,pred=cache[v,seed];reftruth,refpred=cache[ref,seed]
        assert np.array_equal(truth,reftruth)
        yes,base=pred==truth,refpred==truth
        expected=dict(both_correct=(yes&base).sum(),both_wrong=(~yes&~base).sum(),
            repaired=(yes&~base).sum(),harmed=(~yes&base).sum(),
            net_correct=yes.sum()-base.sum(),samples=len(truth))
        for key,value in expected.items():assert int(r[key])==value
    save_json(out/'independent_audit.json',dict(status='PASS',class_rows=len(perclass),
        paired_rows=len(pairs),analysis_sha256=sha256(out/'analysis.json'),
        checks=['sklearn per-class precision/recall/F1/support','paired four-cell counts','all source hashes']))
    print('PASS deep analysis:',len(perclass),'class rows;',len(pairs),'paired rows')


if __name__=='__main__':main()
