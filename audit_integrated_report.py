"""Check the integrated presentation against original metrics and selected NPZ rows.

Does not write to original evidence or replace its independent metric audits.
"""
import re
from pathlib import Path
from statistics import mean, stdev, median

import numpy as np
from PIL import Image

from review_data import ROOT, VARIANTS, SEEDS, collect, pm
from src.utils import read_json, save_json, sha256


def audit():
    data=collect(); out=ROOT/'report/integrated'
    summary=read_json(out/'summary.json')
    for row in summary['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    robust={(r['variant'],r['kind'],r['severity']):r for r in summary['robustness']}
    # Recompute all five-model, three-level summaries directly from run metrics.
    for v in VARIANTS:
        for kind in ('brightness','motion','noise','jpeg','resolution'):
            for severity in (1,2,3):
                values=[read_json(ROOT/f'runs/{v}/{s}/robustness_metrics.json')[f'{kind}_{severity}']['accuracy'] for s in SEEDS]
                row=robust[v,kind,severity]
                assert abs(row['accuracy_mean']-mean(values))<1e-12
                assert abs(row['accuracy_sd']-stdev(values))<1e-12
        p=data['profiles'][v]
        assert p['latency_median_ms']==median(x['median_ms'] for x in p['latency_runs'])
        assert p['latency_p95_ms']==median(x['p95_ms'] for x in p['latency_runs'])
        assert p['throughput_images_s']==median(p['throughput_runs_images_s'])
    comparisons=read_json(out/'comparison_sources.json')
    for row in comparisons['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    checked=0
    def check_records(sample_id,records,condition):
        nonlocal checked
        for v,record in records.items():
            with np.load(ROOT/f'runs/{v}/11/predictions/{condition}.npz',allow_pickle=False) as raw:
                indices=np.flatnonzero(raw['sample_ids']==sample_id); assert len(indices)==1
                i=int(indices[0]); pred=int(raw['predicted'][i])
                assert pred==record['class_id']
                assert float(raw['probabilities'][i,pred])==record['probability']
                assert (pred==int(raw['labels'][i]))==record['correct']
                checked+=1
    for row in comparisons['paired_cases']:
        check_records(row['sample_id'],row['models'],'clean_0')
    for row in comparisons['corruption_case']['conditions']:
        check_records(comparisons['corruption_case']['sample_id'],row['models'],row['condition'])
    # Recheck the deterministic paired selection and candidate counts.
    with np.load(ROOT/'runs/A/11/predictions/clean_0.npz') as raw:
        ids=raw['sample_ids'].copy(); labels=raw['labels'].copy()
    predictions={}
    for v in VARIANTS:
        with np.load(ROOT/f'runs/{v}/11/predictions/clean_0.npz') as raw:
            predictions[v]=raw['predicted'].copy()
    for row,(ref,other,ref_correct) in zip(comparisons['paired_cases'],
                                          [('A','C',False),('A','C',True),('B','E',False),('B','E',True)]):
        mask=((predictions[ref]==labels)==ref_correct)&((predictions[other]==labels)!=ref_correct)
        eligible=sorted(ids[mask].tolist())
        assert row['sample_id']==eligible[0] and row['eligible_samples']==len(eligible)
    eligible=np.logical_and.reduce([predictions[v]==labels for v in VARIANTS])
    disagrees=np.zeros(len(labels),dtype=bool)
    for kind in ('brightness','motion','noise','jpeg','resolution'):
        arr=[]
        for v in VARIANTS:
            with np.load(ROOT/f'runs/{v}/11/predictions/{kind}_3.npz') as raw:
                arr.append(raw['predicted'].copy())
        arr=np.stack(arr); disagrees|=(arr!=arr[0]).any(0)
    chosen=sorted(ids[eligible&disagrees].tolist())
    assert comparisons['corruption_case']['sample_id']==chosen[0]
    assert comparisons['corruption_case']['eligible_samples']==len(chosen)
    compact=read_json(out/'compact_sources.json')
    for row in compact['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    with np.load(ROOT/'report/external/C_11.npz') as raw:
        for row in compact['road_cases']['cases']:
            i=np.flatnonzero(raw['sample_ids']==row['sample_id']); assert len(i)==1
            assert int(raw['labels'][i[0]])==row['label']
            assert int(raw['predicted'][i[0]])==row['predicted']
    components=read_json(out/'component_analysis.json')
    for row in components['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    for v in VARIANTS:
        c=components['models'][v]
        assert sum(c['parameters_by_component'].values())==data['models'][v]['parameters']
        for s in SEEDS:
            import csv
            with (ROOT/f'runs/{v}/{s}/log.csv').open(newline='',encoding='utf-8-sig') as f:
                logs=list(csv.DictReader(f))
            changes=[100*(float(logs[i]['val_macro_f1'])-float(logs[i-1]['val_macro_f1'])) for i in range(5,len(logs))]
            row=next(r for r in components['validation_events'] if r['variant']==v and r['seed']==s)
            assert row['max_f1_drop_pp']==min(changes)
    assert [r['size'] for r in components['receptive_field_standard']]==[12,12,6,3,2]
    assert [r['size'] for r in components['receptive_field_adapted']]==[48,48,24,12,6]
    assert [r['receptive_field'] for r in components['receptive_field_adapted']]==[3,11,25,53,109]
    explanatory=read_json(out/'explanatory_sources.json')
    assert explanatory['generator_sha256']==sha256(ROOT/'make_explanatory_figures.py')
    for row in explanatory['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    assert explanatory['modules']['source_sha256']==sha256(ROOT/'src/models.py')
    assert explanatory['grouping']['manifest_sha256']==read_json(ROOT/'report/data/dataset_summary.json')['manifest_sha256']
    groups=explanatory['grouping']['groups']
    assert len(groups)==2 and groups[0]['track']!=groups[1]['track']
    for group in groups:
        assert len(group['samples'])==3
        for sample in group['samples']:
            path=Path(sample['path'])
            assert path.parent.name+'/'+path.stem.split('_')[0]==group['track']
    scene=explanatory['task_scene']
    assert scene['scene_sha256']==compact['road_cases']['scene_sha256']
    external=read_json(ROOT/'report/external/input_protocol.json')
    crop=next(r for r in external['samples'] if r['sample_id']==scene['sample_id'])
    assert crop['canonical_sha256']==scene['canonical_sha256']
    assert crop['box_inclusive']==scene['box_inclusive']
    for manifest in ('figure_sources.json','comparison_sources.json','compact_sources.json','explanatory_sources.json'):
        for row in read_json(out/manifest)['outputs']:
            path=ROOT/row['path']; assert sha256(path)==row['sha256']
            with Image.open(path) as image:
                assert image.convert('RGB').getpixel((0,0))==(255,255,255),str(path)
    manifest=read_json(out/'report_sources.json')
    report=ROOT/manifest['output']; text=report.read_text(encoding='utf-8')
    headings=re.findall(r'^## .+',text,re.M)
    assert len(headings)==len(set(headings))
    assert not any('补充' in h for h in headings)
    images=re.findall(r'!\[(.*?)\]\((.*?)\)',text)
    assert len(images)==9
    assert [int(re.match(r'图(\d+)',caption).group(1)) for caption,path in images]==list(range(1,10))
    assert len(re.findall(r'^\| ---',text,re.M))==6
    for caption,path in images:
        assert (report.parent/path).is_file(),path
    for v in VARIANTS:
        m=data['models'][v]; value=pm(m['accuracy_mean'],m['accuracy_sd'])
        assert value in text and value in (ROOT/'README.md').read_text(encoding='utf-8')
    for row in read_json(ROOT/'validation/preserved_seed11_files.json'):
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    protocol=read_json(ROOT/'protocol_frozen.json')
    for path,digest in protocol['source_sha256'].items():
        assert sha256(ROOT/path)==digest,path
    manifest=read_json(out/'report_sources.json')
    for row in manifest['sources']:
        assert sha256(ROOT/row['path'])==row['sha256'],row['path']
    assert manifest['output_sha256']==sha256(report)
    assert manifest['readme_sha256']==sha256(ROOT/'README.md')
    result=dict(status='PASS',gtsrb_robustness_aggregates_verified=75,
        illustrated_test_predictions_verified=checked,figures=9,tables=6,
        explanatory_sources_verified=True,
        component_analysis_verified=True,compact_road_predictions_verified=4,
        seed11_files_preserved=len(read_json(ROOT/'validation/preserved_seed11_files.json')),
        frozen_sources_unchanged=True,report_sha256=sha256(report),readme_sha256=sha256(ROOT/'README.md'))
    save_json(out/'audit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    audit()
