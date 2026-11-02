"""Audit and summarize the separately frozen E experiment after all three runs."""
import csv
from pathlib import Path
from statistics import mean,stdev
import numpy as np
from audit_results import audit
from src.utils import read_json,save_json,sha256


def main():
    out=Path('report/supplement');out.mkdir(parents=True,exist_ok=True)
    protocol='protocols/supplement_E.json'
    result=audit('runs',protocol,variants='E')
    assert result['verified_conditions']==48
    a=read_json('runs/A/11/robustness_metrics.json')
    clean_a=read_json('runs/A/11/test_metrics.json')
    with np.load(Path('runs/A/11')/clean_a['prediction_file']) as raw:
        reference_ids,reference_labels=raw['sample_ids'].copy(),raw['labels'].copy()
    for seed in (11,22,33):
        for condition,row in read_json(f'runs/E/{seed}/robustness_metrics.json').items():
            assert row['input_cache_sha256']==a[condition]['input_cache_sha256']
            with np.load(Path(f'runs/E/{seed}')/row['prediction_file']) as raw:
                assert np.array_equal(raw['sample_ids'],reference_ids)
                assert np.array_equal(raw['labels'],reference_labels)
    profile=read_json(out/'profile_E.json')
    assert profile['checkpoint_sha256']==sha256('runs/E/11/best.ckpt')
    sources=[Path(protocol),out/'profile_E.json']
    rows=[]; robust=[];paired=[]
    for seed in (11,22,33):
        run=Path(f'runs/E/{seed}')
        clean=read_json(run/'test_metrics.json');complete=read_json(run/'complete.json')
        rows.append(dict(seed=seed,accuracy=clean['accuracy'],macro_f1=clean['macro_f1'],
            correct=clean['correct'],samples=clean['samples'],best_epoch=complete['best_epoch']+1))
        sources.extend(run/name for name in ('test_metrics.json','robustness_metrics.json','complete.json','log.csv','complexity.json'))
    for condition in a:
        values=[read_json(f'runs/E/{seed}/robustness_metrics.json')[condition] for seed in (11,22,33)]
        row=dict(condition=condition)
        for key in ('accuracy','macro_f1'):
            row[key+'_mean']=mean(r[key] for r in values)
            row[key+'_sample_sd']=stdev(r[key] for r in values)
        robust.append(row)
    for reference in 'AB':
        changes=[]
        for seed,e in zip((11,22,33),rows):
            source=Path(f'runs/{reference}/{seed}/test_metrics.json');sources.append(source)
            ref=read_json(source)
            changes.append(100*(e['accuracy']-ref['accuracy']))
        paired.append(dict(contrast='E-'+reference,seed_differences_pp=changes,
            mean_pp=mean(changes),sample_sd_pp=stdev(changes)))
    clean=next(r for r in robust if r['condition']=='clean_0')
    summary=dict(protocol_sha256=sha256(protocol),scope='Post-test exploratory, not independent confirmation',
        seeds=[11,22,33],per_seed=rows,robustness=robust,contrasts=paired,
        complexity=read_json('runs/E/11/complexity.json'),profile=profile,
        source_files=[dict(path=p.as_posix(),sha256=sha256(p)) for p in sources])
    save_json(out/'summary.json',summary)
    for name,data in [('per_seed.csv',rows),('robustness.csv',robust),('paired_contrasts.csv',paired)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    original=read_json('report/tables/summary.json')['models']
    table=['| 组别 | Accuracy % | Macro-F1 % | 参数量 | MACs M | 中位延迟 ms |',
           '| --- | --- | --- | --- | --- | --- |']
    for v in 'AB':
        r=next(r for r in original if r['variant']==v)
        table.append(f'| {v} | {100*r["accuracy_mean"]:.2f} ± {100*r["accuracy_std"]:.2f} | '
            f'{100*r["macro_f1_mean"]:.2f} ± {100*r["macro_f1_std"]:.2f} | {r["parameters"]:,} | '
            f'{r["conv_linear_macs"]/1e6:.3f} | {r["latency_median_ms"]:.3f} |')
    complexity=summary['complexity']
    table.append(f'| E | {100*clean["accuracy_mean"]:.2f} ± {100*clean["accuracy_sample_sd"]:.2f} | '
        f'{100*clean["macro_f1_mean"]:.2f} ± {100*clean["macro_f1_sample_sd"]:.2f} | {complexity["parameters"]:,} | '
        f'{complexity["conv_linear_macs"]/1e6:.3f} | {profile["latency_median_ms"]:.3f} |')
    paragraphs=['## 补充实验 仅替换stage3/4的深度可分离卷积','',
        'E组保留入口、stage1/2及所有shortcut的标准卷积，仅替换stage3/4残差主分支的3×3卷积。'
        '训练划分、增强、优化器、60轮训练及选模规则与原四组一致，使用全部三个种子。'
        '此问题在观察原A-D测试结果后提出，单独冻结协议，不属于原四组的独立验证，也不以测试集调整配置。','',
        '\n'.join(table),'',
        'Accuracy与Macro-F1为三个种子的均值±样本标准差；延迟仍为种子11检查点三轮硬件计时的中位数。'
        'E与原四组在不同时间测量，沿用相同设备、软件与口径，但功耗、温度或其他负载仍可能影响跨时间性能对比。','']
    for row in paired:
        paragraphs.append(f'{row["contrast"]}的配对准确率差为{row["mean_pp"]:+.3f} ± {row["sample_sd_pp"]:.3f}个百分点，'
            f'逐种子差值为'+ '、'.join(f'{v:+.3f}' for v in row['seed_differences_pp'])+'。')
    paragraphs+=['','分stage MACs为'+ '、'.join(f'stage{i+1}: {value/1e6:.3f} M' for i,value in enumerate(complexity['stage_macs']))+'。'
        '仅压缩深层不能消除浅层高分辨率计算，是否值得采用应同时结合准确率、参数量与实测延迟，'
        '不能仅从参数规模推断运行速度。','',
        '完整的干净集及15种退化条件汇总见`report/supplement/robustness.csv`；'
        '48份逐样本预测通过独立sklearn复核，且每个条件的输入哈希与原A-D完全一致。'
        '来源与协议哈希见`summary.json`和`audit.json`。','']
    (out/'section.md').write_text('\n'.join(paragraphs),encoding='utf-8')
    result['summary_sha256']=sha256(out/'summary.json')
    result['source_files']=summary['source_files']
    save_json(out/'audit.json',result)
    print('PASS supplemental E: 48 conditions')


if __name__=='__main__':main()
