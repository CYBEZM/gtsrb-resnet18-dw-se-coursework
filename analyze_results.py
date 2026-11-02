"""Deeper descriptive analysis from preserved predictions/logs; no test tuning."""
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.labels import CLASS_NAMES_ZH
from src.utils import read_json, save_json, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'report/deep_analysis'
SEEDS = (11, 22, 33)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stat(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))


def save_figure(fig, name):
    for suffix in ('.png', '.svg'):
        fig.savefig(OUT / (name + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, histories, drops, sources, arrays = [], {}, [], [], {}
    colors = dict(zip('ABCD', ['#0072B2', '#E69F00', '#009E73', '#CC79A7']))
    for variant in 'ABCD':
        for seed in SEEDS:
            run = ROOT / 'runs' / variant / str(seed)
            metrics = read_json(run / 'test_metrics.json')
            path = run / 'predictions/clean_0.npz'
            assert sha256(path) == metrics['prediction_sha256']
            with np.load(path) as raw:
                arrays[variant, seed] = {k: raw[k].copy() for k in ('labels', 'predicted', 'sample_ids')}
            with (run / 'log.csv').open(newline='', encoding='utf-8') as stream:
                history = [{key: float(value) for key, value in row.items()}
                           for row in csv.DictReader(stream)]
            histories[variant, seed] = history
            assert len(history) == 60
            for index in range(43):
                rows.append(dict(variant=variant, seed=seed, class_id=index,
                    class_name=CLASS_NAMES_ZH[index], support=metrics['support'][index],
                    precision=metrics['per_class_precision'][index],
                    recall=metrics['per_class_recall'][index], f1=metrics['per_class_f1'][index]))
            for before, after in zip(history, history[1:]):
                drops.append(dict(variant=variant, seed=seed, epoch=int(after['epoch'])+1,
                    macro_f1_change_pp=100*(after['val_macro_f1']-before['val_macro_f1']),
                    val_loss_before=before['val_loss'], val_loss_after=after['val_loss'],
                    val_accuracy_change_pp=100*(after['val_accuracy']-before['val_accuracy']),
                    lr_before=before['lr'], lr_after=after['lr'],
                    train_loss_after=after['train_loss']))
            for name in ('test_metrics.json', 'log.csv', 'complete.json', 'predictions/clean_0.npz'):
                path = run / name
                sources.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path)))
    write_csv(OUT / 'per_class_per_seed.csv', rows)
    per_class = []
    for variant in 'ABCD':
        for index in range(43):
            selected = [r for r in rows if r['variant'] == variant and r['class_id'] == index]
            row = dict(variant=variant, class_id=index, class_name=CLASS_NAMES_ZH[index],
                       support=selected[0]['support'])
            for key in ('precision', 'recall', 'f1'):
                values = stat([r[key] for r in selected])
                row[key + '_mean'], row[key + '_sample_sd'] = values['mean'], values['sample_sd']
            per_class.append(row)
    write_csv(OUT / 'per_class_summary.csv', per_class)
    write_csv(OUT / 'validation_epoch_changes.csv', drops)
    # Exclude the initial learning ramp when describing temporary collapses.
    worst = sorted([r for r in drops if r['epoch'] >= 6], key=lambda r: r['macro_f1_change_pp'])[:8]
    write_csv(OUT / 'largest_validation_drops.csv', worst)
    paired, class_differences, stable = [], [], {}
    for plus, minus in [('B', 'A'), ('C', 'A'), ('D', 'B')]:
        repaired_sets, harmed_sets = [], []
        for seed in SEEDS:
            p, m = arrays[plus, seed], arrays[minus, seed]
            assert np.array_equal(p['sample_ids'], m['sample_ids']) and np.array_equal(p['labels'], m['labels'])
            p_ok, m_ok = p['predicted'] == p['labels'], m['predicted'] == m['labels']
            repair, harm = p_ok & ~m_ok, ~p_ok & m_ok
            repaired_sets.append(set(p['sample_ids'][repair].tolist()))
            harmed_sets.append(set(p['sample_ids'][harm].tolist()))
            paired.append(dict(comparison=f'{plus}-{minus}', seed=seed,
                both_correct=int((p_ok & m_ok).sum()), both_wrong=int((~p_ok & ~m_ok).sum()),
                repaired=int(repair.sum()), harmed=int(harm.sum()),
                net_correct=int(repair.sum()-harm.sum()), samples=len(p_ok)))
            for index in range(43):
                ix = p['labels'] == index
                class_differences.append(dict(comparison=f'{plus}-{minus}', seed=seed,
                    class_id=index, class_name=CLASS_NAMES_ZH[index], support=int(ix.sum()),
                    repaired=int((repair & ix).sum()), harmed=int((harm & ix).sum()),
                    net_correct=int((repair & ix).sum()-(harm & ix).sum()),
                    recall_difference_pp=100*float((p_ok[ix].astype(int)-m_ok[ix].astype(int)).mean())))
        stable[f'{plus}-{minus}'] = dict(
            repaired_all_seeds=sorted(set.intersection(*repaired_sets)),
            harmed_all_seeds=sorted(set.intersection(*harmed_sets)))
    write_csv(OUT / 'paired_errors.csv', paired)
    write_csv(OUT / 'paired_class_errors.csv', class_differences)
    save_json(OUT / 'stable_error_sample_ids.json', stable)
    trajectory = []
    for (variant, seed), history in histories.items():
        complete = read_json(ROOT / f'runs/{variant}/{seed}/complete.json')
        best, final = history[complete['best_epoch']], history[-1]
        trajectory.append(dict(variant=variant, seed=seed, selected_epoch=complete['best_epoch']+1,
            first_train_loss_below_001=next((int(r['epoch'])+1 for r in history if r['train_loss'] < .01), None),
            selected_val_f1=best['val_macro_f1'], selected_val_loss=best['val_loss'],
            final_val_f1=final['val_macro_f1'], final_val_loss=final['val_loss'],
            selected_minus_final_f1_pp=100*(best['val_macro_f1']-final['val_macro_f1']),
            final_train_accuracy=final['train_accuracy'], final_val_accuracy=final['val_accuracy'],
            final_train_val_gap_pp=100*(final['train_accuracy']-final['val_accuracy'])))
    write_csv(OUT / 'training_diagnostics.csv', trajectory)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    for column, seed in enumerate(SEEDS):
        for variant in 'ABCD':
            history = histories[variant, seed]
            epochs = np.arange(1, 61)
            axes[0, column].plot(epochs, [100*r['val_macro_f1'] for r in history],
                                 label=variant, color=colors[variant], lw=1.2)
            axes[1, column].plot(epochs, [r['val_loss'] for r in history], color=colors[variant], lw=1.2)
        axes[0, column].set(title=f'Seed {seed}', ylabel='Validation Macro-F1 (%)')
        axes[1, column].set(xlabel='Epoch', ylabel='Validation cross entropy', yscale='log')
        for ax in axes[:, column]: ax.grid(alpha=.2)
    axes[0, 0].legend(ncol=4)
    save_figure(fig, 'validation_by_seed')
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), layout='constrained')
    class_summary = []
    for ax, comparison in zip(axes, ('B-A', 'C-A', 'D-B')):
        means, sds = [], []
        for index in range(43):
            selected = [r for r in class_differences if r['comparison'] == comparison and r['class_id'] == index]
            values = stat([r['recall_difference_pp'] for r in selected])
            means.append(values['mean']); sds.append(values['sample_sd'])
            class_summary.append(dict(comparison=comparison, class_id=index,
                class_name=CLASS_NAMES_ZH[index], support=selected[0]['support'],
                recall_difference_mean_pp=values['mean'], recall_difference_sd_pp=values['sample_sd'],
                net_correct_mean=statistics.mean(r['net_correct'] for r in selected)))
        ax.bar(range(43), means, color=['#009E73' if v >= 0 else '#D55E00' for v in means])
        ax.errorbar(range(43), means, yerr=sds, fmt='none', ecolor='#444444', lw=.7, capsize=1)
        ax.axhline(0,color='#444444',lw=.7); ax.set(title=comparison,ylabel='Recall difference (pp)',xticks=range(0,43,3))
    axes[-1].set_xlabel('Class ID; error bars = sample SD across 3 seeds')
    save_figure(fig, 'class_recall_differences')
    write_csv(OUT / 'paired_class_summary.csv', class_summary)
    evidence = dict(status='PASS', seeds=list(SEEDS), source_files=sources,
        source_check='all clean prediction file hashes verified; identical sample IDs/labels across paired models',
        paired_errors=paired, largest_validation_drops=worst, training_diagnostics=trajectory,
        stable_errors={k:{name:len(values) for name,values in v.items()} for k,v in stable.items()},
        note='Descriptive post-test analysis; no iid-image significance test; no mechanistic causal claim')
    save_json(OUT / 'analysis.json', evidence)
    parts = ['## 补充分析 逐类别错误与训练稳定性', '',
        '本节对已保存的预测和日志作事后描述分析，不据此调整超参数或重新选择检查点。'
        '同一拍摄序列内的图片相关，因此不把所有测试图片当作独立重复来计算显著性。', '',
        '| 对比 | 种子 | 纠正原错误数 | 引入新错误数 | 净增加正确数 |', '| --- | --- | --- | --- | --- |']
    parts += [f'| {r["comparison"]} | {r["seed"]} | {r["repaired"]} | {r["harmed"]} | {r["net_correct"]} |' for r in paired]
    parts += ['', '“纠正”指参照模型错误而比较模型正确，“引入”指反方向变化。'
        '总精度的微小变化可能掩盖两类变化同时发生，不能把净提升解释为所有图片均有改善。', '',
        '![各类别召回率配对差值](deep_analysis/class_recall_differences.png)', '']
    for comparison in ('C-A','D-B'):
        selected = [r for r in class_summary if r['comparison'] == comparison]
        top = max(selected,key=lambda r:r['recall_difference_mean_pp'])
        low = min(selected,key=lambda r:r['recall_difference_mean_pp'])
        parts += [f'{comparison}的平均召回率差最大的是类别{top["class_id"]}（{top["class_name"]}，'
            f'支持样本{top["support"]}张），为{top["recall_difference_mean_pp"]:+.2f}个百分点；'
            f'最小的是类别{low["class_id"]}（{low["class_name"]}，支持样本{low["support"]}张），'
            f'为{low["recall_difference_mean_pp"]:+.2f}个百分点。'
            '这是对全部类别排序后的描述性极值，不能作为预先指定假设的验证。', '']
    parts += ['![分种子的验证指标与损失](deep_analysis/validation_by_seed.png)', '',
        '| 组别 | 种子 | 波动发生轮次 | Macro-F1变化 pp | 验证损失前后 | 学习率前后 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in worst[:5]:
        parts.append(f'| {r["variant"]} | {r["seed"]} | {r["epoch"]} | {r["macro_f1_change_pp"]:.2f} | '
            f'{r["val_loss_before"]:.3f} → {r["val_loss_after"]:.3f} | {r["lr_before"]:.5f} → {r["lr_after"]:.5f} |')
    parts += ['', '表中从第6轮起选择相邻轮次Macro-F1下降最大的事件，避免把初始学习过程混入暂时退化。'
        '验证损失和准确率与F1一起变化，可帮助区分单一指标波动与整体预测变化；'
        '学习率变化只用于定位训练阶段，不能单凭时间重合断言因果。'
        '训练损失较早接近零而验证损失仍有波动，提示拟合能力和泛化之间存在差距；'
        '小样本类别、序列分布、BN运行统计及优化过程都是待验证解释，现有日志不足以确定D表现较差的机制。', '',
        '所有检查点继续沿用原先的验证选模规则。逐轮变化、每类支持数与全部43类差值见'
        '`report/deep_analysis/*.csv`；来源文件及哈希见`analysis.json`。', '']
    (OUT / 'section.md').write_text('\n'.join(parts),encoding='utf-8')
    print('Deep analysis PASS:', OUT, flush=True)


if __name__ == '__main__':
    main()
