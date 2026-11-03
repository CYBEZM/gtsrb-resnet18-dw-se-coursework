"""Compose a review draft as one experiment narrative, without altering evidence.

Numerical statements and the README result table share review_data.collect().
The previous report versions remain available as historical artifacts.
"""
from report_narrative import compose

from review_data import ROOT, VARIANTS, SEEDS, NAMES, KINDS, collect, csv_rows, pm, write_summary
from src.utils import read_json, save_json, sha256

REPORT='大作业报告_内容整合稿_v2'
URL='https://github.com/CYBEZM/gtsrb-resnet18-dw-se-coursework'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def fig(caption,path):
    return f'![{caption}]({path})'


def readme(data):
    m,p=data['models'],data['profiles']
    results=table(['模型','Accuracy %','Macro-F1 %','参数量','MACs M','GPU中位延迟 ms'],[
        [v+' '+NAMES[v],pm(m[v]['accuracy_mean'],m[v]['accuracy_sd']),
         pm(m[v]['macro_f1_mean'],m[v]['macro_f1_sd']),f'{m[v]["parameters"]:,}',
         f'{m[v]["conv_linear_macs"]/1e6:.3f}',f'{p[v]["latency_median_ms"]:.3f}'] for v in VARIANTS])
    return f'''# 交通标志CNN：结构改进与场景适配

从随机初始化实现小图像ResNet-18，连续研究全阶段DW、SE及深层局部DW的精度、复杂度、退化表现和道路裁剪分类。五组共享数据划分、60轮训练预算及验证选模规则，使用种子11、22、33，共15次训练、900轮、240项GTSRB评价。

![模型结构](report/integrated/architecture.png)

当前阅读版本为**内容整合稿v2，供审阅，版式待调整**：[报告PDF](report/{REPORT}.pdf) · [可编辑正文](report/{REPORT}.md)。正文保留6幅核心图、6个表，增加结构推导与结果讨论；历史版本保留在仓库记录与Releases中。

## 主要结果

{results}

分类指标为三种子均值±样本标准差；设备值统一来自同一时段、固定seed11、三轮FP32计时，包含网络前向，不含读图和预处理。[原始汇总与来源](report/integrated/summary.json)。B具有本设备上最低延迟，C干净精度最高，E提供另一种取舍；三种子支持描述性比较，不宣称显著性。

精简的同图退化与Grad-CAM、道路全景与成功/失败裁剪已在报告中合并。分层参数、理论感受野和验证回落由[组件分析代码](make_compact_evidence.py)重新计算，解释中区分实测、推导和待验证机制。

## 快速运行

本机实测解释器为`D:\\Anaconda3\\python.exe`，torch 2.7.1+cu118，RTX 4060 Laptop 8GB。下列`python`须指向CUDA版PyTorch环境；演示可在CPU运行。

```powershell
git clone {URL}.git
cd gtsrb-resnet18-dw-se-coursework
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python download_weights.py
python -m streamlit run app.py
```

默认下载A-D的seed11权重。`python download_weights.py --all-seeds`下载A-D全部种子；`python download_weights.py --supplement`下载E的三个权重。下载后校验归档和检查点SHA256，已有不同文件不会被覆盖。

上传裁剪后的交通标志即可查看预测、固定退化和Grad-CAM。该应用是已知类别分类，未实现整幅道路图像检测或可靠未知类拒识，概率未校准。

## 复现实验与报告

[完整运行说明](docs/reproduction.md)按数据准备、训练、恢复、评价、审计与报告生成排列；[实验沿革与证据边界](docs/provenance.md)集中说明协议修订及E的探索性质。

GTSRB使用Training_fixed的26,640张供给训练图像，按类别和track分组为21,360训练、5,280验证，官方测试12,630张；不是39,209张最终训练版本。GTSDB固定100张道路图像中的163个标注裁剪仅用于应用分析，未训练或调参。

```text
configs/                 A-E配置
src/                     数据、模型和公共函数
train.py / eval.py       训练与固定条件评价
runs/{{A-E}}/{{11,22,33}}/   原始配置、日志、指标；权重单独下载
report/integrated/       当前五组合并汇总、新图、选样与来源校验
report/deep_analysis/    逐类与配对错误的原始分析
report/external/         道路裁剪结果、输入协议、来源与许可
docs/                    操作说明与协议沿革
```

报告、README表格和新图均从`review_data.py`读取保存结果。`make_compact_evidence.py`生成精简案例与组件分析，`make_integrated_figures.py`生成五组曲线，`build_integrated_report.py`生成本稿及本页。详细来源见报告附录。

## 权重与完整证据

- [A-D十二个权重](https://github.com/CYBEZM/gtsrb-resnet18-dw-se-coursework/releases/tag/v1.0.0-three-seed)：通过`weights_manifest.json`校验。
- [E权重及完整证据包](https://github.com/CYBEZM/gtsrb-resnet18-dw-se-coursework/releases/tag/v1.1.0-supplement-E)：含15个最佳检查点、240份GTSRB预测、15份道路预测及源码快照。该Release内报告为历史版，当前稿见本页上方。

仓库直接保存轻量指标、日志、报告与图表；完整数据集和大权重不进入Git。独立汇总复核不需要GPU和完整数据，逐样本复核则需要完整证据包。许可和来源随数据清单保留。

## 方法来源

ResNet：[He et al., CVPR 2016](https://arxiv.org/abs/1512.03385)；DW：[Howard et al., 2017](https://arxiv.org/abs/1704.04861)；SE：[Hu et al., CVPR 2018](https://arxiv.org/abs/1709.01507)。数据与道路场景文献见报告参考文献。这里的贡献是组件理解、受控实验和任务适配。
'''


def main():
    data=collect(); write_summary(data)
    target=ROOT/f'report/{REPORT}.md'
    target.write_text(compose(data),encoding='utf-8')
    (ROOT/'README.md').write_text(readme(data),encoding='utf-8',newline='\n')
    sources=data['source_paths']+['report/integrated/comparison_sources.json',
        'report/integrated/figure_sources.json','report/integrated/component_analysis.json',
        'report/integrated/compact_sources.json','report_narrative.py','make_compact_evidence.py',
        'build_integrated_report.py','review_data.py']
    save_json(ROOT/'report/integrated/report_sources.json',dict(status='PASS',draft='content review v2',
        output=target.relative_to(ROOT).as_posix(),output_sha256=sha256(target),readme_sha256=sha256(ROOT/'README.md'),
        sources=[dict(path=p,sha256=sha256(ROOT/p)) for p in sources]))
    print(target,flush=True)


if __name__=='__main__':
    main()
