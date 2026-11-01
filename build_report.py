"""Draft a Chinese report from complete measured results, never estimates.

Generated prose remains subject to final scientific and document-layout review.
"""
import argparse
from pathlib import Path

import numpy as np

from src.utils import read_json, sha256

REPO = 'https://github.com/CYBEZM/gtsrb-resnet18-dw-se-coursework'
NAMES = {'brightness': '亮度降低', 'motion': '水平运动模糊', 'noise': '高斯噪声',
         'jpeg': 'JPEG 压缩', 'resolution': '分辨率降低'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pm(mean, std):
    if std is None:
        return f'{100*mean:.2f}'
    return f'{100*mean:.2f} ± {100*std:.2f}'


def build(runs_root, report_root, protocol_path='protocol_frozen.json'):
    runs_root, report_root = Path(runs_root), Path(report_root)
    raw = read_json(report_root / 'tables/summary.json')
    data = read_json(report_root / 'data/dataset_summary.json')
    protocol = read_json(protocol_path)
    seeds = protocol['seeds']
    n_seeds = len(seeds)
    models = {r['variant']: r for r in raw['models']}
    expected_runs = {(v, seed) for v in 'ABCD' for seed in seeds}
    actual_runs = {(r['variant'], r['seed']) for r in raw['per_seed']}
    if (set(models) != set('ABCD') or actual_runs != expected_runs or
            len(raw['per_seed']) != len(expected_runs) or raw['seeds'] != seeds or
            raw['protocol_sha256'] != sha256(protocol_path)):
        raise ValueError('Report requires exactly the runs in the active frozen protocol')
    for row in raw['source_files']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError(f'Result changed after aggregation: {row["path"]}')
    a, b, c, d = (models[v] for v in 'ABCD')
    env = read_json(runs_root / 'A' / str(seeds[0]) / 'environment.json')
    cfg = protocol['configs']['A']
    dw_parameter_reduction = 100 * (1 - b['parameters'] / a['parameters'])
    dw_mac_reduction = 100 * (1 - b['conv_linear_macs'] / a['conv_linear_macs'])
    delta_b = 100 * (b['accuracy_mean'] - a['accuracy_mean'])
    delta_c = 100 * (c['accuracy_mean'] - a['accuracy_mean'])
    delta_d = 100 * (d['accuracy_mean'] - b['accuracy_mean'])
    latency_delta = 100 * (b['latency_median_ms'] / a['latency_median_ms'] - 1)
    main_table = table(['组别', 'Accuracy %', 'Macro-F1 %', '参数量', 'MACs M'],
        [[v, pm(r['accuracy_mean'], r['accuracy_std']), pm(r['macro_f1_mean'], r['macro_f1_std']),
          f'{r["parameters"]:,}', f'{r["conv_linear_macs"]/1e6:.3f}'] for v, r in models.items()])
    seed_table = table(['组别', '种子', 'Accuracy %', 'Macro-F1 %', '正确数 / 总数', '最佳 epoch'],
        [[r['variant'], r['seed'], f'{100*r["accuracy"]:.2f}', f'{100*r["macro_f1"]:.2f}',
          f'{r["correct"]} / {r["samples"]}', r['best_epoch']] for r in raw['per_seed']])
    hardware_table = table(['组别', '中位延迟 ms', 'P95 ms', '吞吐 images/s', '检查点 MB'],
        [[v, f'{r["latency_median_ms"]:.3f}', f'{r["latency_p95_ms"]:.3f}',
          f'{r["throughput_images_s"]:.1f}', f'{r["weight_file_bytes"]/1e6:.3f}'] for v, r in models.items()])
    robust_table = table(['退化', '强度', 'A Accuracy %', 'B Accuracy %', 'C Accuracy %', 'D Accuracy %'],
        [[NAMES[kind], level] +
         [pm(next(r for r in raw['robustness'] if r['variant'] == v and r['kind'] == kind and r['severity'] == level)['accuracy_mean'],
             next(r for r in raw['robustness'] if r['variant'] == v and r['kind'] == kind and r['severity'] == level)['accuracy_std'])
          for v in 'ABCD'] for kind in NAMES for level in (1, 2, 3)])
    contrast_headers = ['对比', '准确率差 pp'] if n_seeds == 1 else ['对比', '平均准确率差 pp', '差值标准差 pp']
    contrast_table = table(contrast_headers,
        [[r['contrast'], f'{r["mean_accuracy_difference_pp"]:+.3f}'] +
         ([] if n_seeds == 1 else [f'{r["std_accuracy_difference_pp"]:.3f}']) for r in raw['contrasts']])
    seed_description = (f'每组使用一个预先指定的训练种子{seeds[0]}' if n_seeds == 1 else
                        f'每组使用{n_seeds}个训练种子')
    uncertainty_note = ('各表为单次运行结果，不报告跨种子标准差或置信区间。'
                        if n_seeds == 1 else f'表中“±”为{n_seeds}个训练种子的样本标准差，而非置信区间。')
    contrast_note = ('下表为固定同一种子下的模型差值。未进行多种子重复，不能据此判断收益的稳定性或统计显著性。'
                     if n_seeds == 1 else '下表列出同种子配对差值及其样本标准差，不能仅凭平均数或标准差就宣称统计显著。')
    limitation_note = ('局限包括单一数据集版本、单次固定分组划分、单个训练种子和单一GPU平台。'
                       '本实验未估计训练随机性或其他数据划分带来的不确定性，结构差值只作描述性比较。'
                       if n_seeds == 1 else '局限包括单一数据集版本、单次固定分组划分、有限训练种子和单一GPU平台。'
                       '样本标准差主要描述训练随机性，没有覆盖其他数据划分的不确定性。')
    strongest = {}
    for v in 'ABCD':
        row = max((r for r in raw['robustness'] if r['variant'] == v and r['severity'] == 3),
                  key=lambda r: r['accuracy_drop_pp'])
        strongest[v] = row
    robust_text = '\n\n'.join(
        f'{v} 组在所设最高强度条件中，准确率下降最大的是{NAMES[r["kind"]]}，'
        f'较本组干净输入下降 {r["accuracy_drop_pp"]:.2f} 个百分点，'
        f'该条件准确率为 {pm(r["accuracy_mean"],r["accuracy_std"])}%。'
        for v, r in strongest.items())
    error_rows = []
    for v in ('A', 'D'):
        clean = read_json(runs_root / v / str(seeds[0]) / 'test_metrics.json')
        cm = np.array(clean['confusion_matrix'])
        np.fill_diagonal(cm, 0)
        for flat in np.argsort(cm.ravel())[-5:][::-1]:
            truth, pred = np.unravel_index(flat, cm.shape)
            if cm[truth, pred] > 0:
                error_rows.append([v, int(truth), int(pred), int(cm[truth, pred])])
    errors = table([f'组别 seed{seeds[0]}', '真实类别ID', '预测类别ID', '错误图片数'], error_rows)
    sources = table(['表图', '原始文件与字段'], [
        ['干净性能与逐种子结果', 'runs/{variant}/{seed}/test_metrics.json: accuracy, macro_f1, correct, samples'],
        ['模型复杂度', 'runs/{variant}/{seed}/complexity.json: parameters, conv_linear_macs'],
        ['训练曲线', 'runs/{variant}/{seed}/log.csv'],
        ['鲁棒性', 'runs/{variant}/{seed}/robustness_metrics.json'],
        ['推理性能', 'report/profiles/{variant}.json: latency_runs, throughput_runs_images_s'],
        ['数据划分', 'report/data/dataset_summary.json; data cache manifest.json'],
        ['汇总与来源校验', 'report/tables/summary.json: source_files'],
    ])
    text = f'''# 轻量化 ResNet 与 SE 注意力在交通标志识别中的应用研究

## 摘要

本项目针对裁剪交通标志的多类别图像分类，研究 CNN 结构对识别精度、模型复杂度和退化鲁棒性的影响。
以适配小尺寸图像的 ResNet-18 为基线，分别引入深度可分离卷积和 SE 通道注意力，构建四组控制变量实验。
使用 GTSRB 的 Training_fixed 数据版本，按完整拍摄序列分组划分训练集和验证集，并在固定官方测试集上评价。
{seed_description}，基线干净测试准确率为 {pm(a['accuracy_mean'],a['accuracy_std'])}%，
DW 组参数量减少 {dw_parameter_reduction:.2f}%，Conv/Linear MACs 减少 {dw_mac_reduction:.2f}%，
其准确率相对基线变化 {delta_b:+.2f} 个百分点。
SE 在标准卷积和 DW 骨架上的准确率变化分别为 {delta_c:+.2f} 和 {delta_d:+.2f} 个百分点。
此外，以五类固定合成退化和同一 GPU 上的实际计时分析结构取舍，不将理论 MACs 直接解释为硬件加速。

关键词：卷积神经网络；残差连接；深度可分离卷积；通道注意力；交通标志识别

## 1 任务背景与研究目标

交通标志的颜色、边缘轮廓和内部图案共同决定类别。小尺寸图片中的数字和方向图案可能相近，
光照、模糊和压缩还会进一步改变可见特征。因此，模型设计需要同时考虑干净输入上的识别能力、
计算资源和输入质量下降时的表现。

本项目的问题是：降低卷积复杂度会产生多大的精度代价；SE 在不同骨架上的识别表现如何变化；
结构变化是否同时改善理论复杂度与实际推理性能。目标不限定为改进模型必须超过基线。
本项目的输入是已裁剪的标志图片，不包含道路照片中的目标定位。
实验三复现的是 YOLO 和超分辨率算法；本项目在任务、数据处理、网络主体和消融设计上均与之不同。

## 2 模型原理与结构修改

### 2.1 残差卷积网络

ResNet 通过捷径连接将块输入与学习到的残差分支相加，使深层网络能够学习对已有表示的修正[1]。
本项目的四个残差阶段通道数为 64、128、256、512，每阶段两个 BasicBlock。
入口采用 3×3、步长为1的卷积并去除入口最大池化，以保留小图像细节。
后三个阶段首块执行下采样，随后全局平均池化并连接43类分类层。
该网络称为小图像适配版 ResNet-18，未直接宣称为原始 ImageNet 配置。

### 2.2 深度可分离卷积

标准 K×K 卷积同时完成空间和通道间的混合，不计偏置时参数量为核面积与输入、输出通道数的乘积。
深度可分离卷积先逐通道执行 K×K 空间滤波，再用1×1逐点卷积完成通道混合[2]。
其参数量为“核面积乘输入通道数”与“输入通道数乘输出通道数”之和。
当输入输出空间尺寸相同，MACs 可由相应卷积参数量乘输出位置数得到。

本项目只替换残差主分支中的3×3卷积，入口卷积和捷径投影保持一致。
DW 与 PW 之间不增加额外 BN 或激活，原主分支外部 BN/ReLU 位置不变，
避免将额外非线性与卷积类型的影响混在一起。

### 2.3 SE 通道注意力

SE 将特征图进行全局平均池化，通过压缩和恢复通道维度的两层映射产生权重，
再用 Sigmoid 得到每个通道的缩放系数[3]。本项目压缩比为16，
模块放在残差分支最后一个 BN 之后、与捷径相加之前。
SE 是已有方法，本项目的工作为任务适配、结构组合与实验分析。

### 2.4 四组控制变量实验

| 组别 | 残差主分支卷积 | SE |
| --- | --- | --- |
| A | 标准卷积 | 无 |
| B | 深度可分离卷积 | 无 |
| C | 标准卷积 | 有 |
| D | 深度可分离卷积 | 有 |

主比较 B−A 对应轻量化，C−A 对应标准骨架注意力，D−B 对应轻量化骨架注意力。
另计算两种骨架上的注意力收益差，作为交互效应的描述统计。

## 3 数据处理与防泄漏

使用 GTSRB 的 `GTSRB-Training_fixed.zip` 数据版本，包含 {data['supplied_training_images']:,} 张训练图片，
官方测试图片为 {data['test_images']:,} 张。本项目没有使用常见的39,209张最终训练版本；
因此不能直接将本项目与采用该版本或其他裁剪方法的文献数值作公平排名。
图片 ZIP 通过 torchvision 所列 MD5 校验。由于原托管地址连接失败，
实际下载使用公开原 ZIP 镜像；测试标签镜像校验 SHA256，并核对编号、图片尺寸及 ROI。

同一标志拍摄序列的连续帧可能高度相似。使用类别ID与文件名前缀共同构成 track 键，
固定划分种子2026，在每类内划分完整序列。
得到训练集 {data['train_images']:,} 张、{data['train_tracks']} 个序列，
验证集 {data['val_images']:,} 张、{data['val_tracks']} 个序列，
两者序列交集为 {data['track_overlap']}。所有模型与训练种子共用同一划分。
没有将随机相邻帧拆分后得到的高验证准确率作为依据。

所有图片保持长宽比，以边缘像素补成正方形后双线性缩放至48×48。
不使用ROI进一步裁剪，固定各通道均值和标准差为0.5进行归一化。
训练增强采用±10°旋转、±5%平移、0.95–1.05缩放和0.8–1.2亮度变化，不进行翻转。

![训练与验证类别分布](data/class_distribution.png)

## 4 实验设置与评价协议

实验 GPU 为 {env['gpu']}，PyTorch {env['torch']}，CUDA运行库 {env['cuda_runtime']}。
模型从随机初始化开始训练，统一使用 {cfg['epochs']} 个epoch、batch size {cfg['batch_size']}、
SGD动量 {cfg['momentum']}、权重衰减 {cfg['weight_decay']}、初始学习率 {cfg['lr']}。
学习率前 {cfg['warmup_epochs']} 轮预热，之后余弦衰减至 {cfg['min_lr']}。
训练启用AMP FP16，验证和最终推理使用FP32，关闭TF32。
训练种子为{seeds}，不以测试集选出最佳种子。

模型按验证 Macro-F1 最大的epoch选择，平局时优先较低验证损失，再优先较早epoch。
模型、训练规则和退化参数冻结后统一评价测试集，不根据测试结果修改超参数。
Accuracy为正确预测图片数占比；Macro-F1对全部43类F1等权平均，
未预测出的类别按零处理。{uncertainty_note}

## 5 干净测试结果与消融分析

{main_table}

基线A的干净测试准确率为 {pm(a['accuracy_mean'],a['accuracy_std'])}%。
B相对A的准确率变化为 {delta_b:+.3f} 个百分点，
同时参数和Conv/Linear MACs分别减少 {dw_parameter_reduction:.2f}% 和 {dw_mac_reduction:.2f}%。
这组结果应与实际延迟和退化结果共同判断，不能只据参数量选择最终模型。

C相对A、D相对B的准确率变化分别为 {delta_c:+.3f} 和 {delta_d:+.3f} 个百分点。
{contrast_note}

{contrast_table}

![训练和验证曲线](figures/learning_curves.png)

训练曲线用于检查收敛、过拟合以及固定训练预算的限制。
各模型均使用同一预算，这不等同于证明各架构已达到其所有可能超参数下的最优性能。

![精度和复杂度取舍](figures/tradeoffs.png)

### 5.1 每个种子的原始结果

{seed_table}

### 5.2 错误类型

以下使用预先指定的seed{seeds[0]}，列出A与D中出现最多的非对角混淆，不按测试准确率选择展示种子。
类别中文含义可在代码 `src/labels.py` 中查到。错误数量不能直接解释为某类别更难，
还需考虑每类支持样本数、拍摄条件和外观相似性。

{errors}

![混淆矩阵](figures/confusion.png)

## 6 合成退化下的场景适配分析

固定五类退化，每类三档强度。亮度倍率为0.7、0.4、0.2；水平运动模糊核长3、5、9；
高斯噪声在归一到0–1的像素上标准差为0.03、0.06、0.12；JPEG quality为50、25、10；
降分辨率先缩至32、24、12再回采样到48。
所有退化都作用于同一标准预处理后的RGB输入，之后再转张量归一化。
随机噪声由样本ID和协议哈希固定，所有模型、种子使用完全相同的退化图像。

![验证集退化样例](data/validation_corruptions.png)

{robust_table}

{robust_text}

![Accuracy与退化强度](figures/robustness.png)

这些结果支持的是所定义图像退化条件下的性能描述。亮度降低只模拟部分低照度影响，
水平模糊也不覆盖所有相机运动；本项目未采集真实夜间、雨雾或车载数据。
不同退化强度不具有统一物理尺度，因此主要逐类比较，不把跨种类平均分作为唯一排名。

## 7 实际推理性能

性能测试使用同一GPU、FP32、channels-last、eval与inference_mode，输入预先放在GPU。
batch=1预热50次、采样200次、重复3轮，以CUDA Events计时并逐次同步；
表中的中位延迟和P95分别取三轮对应统计量的中位数。
batch=32连续执行200批，仅在计时边界同步，以处理图片数除总时间计算吞吐。
该计时不包括读图、CPU预处理和主机到GPU传输，也不等同于完整应用响应时间。

{hardware_table}

B相对A的中位延迟变化为 {latency_delta:+.2f}%。
MACs计数只包含卷积与线性层的乘加，未计池化、归一化和逐元素操作；
即使理论复杂度下降，实际运行仍受算子实现、访存和kernel开销等因素影响。
本实验记录到的计时差异不能单独用于证明具体硬件瓶颈，若需定位应进一步使用分析器。
桌面GPU还承担显示任务，重复测量反映的波动保存在原始计时文件中。

![硬件性能](figures/hardware.png)

## 8 应用与可复现运行

项目提供图片上传、模型选择、Top-3预测概率、固定退化对比及stage3的12×12 Grad-CAM。
热力图用于辅助观察，不作为注意力有效性的因果证明。分类概率未经校准，
模型也不具备可靠的未知类别拒识或道路照片目标定位能力。

代码仓库：{REPO}

```powershell
python prepare_data.py --data-root data/gtsrb --mirror
python verify.py --data-root data/gtsrb
python inspect_data.py --data-root data/gtsrb
python freeze_protocol.py --data-root data/gtsrb
python run_experiments.py --data-root data/gtsrb
python -m streamlit run app.py
```

环境安装、GPU解释器核验、试跑、单次训练和断点恢复见仓库README。
每次运行保存实际配置、源码快照、数据清单哈希、环境版本与检查点校验值。
表图直接由JSON和日志生成，内部保留未舍入数值，正文仅在显示时舍入。

## 9 结论与局限

本项目完成经典残差CNN的实现、深度可分离卷积与SE两因素消融，
并将场景适配落实为固定退化条件的定量评价。
DW相对基线减少 {dw_parameter_reduction:.2f}% 参数和 {dw_mac_reduction:.2f}% Conv/Linear MACs，
干净准确率变化 {delta_b:+.3f} 个百分点；SE在两种骨架上的效果分别由C−A和D−B给出。
最终选择需要结合表中的干净精度、各类退化和目标硬件性能，而非预设D组必然最好。

{limitation_note}
合成退化不代表真实道路泛化，Grad-CAM也不能证明因果机制。
后续可在新增独立场景数据和更严格统计设计下验证结论，再考虑stage局部替换或低数据量实验。

## 附录 结果来源映射

{sources}

本报告输入汇总文件SHA256：`{sha256(report_root / 'tables/summary.json')}`。
数据清单SHA256：`{data['manifest_sha256']}`。
冻结协议SHA256：`{sha256(protocol_path)}`。

## 参考文献

[1] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016. https://arxiv.org/abs/1512.03385

[2] Howard A G, et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 2017. https://arxiv.org/abs/1704.04861

[3] Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks. CVPR, 2018. https://arxiv.org/abs/1709.01507

[4] Stallkamp J, Schlipsing M, Salmen J, Igel C. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks, 2012. https://doi.org/10.1016/j.neunet.2012.02.016
'''
    dest = report_root / '大作业报告.md'
    dest.write_text(text, encoding='utf-8')
    print(dest, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-root', default='runs')
    parser.add_argument('--report-root', default='report')
    parser.add_argument('--protocol', default='protocol_frozen.json')
    args = parser.parse_args()
    build(args.runs_root, args.report_root, args.protocol)
