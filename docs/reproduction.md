# 运行与复现说明

以下命令在仓库根目录执行，`python`须指向已安装依赖的解释器。本机训练使用`D:\Anaconda3\python.exe`，PowerShell中可将命令开头替换为`& 'D:\Anaconda3\python.exe'`。环境安装与最快演示见README，本页集中记录完整操作。

## 1. 核验环境与数据

```powershell
python -c "import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())"
python prepare_data.py --data-root data/gtsrb --mirror
python verify.py --data-root data/gtsrb
```

原托管可用时可去掉`--mirror`。图片ZIP校验官方MD5，镜像测试标签转换与SHA256写入`download_manifest.json`。只下载数据，不执行镜像脚本。GTSRB先边缘补方、再缩放到48×48，保留图片全部内容；训练/验证按类别和完整track分组，固定划分种子2026。

## 2. 使用已发布的结果与权重

```powershell
python status.py
python download_weights.py --all-seeds
python download_weights.py --supplement
python infer.py --checkpoint runs/E/11/best.ckpt --image YOUR_SIGN.png --output prediction.json --cam gradcam.png
```

下载脚本支持`--archive YOUR_DOWNLOADED_ZIP`离线安装。逐样本复核还需要README链接的完整证据包；只下载权重不会获得240份GTSRB预测数组。解压时与已有文件比对，不覆盖不同内容的原始证据。

## 3. 从头训练一个新的复现副本

已发布的`runs/`含原始指标，不能把新训练与这些记录混用。**仅在专门用于复现的全新克隆副本**中，先归档已发布结果：

```powershell
New-Item -ItemType Directory -Path published_results
Move-Item -LiteralPath runs,report -Destination published_results
python inspect_data.py --data-root data/gtsrb
python train.py --config configs/A.yaml --seed 11 --epochs 8 --data-root data/gtsrb --output pilot/A11
python freeze_protocol.py --data-root data/gtsrb
python run_experiments.py --data-root data/gtsrb
python run_supplement.py --data-root data/gtsrb
```

这会重跑A-D及E的三个种子，耗时远大于仅阅读或复核结果。试跑只看训练/验证，正式配置保持60轮。原源码哈希与配置匹配时，冻结脚本验证既有协议而不改写历史协议；新run记录自己的源码状态、数据与检查点哈希。精确数值可能因硬件、库或CUDA非完全确定算子而变化，新的结果须重新汇总，不能冒充原运行。

单次训练或恢复示例：

```powershell
python train.py --config configs/B.yaml --seed 22 --data-root data/gtsrb --resume
```

`--resume`读取最近完整epoch的`last.ckpt`，恢复模型、优化器和AMP scaler，随机流由种子/epoch重建。原已完成目录自动跳过。改变epoch预算时应使用新输出目录，不能修改已有run。已有种子11的评价协议仍为`protocols/protocol_v2_single_seed.json`；22/33使用`protocol_frozen.json`；E使用`protocols/supplement_E.json`，队列会处理对应关系。

运行中可用`python control_training.py pause`暂停，用`python control_training.py resume`继续，未保存的当前epoch恢复后重跑。`python control_training.py status`核验队列进程；`python status.py`查看训练和评价计数。五组实验目前都已完成，阅读报告不需要启动训练。

## 4. 道路输入与设备测量

```powershell
python prepare_external.py --count 100 --output data/gtsdb_external
python evaluate_external.py --external-root data/gtsdb_external --gtsrb-root data/gtsrb
python audit_external.py --external-root data/gtsdb_external
python make_external_scenes.py --external-root data/gtsdb_external
```

道路输入按推理前固定规则裁剪，类别沿用GTSRB定义。仅做分类，不输出检测指标。下载采用固定镜像版本的指定ZIP成员，校验范围、CRC32和逐图SHA256；未下载整包时不声称已在本地校验发布者整包SHA256。

新复现时须对五组在同一时段重新计时，不能用旧A-D计时和新E拼表。下面代码仅用于**新复现副本**，遇到已有文件会停止；原工作目录直接保留既有测量：

```powershell
@'
from pathlib import Path
from profile_model import profile
from src.utils import save_json
out = Path('report/supplement/contemporary_profiles')
out.mkdir(parents=True, exist_ok=True)
if any((out / f'{v}.json').exists() for v in 'ABCDE'):
    raise RuntimeError('Existing profiles must be preserved; use a fresh reproduction copy')
for v in 'ABCDE':
    save_json(out / f'{v}.json', profile(v, f'runs/{v}/11/best.ckpt'))
'@ | python -
```

## 5. 独立复核与重建内容整合稿

轻量汇总复核可直接运行`python audit_summary.py`，无需数据、GPU、权重和预测数组。完整复核需先准备完整证据包及相应依赖：

```powershell
python audit_results.py
python analyze_results.py
python audit_deep_analysis.py
python summarize_supplement.py
```

原A-D独立复核192个条件，E复核48个；结果保存在原各自审计文件中。所有训练都按验证Macro-F1、损失、早期epoch顺序选模。日志epoch从0起，报告展示轮次从1起。

生成当前稿的精简图、分层参数/感受野分析与统一正文（需要已下载的五组seed11权重及上述道路数据）：

```powershell
python make_compact_evidence.py --data-root data/gtsrb --external-root data/gtsdb_external
python make_integrated_figures.py
python build_integrated_report.py
python audit_integrated_report.py
python -m pip install -r requirements-report.txt
python export_pdf.py --source report/大作业报告_内容整合稿_v2.md --output report/大作业报告_内容整合稿_v2.pdf
```

精简案例/组件分析与统计制图脚本分进程执行，避免PyTorch和绘图库在本机加载不同OpenMP运行时。默认使用Windows宋体/黑体；PDF导出可用`--font`、`--bold-font`指定中文TrueType字体，制图脚本的字体也需相应修改。PDF导出后仍须渲染检查分页、字体和图表清晰度。

历史`build_report.py`、`build_improved_report.py`供重建旧版报告；A-D与E旧队列的自动收尾也沿用历史生成器。要得到当前内容整合稿，使用上面的新命令，避免混淆输出文件。

当前稿、代码与轻量结果打包：

```powershell
python package_results.py --output ../traffic_sign_cnn_内容整合稿_v2_代码与结果.zip
```

包内有文件级SHA256清单，不包含数据集、凭据、权重和大预测数组。添加`--include-weights`才包含15个最佳检查点、源码快照和240份GTSRB预测。恢复训练的`last.ckpt`保留本机，不进入交付包。
