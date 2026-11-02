# 交通标志 CNN 结构改进与鲁棒性实验

从随机初始化训练 GTSRB 适配版 ResNet-18，比较标准卷积、深度可分离卷积与 SE 注意力。研究分类精度、复杂度、硬件性能及五种合成退化下的鲁棒性。与实验三的 YOLO 目标检测、超分辨率复现不同，本项目为裁剪交通标志的多类别图像分类。

**已完成：四组统一使用种子11、22、33，每次60轮，共12次训练、720轮和192项评价。** 逐样本指标独立复核与均值／样本标准差复核均为PASS。运行 `python status.py` 查看本地状态。

下表为全部三个种子的均值±样本标准差（ddof=1）；GPU延迟沿用固定种子11的三轮计时结果，不是跨训练种子的均值。原单种子报告在本地归档于 `archives/single_seed_11/report/`，原评价协议为 `protocols/protocol_v2_single_seed.json`。

| 组别 | 测试 Accuracy % | Macro-F1 % | 参数量 | Conv/Linear MACs M | GPU中位延迟 ms |
|---|---:|---:|---:|---:|---:|
| A ResNet-18 | 98.56 ± 0.02 | 97.35 ± 0.28 | 11,190,891 | 1249.712 | 10.465 |
| B DW | 98.42 ± 0.15 | 97.31 ± 0.31 | 1,456,555 | 164.371 | 3.071 |
| C SE | 98.73 ± 0.07 | 97.69 ± 0.33 | 11,279,971 | 1249.799 | 12.176 |
| D DW+SE | 98.04 ± 0.42 | 96.88 ± 0.97 | 1,545,635 | 164.458 | 5.309 |

B相对A减少86.98%参数，平均准确率降低0.135个百分点；C在三个种子上均比A准确率高，平均提高0.172个百分点；D相对B的平均变化为−0.383个百分点且波动较大。三个种子仅作描述性比较，不宣称统计显著。完整报告见 [报告正文](report/大作业报告.md)，原始汇总见 [summary.json](report/tables/summary.json)。

仓库包含12次运行的JSON和日志，以及报告、表图和审计记录。权重、192份逐样本预测和源码快照保存在本地完整证据包；训练恢复检查点仍在本机 `runs/`。Git不直接存放这些大文件，数据集不再分发。

## 环境

本机训练解释器为 `D:\Anaconda3\python.exe`，PyTorch `2.7.1+cu118`，NVIDIA RTX 4060 Laptop GPU。不要直接假定 PATH 中的 `python` 支持 CUDA。

推荐在独立环境安装 Python 3.11 或 3.12，然后安装兼容 GPU 驱动的 CUDA 版 PyTorch：

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())"
```

项目代码不依赖 torchvision 的二进制扩展，也不下载预训练模型。数据 ZIP 的官方地址和 MD5 参照 torchvision 的 GTSRB 下载定义。每次运行的确切依赖版本保存在 `environment.json`。

下列命令假定已进入本项目目录，`python` 指向支持 CUDA 的解释器。本机 PowerShell 可将命令开头的 `python` 替换为 `& 'D:\Anaconda3\python.exe'`。

## 数据下载与分组

```powershell
python prepare_data.py --data-root data/gtsrb
# 原托管地址无法连接时，使用公开镜像，仍校验原 ZIP 的官方 MD5：
python prepare_data.py --data-root data/gtsrb --mirror
python verify.py --data-root data/gtsrb
python inspect_data.py --data-root data/gtsrb
```

数据版本明确使用 **`GTSRB-Training_fixed.zip` 的 26,640 张训练图片**和 `GTSRB_Final_Test_Images.zip` 的 12,630 张测试图片，不冒充常见的 39,209 张最终训练版本。训练集再按类别分层、以完整 `(class_id, track_id)` 分组，约 80% 训练、20% 验证；具体计数由 `report/data/dataset_summary.json` 给出。所有模型和种子共用划分种子 2026。`verify.py` 检查组无交集和 43 类覆盖。

镜像模式下训练/测试图片 ZIP 与 torchvision 期望的 MD5 完全一致。测试标签取自公开 GTSRB 镜像 `bazyl/GTSRB/Test.csv`，校验 SHA256；仅将 `Test/NNNNN.png` 路径转换为原始 PPM 的 `NNNNN.ppm`，其余列不变。逐张核对尺寸与 ROI。全部下载 URL、校验值及转换说明保存于 `download_manifest.json`。不执行镜像中的任何脚本。

保留原图全部内容，不使用 ROI 裁剪；按边缘像素补成正方形，双线性缩放至 48×48，固定均值/标准差 `(0.5,0.5,0.5)` 归一化。增强为每图独立 ±10° 旋转、±5% 平移、0.95–1.05 缩放及 0.8–1.2 亮度变化，不翻转。GPU 批量增强使用 `affine_grid/grid_sample`，双线性采样和 border padding。

首次会生成 `cache48/`：训练/测试 uint8 图像缓存、图片与标签元数据、训练/验证索引、逐类 track 计数及内容校验值。正式训练不需要再次解码原图片。数据许可与使用条件以 [GTSRB 项目页面](https://benchmark.ini.rub.de/gtsrb_dataset.html) 为准；数据不随代码仓库再分发。

## 四组模型

| 组别 | 残差主分支 3×3 卷积 | SE |
|---|---|---|
| A | 标准卷积 | 无 |
| B | depthwise 3×3 + pointwise 1×1 | 无 |
| C | 标准卷积 | 每个 BasicBlock 一个 |
| D | depthwise 3×3 + pointwise 1×1 | 每个 BasicBlock 一个 |

入口统一为 3×3 stride=1、无最大池化。四阶段通道 `[64,128,256,512]`，每阶段两个块。DW/PW 之间不额外增加 BN/ReLU。SE 位于主分支末尾 BN 后、残差相加前，r=16。入口和捷径投影不替换。可选 E 仅替换 stage3/4，未进入四组正式实验清单。

## 先试跑 再冻结 再正式运行

仓库已发布的 `runs/` 是原始指标与日志，不含权重。若要从头复现，请在**全新克隆的副本**中先把已有指标及报告移入归档目录，再按下列步骤下载数据、生成数据统计和训练；不要在当前实测工作目录执行这两条移动命令：

```powershell
New-Item -ItemType Directory -Path published_results
Move-Item -LiteralPath runs,report -Destination published_results
python prepare_data.py --data-root data/gtsrb --mirror
python inspect_data.py --data-root data/gtsrb
```

```powershell
# 独立试跑目录，不纳入正式报告；不会查看测试准确率
python train.py --config configs/A.yaml --seed 11 --epochs 8 --data-root data/gtsrb --output pilot/A11

# 检查验证曲线、资源占用及验证集退化样例后冻结协议
python freeze_protocol.py --data-root data/gtsrb

# 单 GPU 串行完成12次训练，已有完成标记且校验通过的运行自动跳过
python run_experiments.py --data-root data/gtsrb
```

正式训练：60 epoch、batch=128、SGD(momentum=0.9, weight_decay=0.0005)、初始学习率0.05、5轮线性预热后余弦衰减至0.0001、统一 AMP FP16。训练种子 `[11,22,33]`，四组共12次训练，其中8次为本次补充。正式配置见 `protocol_frozen.json` 和 run 配置。

根据用户的计算时间要求，在查看测试结果之前，将原计划的每组3个种子减少为每组1个固定种子11。原协议完整保存在 `protocols/protocol_v1_three_seeds.json`，当前协议记录修订原因、时间和原协议哈希，`validation/single_seed_amendment.json` 记录已保存的 A/B 检查点哈希。没有改变结构、数据划分、60轮训练预算、优化器、学习率计划或退化参数；已完成的A与进行中的B无需重训。

种子11测试结果公布后，用户要求四组统一补充22、33。此次属于查看结果后的重复实验扩展，记录于 `validation/three_seed_extension.json`，不宣称全部重复在首次测试前预注册。`extend_seeds.py`归档原协议、报告和所有种子11文件哈希；种子11评价沿用原协议，不重写原指标或预测。`experiment_protocol.py`要求继承协议的数据、结构、训练、源码、选模和退化字段完全一致。原种子11权重和恢复状态均保留。

检查点选择规则为验证 Macro-F1 最大，其次验证损失更低，再平局取更早 epoch。验证和正式评价均使用 FP32。训练计时包含训练与验证，不把它当成纯网络性能测试。

单次运行和恢复：

```powershell
python train.py --config configs/B.yaml --seed 11 --data-root data/gtsrb --resume
python eval.py --run runs/B/22 --data-root data/gtsrb --protocol protocol_frozen.json
# 已有种子11结果必须继续使用归档的原评价协议
python eval.py --run runs/B/11 --data-root data/gtsrb --protocol protocols/protocol_v2_single_seed.json
python profile_model.py --variant B --checkpoint runs/B/11/best.ckpt --output report/profiles/B.json
python make_tables.py
```

`--resume` 从每个 epoch 后的 `last.ckpt` 恢复模型、优化器和 AMP scaler；数据随机流由种子/epoch 重建。恢复目录的配置和数据清单必须一致。`run_experiments.py` 可重跑，跳过已完成训练和已验证评价条件。不得为改变 epoch 数覆盖已有实验目录。

## 评价规则

训练只访问训练/验证数据。测试评价要求冻结文件、模型/数据/训练/评价源码哈希、run 配置与数据清单匹配。冻结后一次最终评价阶段包含全部模型、种子和预先定义的退化条件；不根据测试结果调整结构、训练配置或退化强度。

五种退化各三个强度，作用于统一预处理后的 48×48 RGB 图片：

| 退化 | 三档参数 |
|---|---|
| 亮度降低 | 倍率0.7、0.4、0.2 |
| 水平运动模糊 | 核长3、5、9，边缘补齐 |
| 高斯噪声 | `[0,1]` 像素标准差0.03、0.06、0.12 |
| JPEG压缩 | quality 50、25、10，subsampling=2 |
| 降低分辨率 | 降至32、24、12后双线性回采样到48 |

随机噪声由样本路径、类型、强度与协议版本哈希决定，与模型/训练种子无关。各模型使用相同退化输入，缓存与预测保存校验值。逐条件报告 Accuracy、Macro-F1 和相对干净输入下降的百分点。合成退化不等同于真实夜间、雨雾或车辆部署测试。

复杂度只统计 Conv2d/Linear MACs，1 MAC 表示一次乘加，不计 BN、激活、池化和逐元素操作。GPU 性能使用 eval/inference_mode、FP32、关闭 TF32、channels-last、GPU 常驻输入；预热50次、测量200次、重复3轮。batch=1逐次同步 CUDA Event 测中位数/P95；batch=32连续执行测吞吐，仅边界同步。不包含磁盘解码、预处理和传输。记录 GPU 型号、驱动及可获得的功耗信息；显示桌面工作负载可能带来额外波动。

## 运行目录与数字来源

```text
runs/{A,B,C,D}/{11,22,33}/
  config.json                 实际运行配置
  environment.json            版本、设备、源码清单、git状态
  source_snapshot.zip         含未提交代码的白名单源码快照
  data_reference.json         数据/划分清单的校验值
  complexity.json             实际训练模型的参数和MACs
  log.csv                     每epoch训练与验证指标
  best.ckpt                   按固定验证规则选择的检查点
  last.ckpt                   断点恢复状态
  complete.json               训练完成标记与权重校验值
  test_metrics.json           干净测试结果
  robustness_metrics.json     全部16种输入条件结果
  predictions/*.npz           样本ID、标签、预测和43类概率
report/
  data/                       数据统计与验证集样例
  profiles/                   原始延迟测量及汇总
  tables/                     逐seed、汇总、消融和退化表
  figures/                    由实测文件生成的PNG/SVG
```

表格保留未舍入数值，报告展示时统一舍入。三种子Accuracy和Macro-F1按算术平均与样本标准差（ddof=1）汇总，逐种子原值另存；学习曲线和退化曲线绘制均值±1个样本标准差。消融差值先按相同种子配对，再计算差值的均值与样本标准差；不根据测试准确率挑选最佳种子。标准差不是置信区间，三个种子也不自动证明显著性。`summary.json`记录输入文件及SHA256，`audit_summary.py`用Python标准库独立复核汇总。GPU性能仍沿用固定种子11的三轮计时，这与训练种子重复不同。

队列结束后自动复核预测、生成验证样例与报告。报告及表图来自完整实测文件。运行 `python status.py` 可同时查看训练轮数、测试条件数、GPU性能测量和报告状态。

复核与重新导出：

```powershell
python audit_results.py
python make_tables.py
python audit_summary.py
python make_examples.py --data-root data/gtsrb
python build_report.py
# 可选：导出可编辑Word，需要额外安装python-docx
python -m pip install python-docx
python export_docx.py
```

`audit_results.py`在扩展完成后重新读取192份预测，使用sklearn核对Accuracy、Macro-F1、混淆矩阵、样本对齐、共用退化输入哈希和60轮日志的检查点选择规则。`report/final_audit.json`保存逐样本复核结果，`report/summary_audit.json`保存均值与标准差复核结果。最佳epoch在训练日志中从0计数，报告表格转为从1计数。

独立检查均值与标准差不需要GPU、数据集、权重或预测数组：安装NumPy与PyYAML后，在项目根目录运行 `python audit_summary.py`。完整逐样本复核另需完整证据包中的192份NPZ与12个best.ckpt。重新训练时请使用新目录（或全新克隆后先移出已发布的runs指标目录），避免与已发布记录混用；不可删除原始证据。

PDF导出使用 `python export_pdf.py`，额外依赖reportlab；默认使用Windows宋体和黑体，其他平台通过 `--font` 与 `--bold-font` 传入支持中文的TrueType字体。报告导出依赖可通过 `pip install -r requirements-report.txt` 安装。[已逐页核验的PDF](report/大作业报告_三种子.pdf)与Markdown的数字一致。本机另存了三种子Word文件，但当前会话缺少LibreOffice且无法连接Word，尚未完成Word逐页排版核验；正式阅读以PDF或Markdown为准。

打包使用显式文件白名单并检查当前协议的两份PASS审计；每个包含文件级SHA256清单：

```powershell
python package_results.py --output ../traffic_sign_cnn_代码报告与实测结果_三种子.zip
python package_results.py --include-weights --output ../traffic_sign_cnn_权重预测与完整证据_三种子.zip
```

第一个包提供代码、运行说明、报告图表和12次原始指标；第二个额外提供12个最佳检查点、192份逐样本预测及源码快照。恢复训练使用的last.ckpt保留在本机，不纳入交付包。

### 暂停与继续

在项目目录执行 `python control_training.py pause` 可停止本项目队列和当前训练，释放GPU；已保存检查点保留，尚未保存的当前epoch会在恢复时重跑。执行 `python control_training.py resume` 从最近完整检查点恢复，并跳过已完成实验。控制脚本会核验进程工作目录和脚本名，拒绝重复启动。使用 `python control_training.py status` 查看是否存在本项目进程。不要删除或修改 `last.ckpt`、配置及冻结协议。

## 场景演示与命令行预测

```powershell
python infer.py --checkpoint runs/D/11/best.ckpt --image YOUR_SIGN.png --output prediction.json --cam gradcam.png
python -m streamlit run app.py
```

演示默认在 CPU 运行，支持上传裁剪后的标志、Top-3 概率、固定退化与 stage3 的12×12 Grad-CAM。概率未校准，不支持未知类别判断，也不自动在道路照片中定位标志。只加载可信的本项目 `.ckpt` 文件，PyTorch pickle 检查点不可视为安全的外来文档。

## 改进实验与最快演示

原A-D三种子实验与已发布报告保持不变。新增逐类别、配对错误与训练波动分析见`report/deep_analysis/`；真实GTSDB道路场景见`report/external/`。外部子集按推理前冻结的规则选择100张有标注道路图像中的全部163个标志框，不用于训练或调参；无法证明与GTSRB的实体标志完全独立，且仅评价已知标注框内分类。E为stage3/4局部DW的事后探索实验，协议独立保存在`protocols/supplement_E.json`。全部三个种子训练、48个条件评估与独立审计完成前，不列入E结论。

E进度：`python status.py`；暂停、恢复仍使用`python control_training.py pause`和`python control_training.py resume`。补充队列完成后自动运行`summarize_supplement.py`、外部评估、独立审计、`build_improved_report.py`及代码结果打包。PDF导出需reportlab；也可通过环境变量`GTSRB_REPORT_PYTHON`指定安装了reportlab的解释器。报告标题含“改进版”，未完成E时会明确标注进度，原报告不会被覆盖。

在Python 3.12环境安装已实测版本，然后下载Release权重，无需先下载数据集或重新训练：

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-tested.txt
python download_weights.py
python infer.py --checkpoint runs/C/11/best.ckpt --image examples/road_sign.png --output prediction.json --cam gradcam.png
python -m streamlit run app.py
```

训练应选择CUDA解释器，本机为`D:\Anaconda3\python.exe`。推理默认CPU，可直接运行。上述道路标志示例的真实类别为11，来源、裁剪坐标和选择规则在`examples/road_sign_source.json`；演示样本属于定性案例，整体准确率以完整结果为准。

[原A-D三种子权重Release](https://github.com/CYBEZM/gtsrb-resnet18-dw-se-coursework/releases/tag/v1.0.0-three-seed)提供种子11的四个检查点及全部十二个检查点。`python download_weights.py --all-seeds`下载全部种子。脚本依据仓库内`weights_manifest.json`验证压缩包和每个检查点SHA256；支持`--archive YOUR_DOWNLOADED_ZIP`离线安装，遇到已存在但内容不同的检查点会停止，不覆盖它。E权重不属于该原始实验Release。

新增分析可分别复核与生成：

```powershell
python analyze_results.py
python audit_deep_analysis.py
python prepare_external.py --count 100 --output data/gtsdb_external
python evaluate_external.py --external-root data/gtsdb_external --gtsrb-root data/gtsrb
python audit_external.py --external-root data/gtsdb_external
python make_external_scenes.py --external-root data/gtsdb_external
python build_improved_report.py
```

外部数据下载仅选择固定镜像版本中的指定ZIP成员，并验证HTTP范围、ZIP CRC32和逐图SHA256；未下载完整ZIP，不将发布方的整包SHA256声称为本地已验证。模型评估需要全部种子的权重。逐类别完整复核还需原完整证据包的NPZ，下载权重本身不包含这些预测数组。改进版PDF可直接阅读；Word仍须在具有可用Word/LibreOffice的环境中完成排版核验。

## 方法参考

- He et al. *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. 2017. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
- Hu et al. *Squeeze-and-Excitation Networks*. CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- Stallkamp et al. *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 2012. [DOI:10.1016/j.neunet.2012.02.016](https://doi.org/10.1016/j.neunet.2012.02.016)

仓库不包含第三方数据或预训练权重。结构实现由本项目编写；参考文献用于说明方法来源，不将已有注意力或可分离卷积作为原创算法宣称。
