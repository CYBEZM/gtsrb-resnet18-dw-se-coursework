# 交通标志 CNN 结构改进与鲁棒性实验

从随机初始化训练 GTSRB 适配版 ResNet-18，比较标准卷积、深度可分离卷积与 SE 注意力。研究分类精度、复杂度、硬件性能及五种合成退化下的鲁棒性。与实验三的 YOLO 目标检测、超分辨率复现不同，本项目为裁剪交通标志的多类别图像分类。

**结果状态：请查看 `runs/queue_status.json` 和每次运行的 `complete.json`。没有生成的指标不代表已完成。** `make_tables.py` 只有在四组 × 三个种子全部完成干净及退化评价后才会生成正式汇总；不填充预估准确率。

## 环境

本机训练解释器为 `D:\Anaconda3\python.exe`，PyTorch `2.7.1+cu118`，NVIDIA RTX 4060 Laptop GPU。不要直接假定 PATH 中的 `python` 支持 CUDA。

推荐在独立环境安装 Python 3.11 或 3.12，然后安装兼容 GPU 驱动的 CUDA 版 PyTorch：

```powershell
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())"
```

项目代码不依赖 torchvision 的二进制扩展，也不下载预训练模型。数据 ZIP 的官方地址和 MD5 参照 torchvision 2.7.1 对应发行版本的 GTSRB 下载定义。每次运行的确切依赖版本保存在 `environment.json`。

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

```powershell
# 独立试跑目录，不纳入正式报告；不会查看测试准确率
python train.py --config configs/A.yaml --seed 11 --epochs 8 --data-root data/gtsrb --output pilot/A11

# 检查验证曲线、资源占用及验证集退化样例后冻结协议
python freeze_protocol.py --data-root data/gtsrb

# 单 GPU 串行完成 12 次训练、所有评价、硬件测试和表图汇总
python run_experiments.py --data-root data/gtsrb
```

正式训练：60 epoch、batch=128、SGD(momentum=0.9, weight_decay=0.0005)、初始学习率0.05、5轮线性预热后余弦衰减至0.0001、统一 AMP FP16。训练种子 `[11,22,33]`。这些是冻结前的候选配置；正式值以 `protocol_frozen.json` 和 run 配置为准。

检查点选择规则为验证 Macro-F1 最大，其次验证损失更低，再平局取更早 epoch。验证和正式评价均使用 FP32。训练计时包含训练与验证，不把它当成纯网络性能测试。

单次运行和恢复：

```powershell
python train.py --config configs/B.yaml --seed 22 --data-root data/gtsrb --resume
python eval.py --run runs/B/22 --data-root data/gtsrb --protocol protocol_frozen.json
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

表格保留未舍入数值；报告展示时统一舍入，均值±样本标准差(ddof=1)。`summary.json` 记录输入文件及 SHA256；三次种子的标准差不当作显著性检验。配对种子差值和注意力交互效应单独导出。

## 场景演示与命令行预测

```powershell
python infer.py --checkpoint runs/D/11/best.ckpt --image YOUR_SIGN.png --output prediction.json --cam gradcam.png
python -m streamlit run app.py
```

演示默认在 CPU 运行，支持上传裁剪后的标志、Top-3 概率、固定退化与 stage3 的12×12 Grad-CAM。概率未校准，不支持未知类别判断，也不自动在道路照片中定位标志。只加载可信的本项目 `.ckpt` 文件，PyTorch pickle 检查点不可视为安全的外来文档。

## 方法参考

- He et al. *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Howard et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. 2017. [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
- Hu et al. *Squeeze-and-Excitation Networks*. CVPR 2018. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- Stallkamp et al. *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*. Neural Networks, 2012. [DOI:10.1016/j.neunet.2012.02.016](https://doi.org/10.1016/j.neunet.2012.02.016)

仓库不包含第三方数据或预训练权重。结构实现由本项目编写；参考文献用于说明方法来源，不将已有注意力或可分离卷积作为原创算法宣称。
