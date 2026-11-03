# 实验沿革与证据口径

本研究沿“经典CNN实现 → DW与SE因素对照 → 替换位置分析 → 场景应用”推进。当前报告按问题组织，原始执行记录与既有发布保留，整理正文不会改变实验发生的先后关系。

## 协议沿革

1. 初始A-D设计使用三个种子，协议为`protocols/protocol_v1_three_seeds.json`。
2. 为适应计算时间，在首次测试结果公布前改成每组种子11，网络、划分、60轮预算、选模和退化规则未改变。评价协议为`protocols/protocol_v2_single_seed.json`；修订记录为`validation/single_seed_amendment.json`。
3. 看到种子11结果后，四组统一增加22、33。当前`protocol_frozen.json`记录继承关系，`validation/three_seed_extension.json`记录扩展。种子11指标仍引用其原评价协议，原结果、权重与预测不重写；保存清单为`validation/preserved_seed11_files.json`。
4. 分析A-D结果后提出E：只替换stage3/4，不加SE。配置与三个种子在执行前写入`protocols/supplement_E.json`。这是事后提出的探索问题，不是对首次假设的独立确认。
5. 逐类别错误、训练波动和案例可视化均使用已保存结果；未据此调参或重新选择检查点。道路输入子集在该子集推理前固定，并保留全部符合规则的裁剪。

报告只在实验设置中交代这段沿革。E及这些分析已按内容融入正文，不再单列“补充实验”章节。磁盘目录中的`supplement`等名称是原始证据定位，不代表另一份实验报告。

## 设备测量

早期A-D计时保存在`report/profiles/`，E初测在`report/supplement/profile_E.json`。不同时间的测量不能直接用于五组排序。

当前报告、README和新图统一引用`report/supplement/contemporary_profiles/{A-E}.json`，五组在同一时段按A-E串行测量，固定种子11、FP32、相同预热及采样规则。`session.json`保留会话记录。历史测量未删除，当前结果表不混用两套口径。同一时段也未随机交替模型顺序，系统负载和温度影响尚不能排除。

## 图像与案例选择

- 数据分布和预处理图使用固定数据划分。四张Grad-CAM输入沿用正式测试前指定的验证样本，A-E逐行比较相同图像；每个模型解释自身预测类别。
- 配对错误图固定种子11，对“C修正A / C新增错误 / E修正B / E新增错误”分别取满足条件的首个样本ID。
- 退化同图对照固定种子11，取五组在干净输入上都正确、至少一种强度3退化造成模型分歧的首个样本ID。
- 道路案例固定C组种子11，按样本ID分别取前四个正确和错误裁剪；原场景图取首个错误裁剪所在道路图像。框来自人工标注。

事后案例用于解释预测变化，不用于估计总体成功比例。全部样本用于定量评价。新增案例的样本ID、选择规则、可选样本数、预测概率、输入及权重哈希见`report/integrated/comparison_sources.json`。

## 证据边界与历史文件

GTSDB仅评价标注框内分类；精确像素去重不足以证明实体标志独立。来源、许可、镜像固定版本与裁剪规则见`report/external/source_manifest.json`、`input_protocol.json`。GTSRB使用Training_fixed版本，本文不将其与39,209张最终训练版本混同。

旧的单种子、三种子及拼接式改进报告作为历史产物保留。当前入口是README所链接的内容整合稿v2；这仍是供用户修改内容和格式的审阅版本。历史Release附带的报告不自动替换成当前稿，大文件证据与权重继续沿用其校验清单。

当前正文保留6幅核心图和6个表。两张同图退化Grad-CAM使用原先固定验证样本的第3、4张，统一施加分辨率第3档，不按正确与否选图；道路复合图按样本ID各取前两个正确与错误裁剪。选择与实际CPU预测见`report/integrated/compact_sources.json`。旧版的完整案例图作为历史材料保留。分层参数、感受野及逐run验证回落由`make_compact_evidence.py`计算并保存为`component_analysis.json`，不涉及重新训练。
