# 行政许可BERTopic与BERT分类管线

本项目面向中国17个国家级新区的行政许可事项，提供一套可复现的主题分析与分类实验代码。核心目标是用领域化的分词词典和 BERT/BERTopic 结合的方法，对“省级行政许可是否下放给新区”这一问题进行可解释的主题归类与监督评估。仓库内容遵循顶级公共管理期刊的数据分析标准，便于论文撰写与复现。

## 数据要求

- **格式**：UTF-8 编码的制表符分隔（TSV）文本，列名包含：`序号`、`省份`、`新区`、`省级主管单位事项`、`名称`、`是否下放`、`下放方式`、`首次下放时间`。
- **示例路径**：`/Users/weihanlu/Desktop/bert/your_data.tsv`。
- **示例数据**：仓库内的 `data/sample_admin_items.tsv` 提供了 5 条样例记录，便于快速验证管线。

## test1-test5 方案回顾

| 方案 | 模型/特征 | 主题/分类效果概述 | 适用场景 | 风险与改进 |
| --- | --- | --- | --- | --- |
| test1 | 默认 BERTopic + 字符级切分 | 主题可分但含大量噪声，难以定位“下放/委托”语义 | 快速原型 | 缺乏领域词，需自定义词典与停用词 |
| test2 | BERTopic + 通用中文停用词 | 噪声减轻但对“生态环境/规划许可”等专业短语仍有分裂 | 小规模预研 | 需引入行政许可专用词表、双字/三字短语 |
| test3 | SciBERT 风格（强调科技/审批名词）+ 双字短语 | 专业名词聚合改善，能区分“环境影响评价/港口经营”主题 | 专业事项聚类 | 对委托/下放措辞敏感度不足，需添加管理动词词表 |
| test4 | FinBERT 风格（强调监管/责任词）+ 责任/权责短语 | 能突出“委托/监管责任”主题，识别法律责任表达 | 权责划分分析 | 对技术性许可名称覆盖不足，需与专业名词结合 |
| test5 | **混合词典 + 自适应停用词 + Sentence-BERT 嵌入**（推荐） | 同时保留专业名词与权责动词，主题可解释性与下放/委托判别提升 | 论文主实验 | 需调参 `min_topic_size`、`nr_topics`，并根据样本量调整 `min_df` |

## 新的改进方案（推荐实施）

1. **领域词典构建**：
   - 专业名词：生态环境、建设项目、规划许可、港口经营、食品经营、外商投资等。
   - 权责/流程动词：下放、委托、审批、核准、备案、监督、承担、实施。
   - 机构与地域：省级主管部门、行政机关、新区管委会、两江新区、天府新区等。
   - 停用词：常见虚词（的、了、其中）与无判别力的时间表达。
2. **主题建模**：使用 `BERTopic` + `SentenceTransformer("shibing624/text2vec-base-chinese")`，向量化时采用自定义分词和双字/三字短语，聚焦行政许可核心语义。
3. **下放判别分类**：用 `bert-base-chinese` 监督分类 `是否下放`，利用 HF `Trainer` 进行 3-5 epoch 轻量训练，保存最佳模型并输出分类报告、混淆矩阵和训练曲线。
4. **可视化与报告**：输出主题词表、主题-文档分布、混淆矩阵、标签分布和训练损失曲线，满足论文复现要求。

## 运行步骤

1. 安装依赖（建议使用 Python 3.10+）：
   ```bash
   pip install -U pip
   pip install pandas scikit-learn jieba bertopic sentence-transformers transformers datasets matplotlib seaborn
   ```
2. 运行核心脚本（默认用示例数据，仅做主题建模，避免长时间训练）：
   ```bash
   python src/admin_license_analysis.py \
     --data-path data/sample_admin_items.tsv \
     --output-dir artifacts \
     --embed-model shibing624/text2vec-base-chinese \
     --train-classifier false
   ```
3. 若需训练 `是否下放` 分类器（需 GPU/较长时间）：
   ```bash
   python src/admin_license_analysis.py \
     --data-path /Users/weihanlu/Desktop/bert/your_data.tsv \
     --output-dir artifacts \
     --embed-model shibing624/text2vec-base-chinese \
     --bert-model bert-base-chinese \
     --train-classifier true \
     --num-epochs 4
   ```
4. 主要输出（保存在 `--output-dir`）：
   - `topic_info.csv`：主题编号、频次、Top10 关键词。
   - `document_topics.csv`：每条记录的主题分配与概率。
   - `classification_report.txt`：`是否下放` 的精确率/召回率/F1。
   - `confusion_matrix.png`、`label_distribution.png`、`training_loss.png`：可视化结果。
   - `best_classifier/`：保存的下放判别 BERT 模型。

## 目录结构

```
.
├── README.md
├── src
│   └── admin_license_analysis.py   # 主题建模与分类主脚本
├── data
│   └── sample_admin_items.tsv      # 示例数据（TSV）
└── artifacts/                      # 运行后生成的模型与可视化
```

## 参数说明

- `--data-path`：输入数据路径（TSV）。
- `--output-dir`：保存模型与图表的目录。
- `--embed-model`：BERTopic 使用的句向量模型，默认 `shibing624/text2vec-base-chinese`。
- `--bert-model`：监督分类使用的预训练模型，默认 `bert-base-chinese`。
- `--train-classifier`：是否训练 `是否下放` 分类器（true/false）。
- `--num-epochs`：分类器训练轮数，默认 3。
- `--max-length`：BERT 分词最大长度，默认 128。
- `--test-size`：划分测试集比例，默认 0.2。

## 调参建议与潜在问题

- **主题颗粒度**：若主题过碎，可提高 `min_topic_size` 或设置 `nr_topics`；若主题过粗，可降低 `min_df` 或增大 `n_gram` 范围。
- **分词准确性**：确保将行政许可短语加入 `CUSTOM_DOMAIN_TERMS`，并在训练前扩充词典，减少被切断的专业名词。
- **类别不平衡**：`是否下放` 若严重失衡，可在 `TrainingArguments` 中设置 `class_weight`（需自定义 loss）或采用过采样。
- **超参数参考**：
  - `learning_rate`: 2e-5 ~ 3e-5
  - `per_device_train_batch_size`: 8 ~ 16
  - `num_train_epochs`: 3 ~ 5
  - `warmup_ratio`: 0.1
- **论文复现**：固定 `seed=42`，保存 `log_history`，并在报告中描述词典来源与停用词表。

## 背景说明

- **模型选择**：SciBERT 擅长科技/审批名词，FinBERT 擅长监管责任词。本方案融合二者思路，构建“行政许可专用词典”，兼顾专业名词与权责动词，提升主题可解释性与“下放/委托”判别力。
- **应用场景**：
  - 比较不同省份/新区的下放方式与时间节点
  - 分析不同主管部门事项在新区的主题分布
  - 构建论文附录的主题词表与模型评估指标

