"""
行政许可主题分析与下放判别实验脚本。

功能：
- 环境检查与依赖版本打印
- 加载新区行政许可下放数据集（TSV），构造文本字段
- 基于自定义领域词典的中文分词与向量化
- 使用 BERTopic 进行主题建模
- 可选：使用 BERT 分类 `是否下放`，包含训练、评估、可视化
- 保存主题与分类结果，便于论文复现

说明：
- 默认使用示例数据，仅运行 BERTopic 以避免长时间训练。
- 训练分类器需要 GPU 以获得合理速度；如使用 CPU，请适当减少 epoch 与 batch size。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
from bertopic import BERTopic
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DOMAIN_STOP_WORDS: List[str] = [
    "的",
    "了",
    "其中",
    "有关",
    "以上",
    "以下",
    "以及",
    "按",
    "根据",
    "通过",
    "实施",
    "管理办法",
    "单位",
    "机关",
    "申请",
    "办理",
    "材料",
    "规定",
]

CUSTOM_DOMAIN_TERMS: List[str] = [
    "行政许可",
    "下放",
    "委托",
    "审批",
    "核准",
    "备案",
    "监管",
    "承担",
    "港口经营",
    "规划许可",
    "环境影响评价",
    "食品经营",
    "外商投资",
    "建设项目",
    "自然资源",
    "生态环境",
    "市场监管",
    "交通运输",
    "国土空间",
    "新区管委会",
    "省级主管部门",
    "责任",
    "权限",
    "监督",
    "法律责任",
]


@dataclass
class RunConfig:
    data_path: str
    output_dir: str
    embed_model: str = "shibing624/text2vec-base-chinese"
    bert_model: str = "bert-base-chinese"
    max_length: int = 128
    test_size: float = 0.2
    num_epochs: int = 3
    train_classifier: bool = False


def environment_report() -> Dict[str, str]:
    report = {
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pandas": pd.__version__,
        "bertopic": BERTopic.__version__ if hasattr(BERTopic, "__version__") else "unknown",
    }
    print("环境检查:")
    for k, v in report.items():
        print(f"- {k}: {v}")
    return report


def load_admin_dataset(path: str, sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    expected_cols = ["序号", "省份", "新区", "省级主管单位事项", "名称", "是否下放", "下放方式", "首次下放时间"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列: {missing}")
    df["文本"] = (
        df["省份"].astype(str)
        + " "
        + df["新区"].astype(str)
        + " "
        + df["省级主管单位事项"].astype(str)
        + " "
        + df["名称"].astype(str)
        + " "
        + df["下放方式"].astype(str)
    )
    return df


def extend_jieba_dict(custom_terms: List[str]) -> None:
    for term in custom_terms:
        jieba.add_word(term)


def domain_tokenize(text: str) -> List[str]:
    tokens = []
    for tok in jieba.cut(text):
        tok = tok.strip()
        if not tok or tok.isdigit():
            continue
        if any(stop in tok for stop in DOMAIN_STOP_WORDS):
            continue
        tokens.append(tok)
    return tokens


def build_vectorizer() -> CountVectorizer:
    extend_jieba_dict(CUSTOM_DOMAIN_TERMS)
    return CountVectorizer(
        tokenizer=domain_tokenize,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words=DOMAIN_STOP_WORDS,
    )


def train_topic_model(texts: List[str], embed_model: str, output_dir: str) -> BERTopic:
    embedding_model = SentenceTransformer(embed_model)
    vectorizer = build_vectorizer()
    topic_model = BERTopic(
        language="chinese",
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=5,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts)
    topic_info = topic_model.get_topic_info()
    doc_info = topic_model.get_document_info(texts)
    os.makedirs(output_dir, exist_ok=True)
    topic_info.to_csv(os.path.join(output_dir, "topic_info.csv"), index=False)
    doc_info.to_csv(os.path.join(output_dir, "document_topics.csv"), index=False)
    print("已保存主题信息到 output_dir")
    return topic_model


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int]]:
    unique = sorted(labels.unique())
    mapping = {label: idx for idx, label in enumerate(unique)}
    encoded = labels.map(mapping).to_numpy()
    return encoded, mapping


def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, List[List[int]]]:
    return tokenizer(examples["文本"], truncation=True, max_length=max_length)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    return {"accuracy": acc, "macro_f1": macro_f1}


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], output_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_label_distribution(labels: np.ndarray, output_path: str) -> None:
    plt.figure(figsize=(5, 4))
    unique, counts = np.unique(labels, return_counts=True)
    sns.barplot(x=unique, y=counts, palette="crest")
    plt.xlabel("标签编号")
    plt.ylabel("数量")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_loss(log_history: List[Dict], output_path: str) -> None:
    steps = [entry["step"] for entry in log_history if "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    if not steps:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(steps, losses, marker="o")
    plt.xlabel("步骤")
    plt.ylabel("训练损失")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_classifier(
    df: pd.DataFrame, config: RunConfig, label_mapping: Optional[Dict[str, int]] = None
) -> Tuple[Trainer, Dict[str, int]]:
    labels, mapping = encode_labels(df["是否下放"]) if label_mapping is None else (
        df["是否下放"].map(label_mapping).to_numpy(),
        label_mapping,
    )
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["文本"], labels, test_size=config.test_size, random_state=42, stratify=labels
    )
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model)
    train_dataset = Dataset.from_dict({"文本": list(train_texts), "label": train_labels})
    test_dataset = Dataset.from_dict({"文本": list(test_texts), "label": test_labels})
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer, config.max_length), batched=True)
    tokenized_test = test_dataset.map(lambda x: tokenize_function(x, tokenizer, config.max_length), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.bert_model, num_labels=len(mapping)
    )

    args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=config.num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        save_total_limit=2,
        seed=42,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - start
    print(f"训练完成，用时 {elapsed:.2f} 秒")

    eval_metrics = trainer.evaluate()
    print(f"评估结果: {json.dumps(eval_metrics, ensure_ascii=False, indent=2)}")

    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    report_text = classification_report(test_labels, y_pred, target_names=list(mapping.keys()), zero_division=0)
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    plot_confusion(test_labels, y_pred, list(mapping.keys()), os.path.join(config.output_dir, "confusion_matrix.png"))
    plot_label_distribution(labels, os.path.join(config.output_dir, "label_distribution.png"))
    plot_training_loss(trainer.state.log_history, os.path.join(config.output_dir, "training_loss.png"))

    best_dir = os.path.join(config.output_dir, "best_classifier")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    return trainer, mapping


def analyze_new_items(trainer: Trainer, tokenizer: AutoTokenizer, texts: List[str], label_mapping: Dict[str, int]) -> List[str]:
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    outputs = trainer.model(**encodings)
    preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
    inv_mapping = {v: k for k, v in label_mapping.items()}
    return [inv_mapping[p] for p in preds]


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="行政许可主题建模与下放判别")
    parser.add_argument("--data-path", required=True, help="TSV 数据路径")
    parser.add_argument("--output-dir", default="artifacts", help="输出目录")
    parser.add_argument("--embed-model", default="shibing624/text2vec-base-chinese", help="BERTopic 嵌入模型")
    parser.add_argument("--bert-model", default="bert-base-chinese", help="分类模型名称")
    parser.add_argument("--max-length", type=int, default=128, help="BERT 分词最大长度")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--num-epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--train-classifier", type=str, default="false", help="是否训练分类器 (true/false)")
    args = parser.parse_args()
    train_classifier = str(args.train_classifier).lower() == "true"
    return RunConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        embed_model=args.embed_model,
        bert_model=args.bert_model,
        max_length=args.max_length,
        test_size=args.test_size,
        num_epochs=args.num_epochs,
        train_classifier=train_classifier,
    )


def main() -> None:
    config = parse_args()
    os.makedirs(config.output_dir, exist_ok=True)
    environment_report()
    df = load_admin_dataset(config.data_path)
    topic_model = train_topic_model(df["文本"].tolist(), config.embed_model, config.output_dir)
    print(topic_model.get_topic_info().head())

    if config.train_classifier:
        trainer, mapping = train_classifier(df, config)
        sample_preds = analyze_new_items(
            trainer,
            trainer.tokenizer,  # type: ignore[arg-type]
            [
                "四川省 天府新区 市场监管 食品经营许可 下放",
                "上海市 浦东新区 商务 外商投资企业设立审批 委托",
            ],
            mapping,
        )
        print(f"示例预测: {sample_preds}")


if __name__ == "__main__":
    main()
