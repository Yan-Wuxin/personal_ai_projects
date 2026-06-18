# Transformer论文复现——基于PyTorch的中英机器翻译

> **论文**：Attention Is All You Need（NeurIPS 2017）
> **框架**：PyTorch
> **任务**：Chinese → English Neural Machine Translation (NMT)

---

## 项目简介

本项目基于论文 **Attention Is All You Need** 对 Transformer 神经机器翻译模型进行了完整复现，并在中英平行语料上完成训练与推理实验。

项目实现了 Transformer 的核心结构，包括：

* Multi-Head Self-Attention
* Position-wise Feed Forward Network
* Positional Encoding
* Encoder-Decoder Attention
* Masked Multi-Head Attention
* Label Smoothing
* Noam Learning Rate Scheduler
* Beam Search / Greedy Decoding（可扩展）

训练过程中结合混合精度训练（AMP）、梯度裁剪、模型断点恢复等策略，实现了完整的训练与推理流程。

---

# 项目结构

```text
Transformer-NMT/
│
├── model.py                 # Transformer模型
├── train.py                 # 模型训练
├── inference.py             # 翻译推理
├── dataset.py               # 数据集构建
├── utils.py                 # 工具函数
├── config.py                # 参数配置
│
├── data/
│   ├── train.cn
│   ├── train.en
│   ├── vocab.cn
│   └── vocab.en
│
├── checkpoints/
│   ├── best_model.pt
│   └── latest.pt
│
├── logs/
│
└── README.md
```

---

# 模型结构

Transformer 由 Encoder 与 Decoder 两部分组成。

## Encoder

每层 Encoder 包括：

```
Multi-Head Attention
        │
Residual + LayerNorm
        │
Feed Forward
        │
Residual + LayerNorm
```

共堆叠 6 层。

---

## Decoder

每层 Decoder 包括：

```
Masked Multi-Head Attention
          │
Encoder-Decoder Attention
          │
Feed Forward Network
```

同样堆叠 6 层。

---

# 训练配置

| 参数                | 数值             |
| ----------------- | -------------- |
| Encoder Layers    | 6              |
| Decoder Layers    | 6              |
| d_model           | 512            |
| FeedForward       | 2048           |
| Heads             | 8              |
| Dropout           | 0.2            |
| Batch Size        | 64             |
| Optimizer         | Adam           |
| Learning Rate     | Noam Scheduler |
| Label Smoothing   | 0.1            |
| Gradient Clipping | 1.0            |
| Mixed Precision   | AMP            |

---

# 数据集

采用中英平行语料进行训练。

预处理流程包括：

* 中文分词
* 英文Tokenize
* 建立中英文词表
* Padding
* Mask生成

最终构建 Encoder 输入与 Decoder 输入。

---

# 训练过程

训练过程中实现：

* 自动保存最佳模型
* Checkpoint恢复训练
* 学习率动态调整（Noam Scheduler）
* AMP混合精度训练
* Loss实时记录
* 梯度裁剪防止梯度爆炸

训练约百余轮后，模型训练过程稳定，Loss持续下降并收敛。

---

# 论文复现内容

本项目复现了论文中的主要技术模块：

* Positional Encoding
* Multi-Head Attention
* Scaled Dot-Product Attention
* Residual Connection
* Layer Normalization
* Position-wise Feed Forward
* Encoder-Decoder Architecture
* Mask机制
* Noam Learning Rate Scheduler
* Label Smoothing

未复现部分：

* 大规模WMT机器翻译数据集
* 多GPU分布式训练
* Beam Search优化
* BLEU系统评测

---

# 参考文献

Ashish Vaswani, Noam Shazeer, Niki Parmar, et al.

**Attention Is All You Need**

NeurIPS 2017.
