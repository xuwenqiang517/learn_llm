# Week1 理论与基础 - 大模型工程师核心技能

本文件夹包含大模型工程师必须掌握的数学基础和核心理论知识。

## 目录结构

### 1. 概率论与信息论基础
- `probability_basics.py` - 概率论核心概念（条件概率、贝叶斯定理、期望方差）
- `information_theory.py` - 信息论基础（熵、交叉熵、KL散度）

### 2. 线性代数与优化
- `linear_algebra.py` - 矩阵运算、特征分解、奇异值分解
- `optimization.py` - 梯度下降、随机梯度下降、Adam优化器

### 3. NLP 基础
- `tokenization.py` - 分词算法（BPE、WordPiece、SentencePiece）
- `word_embeddings.py` - 词向量表示（Word2Vec、GloVe、位置编码）

### 4. 注意力机制
- `attention_theory.py` - 注意力机制的理论基础
- `scaled_dot_product_attention.py` - 缩放点积注意力的 NumPy 实现
- `multi_head_attention.py` - 多头注意力机制实现
- `transformer_block.py` - Transformer 编码器/解码器块

## 学习建议

1. **第一阶段**：先理解概率论和信息论基础，这些是理解损失函数的基础
2. **第二阶段**：掌握线性代数运算，为理解矩阵运算打下基础
3. **第三阶段**：学习 NLP 基础，理解文本如何被计算机处理
4. **第四阶段**：深入注意力机制，这是 Transformer 的核心

## 核心概念速览

### 概率论在 LLM 中的应用
- **交叉熵损失**：语言模型训练的核心损失函数
- **Softmax**：将 logits 转换为概率分布
- **采样策略**：Greedy Search、Beam Search、Top-k、Top-p

### 线性代数在 LLM 中的应用
- **矩阵乘法**：注意力计算的核心运算
- **矩阵分解**：理解 LoRA 等微调方法的基础
- **特征值/奇异值**：理解模型压缩和蒸馏

### 注意力机制的本质
- **Query-Key-Value**：信息检索的数学抽象
- **自注意力**：序列内部依赖关系的建模
- **多头注意力**：并行学习不同子空间的特征
