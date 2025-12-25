"""
信息论基础 - 大模型工程师必备知识

本模块涵盖信息论的核心概念，这些概念在理解和训练大语言模型时至关重要。

核心知识点：
1. 熵 (Entropy) - 衡量随机变量不确定性的度量
2. 交叉熵 (Cross-Entropy) - 衡量两个概率分布差异的度量
3. KL 散度 (KL Divergence) - 衡量信息损失的度量
4. 互信息 (Mutual Information) - 衡量两个变量相关性的度量
5. 信息瓶颈 (Information Bottleneck) - 深度学习中的信息压缩理论

这些概念与 LLM 的关系：
- 困惑度 (Perplexity) = 2^交叉熵，评估语言模型的核心指标
- 最小化交叉熵损失 = 最小化 KL 散度（让预测分布接近真实分布）
- 互信息可用于分析上下文与生成文本的关系
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


# ============================================================
# 1. 熵 (Entropy)
# ============================================================

def calculate_entropy(probabilities):
    """
    计算香农熵
    
    公式：H(X) = -Σ P(x) * log₂(P(x))
    
    参数：
        probabilities: 概率分布数组，总和应为 1
        
    返回：
        熵值（以比特为单位）
        
    示例：
        - 均匀分布的熵最大（最不确定）
        - One-hot 分布的熵为 0（最确定）
    """
    # 过滤掉零概率（log(0) 未定义）
    probs = np.array(probabilities)
    probs = probs[probs > 0]
    
    # 计算熵
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy


def explain_entropy():
    """
    熵的直观解释
    
    熵衡量的是"平均信息量"或"不确定性程度"：
    - 熵越高，信息量越大，不确定性越高
    - 熵越低，信息量越小，系统越确定
    
    在语言模型中的意义：
    - 高熵：模型预测分散，有很多可能的下一个词
    - 低熵：模型预测集中，对下一个词很有把握
    """
    
    print("=" * 70)
    print("1. 熵 (Entropy) - 衡量不确定性")
    print("=" * 70)
    
    # 示例1：均匀分布 vs 偏斜分布
    uniform_dist = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 均匀分布
    skewed_dist = np.array([0.9, 0.025, 0.025, 0.025, 0.025])  # 偏斜分布
    onehot_dist = np.array([0, 0, 1, 0, 0])  # One-hot 分布
    
    print("\n示例1：不同分布的熵对比")
    print("-" * 50)
    
    distributions = [
        (uniform_dist, "均匀分布 [0.2, 0.2, 0.2, 0.2, 0.2]"),
        (skewed_dist, "偏斜分布 [0.9, 0.025, 0.025, 0.025, 0.025]"),
        (onehot_dist, "One-hot 分布 [0, 0, 1, 0, 0]"),
    ]
    
    for dist, desc in distributions:
        entropy = calculate_entropy(dist)
        bar = '▓' * int(entropy * 10)
        print(f"{desc}")
        print(f"  熵 H(X) = {entropy:.4f} bits {bar}")
        if entropy > 2.0:
            print("  → 高熵：高度不确定，信息量大")
        elif entropy > 0.5:
            print("  → 中等熵：有一定确定性")
        else:
            print("  → 低熵：高度确定，信息量小")
    
    # 示例2：语言模型的熵
    print("\n示例2：语言模型预测的熵")
    print("-" * 50)
    
    # 模拟不同场景下的语言模型预测
    scenarios = [
        ("高确定性场景：'太阳从___升起'", 
         [0.01, 0.01, 0.01, 0.95, 0.01, 0.01],
         "模型对'东'很有把握"),
        ("中等确定性：'今天天气___'",
         [0.1, 0.3, 0.2, 0.15, 0.15, 0.1],
         "多个词都可能，需要更多上下文"),
        ("低确定性：'他说：___'",
         [0.15, 0.2, 0.15, 0.15, 0.2, 0.15],
         "各种词都可能，模型不确定"),
    ]
    
    for context, dist, comment in scenarios:
        entropy = calculate_entropy(dist)
        print(f"场景: {context}")
        print(f"  预测分布: {[f'{p:.2f}' for p in dist]}")
        print(f"  熵: {entropy:.4f} bits - {comment}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：不同分布的熵对比
    labels = ['均匀分布\n(高熵)', '偏斜分布\n(中熵)', 'One-hot\n(低熵)']
    entropies = [calculate_entropy(d) for d, _ in distributions]
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    bars = axes[0].bar(labels, entropies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Entropy (bits)', fontsize=12)
    axes[0].set_title('Entropy of Different Distributions', fontsize=14)
    axes[0].set_ylim(0, max(entropies) * 1.2)
    
    for bar, entropy in zip(bars, entropies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{entropy:.2f} bits', ha='center', va='bottom', fontsize=11)
    
    # 图2：概率分布对熵的影响
    p = np.linspace(0.01, 0.99, 100)
    entropy_binary = [-p[i] * np.log2(p[i]) - (1-p[i]) * np.log2(1-p[i]) if 0 < p[i] < 1 else 0 
                      for i in range(len(p))]
    
    axes[1].plot(p, entropy_binary, 'b-', linewidth=2)
    axes[1].set_xlabel('Probability of One Outcome', fontsize=12)
    axes[1].set_ylabel('Entropy (bits)', fontsize=12)
    axes[1].set_title('Binary Entropy Function H(p)', fontsize=14)
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Max entropy at p=0.5')
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/entropy_analysis.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/entropy_analysis.png")
    print()


# ============================================================
# 2. 交叉熵 (Cross-Entropy)
# ============================================================

def calculate_cross_entropy(p_true, q_pred):
    """
    计算交叉熵
    
    公式：H(P, Q) = -Σ P(x) * log₂(Q(x))
    
    参数：
        p_true: 真实概率分布
        q_pred: 预测概率分布
        
    返回：
        交叉熵值
        
    重要性质：
        - H(P, Q) ≥ H(P)，当 Q = P 时取等号
        - 交叉熵可以分解为：H(P, Q) = H(P) + D_KL(P || Q)
    """
    # 过滤掉零概率
    p_true = np.array(p_true)
    q_pred = np.array(q_pred)
    
    # 找出非零的索引
    nonzero_mask = p_true > 0
    p_nonzero = p_true[nonzero_mask]
    q_nonzero = q_pred[nonzero_mask]
    
    # 计算交叉熵
    cross_entropy = -np.sum(p_nonzero * np.log2(q_nonzero))
    
    return cross_entropy


def explain_cross_entropy():
    """
    交叉熵详解
    
    交叉熵衡量的是"用 Q 来编码 P 所需的平均比特数"。
    
    在机器学习中的应用：
    - 分类问题的损失函数
    - 语言模型的训练目标
    - 最小化交叉熵 ≈ 让预测分布接近真实分布
    
    为什么交叉熵比均方误差更适合分类？
    - 交叉熵直接作用于概率分布
    - 梯度更"陡峭"，收敛更快
    - 符合概率的语义
    """
    
    print("=" * 70)
    print("2. 交叉熵 (Cross-Entropy)")
    print("=" * 70)
    
    # 示例1：简单分类的交叉熵
    print("\n示例1：分类问题的交叉熵损失")
    print("-" * 50)
    
    # 真实标签（one-hot 编码）
    y_true = np.array([0, 0, 1, 0, 0])  # 正确类别是第3个
    
    # 不同质量的预测
    predictions = [
        np.array([0.05, 0.10, 0.70, 0.10, 0.05]),  # 好预测
        np.array([0.20, 0.20, 0.30, 0.20, 0.10]),  # 一般预测
        np.array([0.05, 0.05, 0.05, 0.05, 0.80]),  # 错误预测
    ]
    
    print(f"真实标签 (one-hot): {y_true}")
    print()
    
    for i, pred in enumerate(predictions, 1):
        ce = calculate_cross_entropy(y_true, pred)
        perplexity = 2 ** ce
        
        pred_str = '[' + ', '.join([f'{p:.2f}' for p in pred]) + ']'
        quality = ["高质量", "中等质量", "低质量"][i-1]
        
        print(f"预测 {i}: {pred_str}")
        print(f"  交叉熵: {ce:.4f} bits")
        print(f"  困惑度: {perplexity:.2f}")
        print(f"  质量: {quality}")
        print()
    
    # 示例2：语言模型中的交叉熵
    print("示例2：语言模型中的交叉熵")
    print("-" * 50)
    
    # 假设我们有一个句子 "今天 天气 很好"
    # 真实下一个词是 "适合"，模型预测了多个词的概率
    vocab = ["适合", "不错", "糟糕", "一般", "出门"]
    true_probs = np.array([0.5, 0.25, 0.05, 0.15, 0.05])  # 简化的真实分布
    
    # 模型预测
    model_pred = np.array([0.45, 0.30, 0.05, 0.10, 0.10])
    
    ce = calculate_cross_entropy(true_probs, model_pred)
    print(f"词汇表: {vocab}")
    print(f"真实分布: {[f'{p:.2f}' for p in true_probs]}")
    print(f"预测分布: {[f'{p:.2f}' for p in model_pred]}")
    print(f"交叉熵: {ce:.4f} bits")
    
    # 计算各个词对交叉熵的贡献
    print("\n各词对交叉熵的贡献:")
    for word, true_p, pred_p in zip(vocab, true_probs, model_pred):
        contribution = -true_p * np.log2(pred_p) if true_p > 0 else 0
        bar = '█' * int(contribution * 20)
        print(f"  {word}: -{true_p:.2f} × log₂({pred_p:.2f}) = {contribution:.4f} {bar}")


# ============================================================
# 3. KL 散度 (KL Divergence)
# ============================================================

def calculate_kl_divergence(p_true, q_pred):
    """
    计算 KL 散度 (Kullback-Leibler Divergence)
    
    公式：D_KL(P || Q) = Σ P(x) × log₂(P(x) / Q(x))
    
    参数：
        p_true: 真实概率分布 P
        q_pred: 预测概率分布 Q
        
    返回：
        KL 散度值
        
    重要性质：
        - D_KL(P || Q) ≥ 0 （吉布斯不等式）
        - D_KL(P || Q) ≠ D_KL(Q || P)（不对称）
        - D_KL(P || Q) = 0 当且仅当 P = Q
    """
    p_true = np.array(p_true)
    q_pred = np.array(q_pred)
    
    # 找出非零的索引
    nonzero_mask = (p_true > 0) & (q_pred > 0)
    p_nonzero = p_true[nonzero_mask]
    q_nonzero = q_pred[nonzero_mask]
    
    # 计算 KL 散度
    kl_div = np.sum(p_nonzero * np.log2(p_nonzero / q_nonzero))
    
    return kl_div


def explain_kl_divergence():
    """
    KL 散度详解
    
    KL 散度衡量的是"使用 Q 来近似 P 时损失的信息量"。
    
    在机器学习中的应用：
    - 变分自编码器 (VAE) 的损失函数
    - 知识蒸馏 (Knowledge Distillation)
    - 强化学习中的策略梯度方法
    - 语言模型的正则化
    
    与交叉熵的关系：
    - H(P, Q) = H(P) + D_KL(P || Q)
    - 当 P 固定时，最小化 H(P, Q) 等价于最小化 D_KL(P || Q)
    """
    
    print("\n" + "=" * 70)
    print("3. KL 散度 (KL Divergence)")
    print("=" * 70)
    
    # 示例1：KL 散度的基本性质
    print("\n示例1：KL 散度的不对称性")
    print("-" * 50)
    
    P = np.array([0.9, 0.1])
    Q = np.array([0.5, 0.5])
    
    kl_P_Q = calculate_kl_divergence(P, Q)  # D_KL(P || Q)
    kl_Q_P = calculate_kl_divergence(Q, P)  # D_KL(Q || P)
    
    print(f"分布 P: {P}")
    print(f"分布 Q: {Q}")
    print(f"\nD_KL(P || Q) = {kl_P_Q:.4f}")
    print(f"D_KL(Q || P) = {kl_Q_P:.4f}")
    print(f"\n注意：KL 散度是不对称的！")
    print(f"  - D_KL(P || Q) ≠ D_KL(Q || P)")
    print(f"  - 这意味着用 Q 近似 P 与用 P 近似 Q 是不同的")
    
    # 示例2：不同分布对的 KL 散度
    print("\n示例2：不同分布对的 KL 散度")
    print("-" * 50)
    
    distributions = {
        "均匀分布": np.array([0.25, 0.25, 0.25, 0.25]),
        "轻微偏斜": np.array([0.35, 0.25, 0.25, 0.15]),
        "中度偏斜": np.array([0.5, 0.2, 0.2, 0.1]),
        "高度偏斜": np.array([0.8, 0.1, 0.05, 0.05]),
    }
    
    target = distributions["均匀分布"]
    
    print(f"目标分布 P (均匀): {target}")
    print(f"\n{'近似分布 Q':<30} {'D_KL(P || Q)':<15} {'解读'}")
    print("-" * 70)
    
    for name, approx in distributions.items():
        kl = calculate_kl_divergence(target, approx)
        if kl < 0.1:
            interpretation = "近似很好"
        elif kl < 0.5:
            interpretation = "近似一般"
        else:
            interpretation = "近似较差"
        print(f"{name:<30} {kl:<15.4f} {interpretation}")
    
    # 示例3：语言模型中的 KL 散度
    print("\n示例3：知识蒸馏中的 KL 散度")
    print("-" * 50)
    
    # 教师模型的预测（软标签）
    teacher_pred = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    
    # 学生模型的预测
    student_pred = np.array([0.35, 0.35, 0.15, 0.1, 0.05])
    
    # 硬标签（one-hot）
    hard_label = np.array([0, 1, 0, 0, 0])
    
    kl_soft = calculate_kl_divergence(teacher_pred, student_pred)
    ce_hard = calculate_cross_entropy(hard_label, student_pred)
    
    print(f"教师预测 (软标签): {[f'{p:.2f}' for p in teacher_pred]}")
    print(f"学生预测:          {[f'{p:.2f}' for p in student_pred]}")
    print(f"\n与硬标签的交叉熵: {ce_hard:.4f}")
    print(f"与软标签的 KL 散度: {kl_soft:.4f}")
    print("\n知识蒸馏使用 KL 散度的好处：")
    print("  - 软标签保留了类别间的相似性信息")
    print("  - 让学生模型学习到更丰富的知识结构")


# ============================================================
# 4. 困惑度 (Perplexity)
# ============================================================

def calculate_perplexity(cross_entropy):
    """
    计算困惑度
    
    公式：Perplexity = 2^H(P, Q)
    
    困惑度可以理解为"平均而言，模型认为有多少个词是合理的候选"。
    - 困惑度越低，模型越好
    - 困惑度 = 词汇表大小时，模型相当于随机猜测
    """
    return 2 ** cross_entropy


def explain_perplexity():
    """
    困惑度详解
    
    困惑度 (Perplexity, PPL) 是评估语言模型质量的常用指标。
    
    直观理解：
    - PPL = 2^H 意味着平均有 PPL 个词是合理的选择
    - 如果 PPL = 100，模型平均认为有 100 个词都可能出现在下一个位置
    - 如果 PPL = 10，模型平均认为有 10 个词是可能的
    
    实际应用：
    - GPT-2 Small: PPL ≈ 50
    - GPT-2 Medium: PPL ≈ 20
    - GPT-3: PPL ≈ 15 (few-shot)
    """
    
    print("\n" + "=" * 70)
    print("4. 困惑度 (Perplexity) - 语言模型评估指标")
    print("=" * 70)
    
    # 示例1：不同交叉熵对应的困惑度
    print("\n示例1：交叉熵与困惑度的关系")
    print("-" * 50)
    
    cross_entropies = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    
    print(f"{'交叉熵 (bits)':<15} {'困惑度':<12} {'直观理解'}")
    print("-" * 70)
    
    for ce in cross_entropies:
        ppl = calculate_perplexity(ce)
        if ppl < 2:
            interpretation = "极好，几乎完全确定"
        elif ppl < 10:
            interpretation = "良好，预测较确定"
        elif ppl < 50:
            interpretation = "一般，有多个候选词"
        elif ppl < 100:
            interpretation = "较差，模型不确定"
        else:
            interpretation = "很差，接近随机猜测"
        
        print(f"{ce:<15.1f} {ppl:<12.1f} {interpretation}")
    
    # 示例2：计算句子的困惑度
    print("\n示例2：计算句子的困惑度")
    print("-" * 50)
    
    # 模拟一个句子的词级概率
    sentence = "今天 天气 很好 适合 出门 散步"
    word_probs = [0.8, 0.7, 0.6, 0.5, 0.7, 0.8]  # 每个词出现的概率
    
    # 句子级别的交叉熵（词概率的平均对数的负值）
    log_probs = [np.log2(p) for p in word_probs]
    avg_log_prob = np.mean(log_probs)
    sentence_ce = -avg_log_prob
    sentence_ppl = calculate_perplexity(sentence_ce)
    
    print(f"句子: {' '.join(sentence)}")
    print(f"词概率: {[f'{p:.2f}' for p in word_probs]}")
    print(f"平均对数概率: {avg_log_prob:.4f}")
    print(f"句子交叉熵: {sentence_ce:.4f}")
    print(f"句子困惑度: {sentence_ppl:.2f}")
    
    # 可视化困惑度
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：交叉熵 vs 困惑度
    ce_range = np.linspace(0.1, 10, 100)
    ppl_range = [2 ** ce for ce in ce_range]
    
    axes[0].semilogy(ce_range, ppl_range, 'b-', linewidth=2)
    axes[0].axhline(y=10000, color='r', linestyle='--', alpha=0.7, label='PPL = 10,000')
    axes[0].axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='PPL = 1,000')
    axes[0].axhline(y=100, color='green', linestyle='--', alpha=0.7, label='PPL = 100')
    axes[0].axhline(y=10, color='purple', linestyle='--', alpha=0.7, label='PPL = 10')
    axes[0].set_xlabel('Cross-Entropy (bits)', fontsize=12)
    axes[0].set_ylabel('Perplexity (log scale)', fontsize=12)
    axes[0].set_title('Cross-Entropy vs Perplexity', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 图2：不同模型规模的困惑度
    models = ['GPT-2 Small', 'GPT-2 Medium', 'GPT-2 Large', 'GPT-2 XL', 'GPT-3']
    params = ['124M', '355M', '774M', '1.5B', '175B']
    ppls = [50, 25, 18, 15, 12]  # 简化数据
    
    bars = axes[1].bar(range(len(models)), ppls, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('Perplexity vs Model Size (example)', fontsize=14)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels([f'{m}\n({p})' for m, p in zip(models, params)])
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, ppl in zip(bars, ppls):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{ppl}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/perplexity_analysis.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/perplexity_analysis.png")


# ============================================================
# 5. 互信息 (Mutual Information)
# ============================================================

def calculate_mutual_information(p_xy, p_x, p_y):
    """
    计算互信息
    
    公式：I(X; Y) = Σ P(x,y) × log₂(P(x,y) / (P(x) × P(y)))
    
    也等价于：
    I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    
    参数：
        p_xy: 联合分布
        p_x: X 的边缘分布
        p_y: Y 的边缘分布
        
    返回：
        互信息值
    """
    p_xy = np.array(p_xy)
    p_x = np.array(p_x)
    p_y = np.array(p_y)
    
    # 计算互信息
    mi = 0
    for i in range(len(p_xy)):
        for j in range(len(p_xy[0])):
            if p_xy[i,j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i,j] * np.log2(p_xy[i,j] / (p_x[i] * p_y[j]))
    
    return mi


def explain_mutual_information():
    """
    互信息详解
    
    互信息衡量的是"知道一个变量后，关于另一个变量减少的不确定性"。
    
    公式：I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    
    在深度学习中的应用：
    - 信息瓶颈理论 (Information Bottleneck)
    - 特征选择
    - 理解神经网络学习到的表示
    """
    
    print("\n" + "=" * 70)
    print("5. 互信息 (Mutual Information)")
    print("=" * 70)
    
    # 示例：上下文与生成文本的互信息
    print("\n示例：上下文与生成文本的关系")
    print("-" * 50)
    
    # 假设我们有以下上下文-文本对
    contexts = ["正面评价", "负面评价", "中性评价"]
    responses = ["积极回复", "消极回复", "中性回复"]
    
    # 联合分布（从数据中统计得到）
    joint_dist = np.array([
        [0.30, 0.05, 0.05],  # 正面评价
        [0.05, 0.30, 0.05],  # 负面评价
        [0.05, 0.05, 0.10],  # 中性评价
    ])
    
    # 边缘分布
    p_context = joint_dist.sum(axis=1)
    p_response = joint_dist.sum(axis=0)
    
    # 归一化
    p_context = p_context / p_context.sum()
    p_response = p_response / p_response.sum()
    
    # 计算互信息
    mi = calculate_mutual_information(joint_dist, p_context, p_response)
    
    # 计算条件熵
    h_response = calculate_entropy(p_response)
    h_response_given_context = 0
    for i, p_c in enumerate(p_context):
        response_given_context = joint_dist[i] / joint_dist[i].sum()
        h_response_given_context += p_c * calculate_entropy(response_given_context)
    
    print(f"上下文分布: {p_context}")
    print(f"回复分布:   {p_response}")
    print(f"\n互信息 I(上下文; 回复) = {mi:.4f} bits")
    print(f"回复的熵 H(回复) = {h_response:.4f} bits")
    print(f"条件熵 H(回复|上下文) = {h_response_given_context:.4f} bits")
    print(f"\n验证: H(回复) - H(回复|上下文) = {h_response:.4f} - {h_response_given_context:.4f} = {mi:.4f}")
    print("\n解读：")
    print(f"  - 知道上下文后，回复的不确定性减少了 {mi:.2f} bits")
    print(f"  - 互信息越高，说明上下文与回复的关联越强")


# ============================================================
# 6. 信息瓶颈理论简介
# ============================================================

def explain_information_bottleneck():
    """
    信息瓶颈理论 (Information Bottleneck)
    
    核心思想：在压缩表示 Z 中保留关于目标 Y 的所有相关信息。
    
    优化目标：最大化 I(Z; Y) - β × I(Z; X)
    - I(Z; Y)：保留关于 Y 的信息（有用信息）
    - I(Z; X)：压缩掉关于 X 的信息（冗余信息）
    
    在深度学习中的意义：
    - 解释为什么深度神经网络有效
    - 为模型压缩提供理论基础
    - 指导特征学习
    """
    
    print("\n" + "=" * 70)
    print("6. 信息瓶颈理论 (Information Bottleneck)")
    print("=" * 70)
    
    print("""
信息瓶颈理论由 Tishby 等人提出，核心思想是：

    优化目标：最大化 I(Z; Y) - β × I(Z; X)
    
    其中：
    - I(Z; Y)：表示 Z 中保留了多少关于 Y 的信息（有用信息）
    - I(Z; X)：表示 Z 中保留了多少关于 X 的信息（输入信息）
    - β：平衡压缩和保留信息的超参数
    
    直观理解：
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │    输入 X ──► 表示 Z ──► 目标 Y                         │
    │           │         │                                  │
    │           │    信息瓶颈                                 │
    │           │    (压缩表示)                               │
    │           └────────────────────────────────             │
    │                 过滤掉冗余信息                          │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    
    在大语言模型中的应用：
    1. 理解预训练：模型学习到压缩的文本表示
    2. 解释微调：微调时只更新与任务相关的信息
    3. 模型压缩：找到最小压缩表示同时保持性能
    
    关键洞见：
    - 深度网络逐层压缩信息
    - 靠近输入的层保留更多信息
    - 靠近输出的层更专注于任务相关信息
    """)


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("信息论基础 - 大模型工程师必备知识")
    print("=" * 70)
    
    # 运行各个示例
    explain_entropy()
    explain_cross_entropy()
    explain_kl_divergence()
    explain_perplexity()
    explain_mutual_information()
    explain_information_bottleneck()
    
    print("\n" + "=" * 70)
    print("信息论学习总结")
    print("=" * 70)
    print("""
核心概念速查表：

┌────────────────┬───────────────────────────────────────────────┐
│ 概念           │ 公式与应用                                      │
├────────────────┼───────────────────────────────────────────────┤
│ 熵             │ H(X) = -Σ P(x)·log₂(P(x))                      │
│                │ 衡量随机变量的不确定性                           │
├────────────────┼───────────────────────────────────────────────┤
│ 交叉熵         │ H(P,Q) = -Σ P(x)·log₂(Q(x))                    │
│                │ 分类/语言模型的损失函数                          │
├────────────────┼───────────────────────────────────────────────┤
│ KL 散度        │ D_KL(P||Q) = Σ P(x)·log₂(P(x)/Q(x))            │
│                │ 信息损失度量，知识蒸馏                           │
├────────────────┼───────────────────────────────────────────────┤
│ 困惑度         │ PPL = 2^H(P,Q)                                 │
│                │ 语言模型评估的核心指标                           │
├────────────────┼───────────────────────────────────────────────┤
│ 互信息         │ I(X;Y) = H(X) - H(X|Y)                         │
│                │ 衡量两个变量的相关性                             │
└────────────────┴───────────────────────────────────────────────┘

与 LLM 的关系：
- 困惑度是评估语言模型质量的核心指标
- 交叉熵损失是训练语言模型的目标函数
- KL 散度用于知识蒸馏和模型压缩
- 信息瓶颈理论解释深度学习的表示学习

下一步学习：
- 线性代数：矩阵运算、特征分解
- 优化算法：梯度下降、Adam 优化器
- 注意力机制：Query-Key-Value 抽象
    """)
