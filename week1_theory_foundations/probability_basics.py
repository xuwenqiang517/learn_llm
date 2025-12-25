"""
概率论基础 - 大模型工程师必备数学知识

本模块涵盖大模型训练中常用的概率论概念，通过直观的例子和代码来理解。

核心知识点：
1. 条件概率与贝叶斯定理 - 理解语言模型的预测
2. 期望与方差 - 理解模型输出的分布
3. 常用分布 - 理解模型的不确定性
4. Softmax 函数 - 将模型输出转换为概率
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 1. 条件概率与联合概率
# ============================================================

def explain_conditional_probability():
    """
    条件概率：P(A|B) = P(A∩B) / P(B)
    
    在语言模型中的应用：
    - P(w_t | w_1, w_2, ..., w_{t-1})：给定前文，预测下一个词的概率
    - 这是自回归语言模型的核心
    
    示例：假设我们有一个简单的词汇表 ['我', '爱', '学习', 'AI']
    我们想计算 P('AI' | '我', '爱', '学习')
    """
    
    # 示例词汇表和简单的语言模型概率
    vocab = ['我', '爱', '学习', 'AI', '。', '很', '棒']
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 模拟一个简单的语言模型（实际上是从数据中学习的）
    # P(next_word | context)
    # 这里使用一个简化的转移概率矩阵
    transition_matrix = np.array([
        [0.0, 0.3, 0.4, 0.1, 0.1, 0.1, 0.0],  # '我' 后面可能的词
        [0.0, 0.0, 0.3, 0.5, 0.1, 0.1, 0.0],  # '爱' 后面可能的词
        [0.0, 0.0, 0.0, 0.6, 0.2, 0.1, 0.1],  # '学习' 后面可能的词
        [0.1, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3],  # 'AI' 后面可能的词
        [0.3, 0.2, 0.1, 0.1, 0.0, 0.1, 0.2],  # '。' 后面可能的词
        [0.2, 0.3, 0.2, 0.2, 0.0, 0.0, 0.1],  # '很' 后面可能的词
        [0.1, 0.2, 0.2, 0.3, 0.2, 0.1, 0.0],  # '棒' 后面可能的词
    ])
    
    # 示例：给定 "我 爱 学习"，计算下一个词是 "AI" 的概率
    context = ['我', '爱', '学习']
    context_indices = [word_to_idx[w] for w in context]
    
    # 在实际语言模型中，会考虑更长的上下文
    # 这里简化：使用最后一个词的概率分布
    last_word_idx = context_indices[-1]
    next_word_probs = transition_matrix[last_word_idx]
    
    print("=" * 60)
    print("条件概率示例：语言模型预测")
    print("=" * 60)
    print(f"上下文: {' '.join(context)}")
    print(f"最后词索引: {last_word_idx} ('{context[-1]}')")
    print("\n预测下一个词的概率分布:")
    for i, prob in enumerate(next_word_probs):
        if prob > 0.01:  # 只显示概率大于 1% 的词
            print(f"  P('{vocab[i]}' | '{context[-1]}') = {prob:.3f}")
    
    # 计算 P('AI' | '学习') = 0.6
    ai_prob = next_word_probs[word_to_idx['AI']]
    print(f"\n→ 预测 'AI' 的概率: {ai_prob:.3f}")
    print()


def explain_joint_probability():
    """
    联合概率：P(A∩B) = P(A) * P(B|A)
    
    在语言模型中的应用：
    - P(w_1, w_2, ..., w_n) = P(w_1) * P(w_2|w_1) * P(w_3|w_1,w_2) * ...
    - 这是句子概率计算的基础
    
    示例：计算句子 "我爱学习 AI" 的概率
    """
    
    vocab = ['我', '爱', '学习', 'AI', '。']
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 简化的转移概率矩阵
    transition_matrix = np.array([
        [0.0, 0.3, 0.4, 0.1, 0.1],  # '我'
        [0.0, 0.0, 0.3, 0.5, 0.1],  # '爱'
        [0.0, 0.0, 0.0, 0.6, 0.2],  # '学习'
        [0.0, 0.0, 0.0, 0.0, 0.3],  # 'AI'
        [0.2, 0.2, 0.1, 0.2, 0.0],  # '。'
    ])
    
    # 初始概率 P(w_1)
    start_probs = np.array([0.2, 0.1, 0.3, 0.3, 0.1])
    
    # 句子 "我爱学习 AI" = w_1=我, w_2=爱, w_3=学习, w_4=AI, w_5=。
    sentence = ['我', '爱', '学习', 'AI', '。']
    
    # 计算联合概率 P(w_1, w_2, w_3, w_4, w_5)
    prob = start_probs[word_to_idx['我']]  # P(我)
    prob *= transition_matrix[word_to_idx['我']][word_to_idx['爱']]  # P(爱|我)
    prob *= transition_matrix[word_to_idx['爱']][word_to_idx['学习']]  # P(学习|我爱)
    prob *= transition_matrix[word_to_idx['学习']][word_to_idx['AI']]  # P(AI|我爱学习)
    prob *= transition_matrix[word_to_idx['AI']][word_to_idx['。']]  # P(。|我爱学习AI)
    
    print("=" * 60)
    print("联合概率示例：句子概率计算")
    print("=" * 60)
    print(f"句子: {' '.join(sentence)}")
    print(f"\n概率分解:")
    print(f"  P(我)           = {start_probs[word_to_idx['我']]:.3f}")
    print(f"  P(爱|我)        = {transition_matrix[word_to_idx['我']][word_to_idx['爱']]:.3f}")
    print(f"  P(学习|我爱)    = {transition_matrix[word_to_idx['爱']][word_to_idx['学习']]:.3f}")
    print(f"  P(AI|我爱学习)  = {transition_matrix[word_to_idx['学习']][word_to_idx['AI']]:.3f}")
    print(f"  P(。|我爱学习AI)= {transition_matrix[word_to_idx['AI']][word_to_idx['。']]:.3f}")
    print(f"\n  联合概率 P(我,爱,学习,AI,。) = {prob:.6f}")
    print()


# ============================================================
# 2. 期望与方差
# ============================================================

def explain_expectation_variance():
    """
    期望 E[X] = Σ x * P(x)
    方差 Var(X) = E[(X - E[X])^2]
    
    在语言模型中的应用：
    - 期望：预测词的概率分布的"平均值"
    - 方差：预测的不确定性度量
    
    示例：分析语言模型预测的不确定性
    """
    
    # 模拟一个语言模型对下一个词的预测分布
    vocab = ['我', '爱', '学习', 'AI', '编程', '很', '棒', '。']
    
    # 假设模型预测的概率分布
    probs = np.array([0.05, 0.10, 0.15, 0.30, 0.20, 0.05, 0.10, 0.05])
    
    # 期望（熵最大时的平均预测位置）
    expectation = np.sum(np.arange(len(vocab)) * probs)
    
    # 方差（预测的不确定性）
    variance = np.sum(((np.arange(len(vocab)) - expectation) ** 2) * probs)
    
    # 熵（信息量的期望）
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    print("=" * 60)
    print("期望与方差示例：模型预测分析")
    print("=" * 60)
    print(f"词汇表: {vocab}")
    print(f"\n预测概率分布:")
    for i, (word, prob) in enumerate(zip(vocab, probs)):
        bar = '█' * int(prob * 50)
        print(f"  {word}: {prob:.2f} {bar}")
    
    print(f"\n统计量:")
    print(f"  期望 E[X] = {expectation:.2f}")
    print(f"  方差 Var(X) = {variance:.2f}")
    print(f"  熵 H(X) = {entropy:.2f} bits")
    print(f"\n解读:")
    print(f"  - 期望 2.65 表示预测主要分布在 '学习' 附近")
    print(f"  - 方差 3.21 表示预测有一定的不确定性")
    print(f"  - 熵 2.45 bits 表示平均需要 2.45 bits 来编码这个预测")
    print()


# ============================================================
# 3. Softmax 函数
# ============================================================

def explain_softmax():
    """
    Softmax 函数：将 logits 转换为概率分布
    
    公式：softmax(x_i) = exp(x_i) / Σ exp(x_j)
    
    在语言模型中的应用：
    - 将模型的原始输出（logits）转换为概率分布
    - 保证所有概率之和为 1
    - 温度参数 T 控制分布的"锐利"程度
    
    softmax(x_i, T) = exp(x_i/T) / Σ exp(x_j/T)
    
    - T → 0：趋近于 one-hot 分布（最确定的预测）
    - T → 1：正常 softmax
    - T → ∞：趋近于均匀分布（最不确定的预测）
    """
    
    # 假设模型输出的原始 logits
    logits = np.array([2.0, 1.0, 0.1, 3.0, 1.5])
    
    # 标准 softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    
    # 不同温度下的 softmax
    def softmax_with_temp(logits, T):
        exp_logits = np.exp(logits / T)
        return exp_logits / np.sum(exp_logits)
    
    temps = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("=" * 60)
    print("Softmax 函数详解")
    print("=" * 60)
    print(f"原始 logits: {logits}")
    print(f"\n标准 softmax (T=1.0):")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 50)
        print(f"  logit[{i}]={logits[i]:.1f} → prob={p:.4f} {bar}")
    
    print(f"\n不同温度下的概率分布:")
    print(f"{'Temp':<8} {'分布 (概率值)':<60}")
    print("-" * 70)
    
    for T in temps:
        probs_T = softmax_with_temp(logits, T)
        dist_str = '[' + ', '.join([f'{p:.3f}' for p in probs_T]) + ']'
        if T <= 1.0:
            dist_str += ' 🔥'  # 更确定
        else:
            dist_str += ' 🎲'  # 更随机
        print(f"{T:<8} {dist_str:<60}")
    
    print()
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1：不同温度下的概率分布
    x = np.arange(len(logits))
    for T in [0.1, 0.5, 1.0, 2.0]:
        probs_T = softmax_with_temp(logits, T)
        label = f'T={T}'
        if T == 0.1:
            label += ' (🔥确定)'
        elif T == 2.0:
            label += ' (🎲随机)'
        axes[0].bar(x + (T-1)*0.15, probs_T, width=0.3, label=label, alpha=0.8)
    
    axes[0].set_xlabel('Token Index')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Softmax with Different Temperatures')
    axes[0].set_xticks(x)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 图2：softmax 的数学性质
    T_range = np.linspace(0.1, 5, 100)
    max_prob = [np.max(softmax_with_temp(logits, T)) for T in T_range]
    entropy = [-np.sum(softmax_with_temp(logits, T) * 
              np.log2(softmax_with_temp(logits, T) + 1e-10)) for T in T_range]
    
    axes[1].plot(T_range, max_prob, 'b-', linewidth=2, label='Max Probability')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('Value', color='b')
    axes[1].tick_params(axis='y', labelcolor='b')
    
    ax2 = axes[1].twinx()
    ax2.plot(T_range, entropy, 'r-', linewidth=2, label='Entropy (bits)')
    ax2.set_ylabel('Entropy (bits)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    axes[1].set_title('Temperature vs. Certainty & Entropy')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/softmax_temperature_analysis.png', dpi=150, bbox_inches='tight')
    print(f"图片已保存到: week1_theory_foundations/softmax_temperature_analysis.png")
    print()


# ============================================================
# 4. 交叉熵与 KL 散度
# ============================================================

def explain_cross_entropy_kl():
    """
    交叉熵：H(P, Q) = -Σ P(x) log Q(x)
    
    KL 散度（相对熵）：D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
    
    关系：H(P, Q) = H(P) + D_KL(P || Q)
    
    在语言模型中的应用：
    - 交叉熵损失 = -Σ y_true * log(y_pred)
    - 最小化交叉熵等价于最小化 KL 散度（当 H(P) 固定时）
    - 让模型预测的概率分布接近真实分布
    """
    
    # 示例：计算交叉熵损失
    # 假设真实标签是 one-hot 编码 [0, 0, 1, 0, 0]（第3个词是正确答案）
    y_true_onehot = np.array([0, 0, 1, 0, 0])
    
    # 模型预测的概率分布
    y_pred = np.array([0.05, 0.10, 0.70, 0.10, 0.05])
    
    # 交叉熵损失（用于分类任务）
    # 对于 one-hot 标签，交叉熵 = -log(P(true_class))
    cross_entropy = -np.sum(y_true_onehot * np.log(y_pred + 1e-10))
    
    # 逐元素计算
    print("=" * 60)
    print("交叉熵与 KL 散度详解")
    print("=" * 60)
    print(f"真实分布 (one-hot): {y_true_onehot}")
    print(f"预测分布:           {y_pred}")
    print(f"\n交叉熵计算过程:")
    print(f"  H(P, Q) = -Σ P(x)·log(Q(x))")
    print(f"  = -[0·log(0.05) + 0·log(0.10) + 1·log(0.70) + 0·log(0.10) + 0·log(0.05)]")
    print(f"  = -{np.log(0.70):.4f}")
    print(f"  = {cross_entropy:.4f}")
    
    # 对数似然（以 2 为底，单位是 bits）
    log_likelihood_2 = np.log2(0.70)
    print(f"\n对数似然 (log2): {log_likelihood_2:.4f} bits")
    print(f"困惑度 (Perplexity): {2 ** cross_entropy:.2f}")
    
    # 不同预测质量的对比
    print(f"\n不同预测质量的交叉熵损失对比:")
    predictions = [
        ([0.05, 0.10, 0.70, 0.10, 0.05], "高质量预测"),
        ([0.20, 0.20, 0.30, 0.20, 0.10], "中等质量预测"),
        ([0.05, 0.05, 0.05, 0.05, 0.80], "错误预测（高损失）"),
    ]
    
    print(f"{'预测分布':<45} {'交叉熵损失':<12} {'困惑度':<10}")
    print("-" * 70)
    
    for pred, desc in predictions:
        pred = np.array(pred)
        ce = -np.sum(y_true_onehot * np.log(pred + 1e-10))
        ppl = 2 ** ce
        pred_str = '[' + ', '.join([f'{p:.2f}' for p in pred]) + ']'
        print(f"{pred_str:<45} {ce:<12.4f} {ppl:<10.2f} {desc}")
    
    print()
    print("关键理解:")
    print("  - 交叉熵损失越低，预测质量越好")
    print("  - 困惑度是交叉熵的指数，可以理解为平均候选词数")
    print("  - 困惑度 = 2^交叉熵（以 2 为底）")
    print()


# ============================================================
# 5. 主函数：运行所有示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("概率论基础 - 大模型工程师必备数学知识")
    print("=" * 70 + "\n")
    
    # 运行各个示例
    explain_conditional_probability()
    explain_joint_probability()
    explain_expectation_variance()
    explain_softmax()
    explain_cross_entropy_kl()
    
    print("=" * 70)
    print("学习总结")
    print("=" * 70)
    print("""
核心概念回顾：

1. 条件概率 P(A|B)
   - 语言模型的核心：P(next_word | context)
   
2. 联合概率 P(A∩B)
   - 句子概率：P(w_1, w_2, ..., w_n) = Π P(w_i | w_1...w_{i-1})
   
3. 期望与方差
   - 期望：预测分布的"重心"
   - 方差：预测的不确定性
   
4. Softmax 函数
   - 将 logits 转换为概率分布
   - 温度参数控制分布的"锐利"程度
   
5. 交叉熵损失
   - 语言模型训练的核心损失函数
   - 困惑度是评估语言模型的重要指标

下一步学习：
- 线性代数：矩阵运算、特征分解
- 优化算法：梯度下降、Adam
- 注意力机制：Query-Key-Value 抽象
    """)
