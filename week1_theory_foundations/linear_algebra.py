"""
线性代数基础 - 大模型工程师必备数学知识

本模块涵盖大模型训练和推理中常用的线性代数知识。

核心知识点：
1. 矩阵基础运算 - 矩阵乘法、转置、逆矩阵
2. 特征分解与奇异值分解 - 理解矩阵的数学性质
3. 范数与距离 - 衡量向量/矩阵的大小和相似度
4. 矩阵运算在 LLM 中的应用 - 注意力计算、嵌入表示

为什么需要线性代数？
- Transformer 中的所有计算都是矩阵运算
- 注意力机制本质上就是矩阵乘法
- 理解 LoRA 等微调方法需要矩阵分解知识
- 模型压缩和量化涉及矩阵操作
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. 矩阵基础运算
# ============================================================

def explain_matrix_basics():
    """
    矩阵基础运算详解
    
    矩阵是线性代数的核心，在 LLM 中无处不在：
    - 词嵌入：矩阵 (vocab_size, embedding_dim)
    - 注意力权重：矩阵 (seq_len, seq_len)
    - 权重矩阵：矩阵 (input_dim, output_dim)
    
    基本运算：
    - 转置：改变矩阵的形状
    - 矩阵乘法：组合线性变换
    - 逆矩阵：解线性方程组
    """
    
    print("=" * 70)
    print("1. 矩阵基础运算")
    print("=" * 70)
    
    # 示例1：创建和操作矩阵
    print("\n示例1：矩阵的创建和基本操作")
    print("-" * 50)
    
    # 创建词嵌入矩阵示例
    # 假设词汇表大小为 5，嵌入维度为 3
    vocab_size, embed_dim = 5, 3
    embeddings = np.array([
        [0.1, 0.2, 0.3],  # 词 0
        [0.4, 0.5, 0.6],  # 词 1
        [0.7, 0.8, 0.9],  # 词 2
        [1.0, 1.1, 1.2],  # 词 3
        [1.3, 1.4, 1.5],  # 词 4
    ])
    
    print(f"词嵌入矩阵形状: {embeddings.shape}")
    print(f"词汇表大小: {vocab_size}")
    print(f"嵌入维度: {embed_dim}")
    print(f"\n嵌入矩阵:\n{embeddings}")
    
    # 矩阵转置
    embeddings_T = embeddings.T
    print(f"\n转置后形状: {embeddings_T.shape}")
    
    # 示例2：矩阵乘法的应用
    print("\n示例2：矩阵乘法 - 计算词间相似度")
    print("-" * 50)
    
    # 计算词嵌入之间的相似度矩阵
    # similarity = embeddings × embeddings^T
    # 每一行表示一个词与其他所有词的相似度
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    
    print("词嵌入相似度矩阵 (embeddings × embeddings^T):")
    print("     词0    词1    词2    词3    词4")
    for i, row in enumerate(similarity_matrix):
        row_str = ' '.join([f'{v:6.2f}' for v in row])
        print(f"词{i}  {row_str}")
    
    print(f"\n解读：对角线是词与自身的相似度（范数的平方）")
    print(f"      非对角线是不同词之间的相似度")
    print(f"      词2和词3最相似（相似度={similarity_matrix[2,3]:.2f}）")
    
    # 示例3：矩阵乘法的维度规则
    print("\n示例3：矩阵乘法的维度规则")
    print("-" * 50)
    
    A = np.random.randn(2, 3)  # 2×3 矩阵
    B = np.random.randn(3, 4)  # 3×4 矩阵
    
    print(f"矩阵 A 形状: {A.shape} (2行3列)")
    print(f"矩阵 B 形状: {B.shape} (3行4列)")
    print(f"矩阵乘法 A × B 形状: {np.matmul(A, B).shape} (2行4列)")
    
    print("\n维度规则：")
    print("  ┌─────────────────────┐")
    print("  │ (m, n) × (n, k) = (m, k) │")
    print("  │   ↑    ↑      ↑          │")
    print("  │   │    │      │          │")
    print("  │  行   中间   结果        │")
    print("  │  数   维度   维度        │")
    print("  └─────────────────────┘")
    
    # 在注意力机制中的应用
    print("\n注意力机制中的矩阵形状：")
    Q = np.random.randn(2, 8, 64)  # (batch, seq_len, d_k)
    K = np.random.randn(2, 8, 64)  # (batch, seq_len, d_k)
    V = np.random.randn(2, 8, 64)  # (batch, seq_len, d_v)
    
    print(f"  Query 形状: {Q.shape}")
    print(f"  Key 形状:   {K.shape}")
    print(f"  Value 形状: {V.shape}")
    print(f"  Q × K^T 形状: {np.matmul(Q, np.transpose(K, (0, 2, 1))).shape}")
    print(f"  (QK^T) × V 形状: {np.matmul(np.matmul(Q, np.transpose(K, (0, 2, 1))), V).shape}")


# ============================================================
# 2. 特征分解与特征值
# ============================================================

def explain_eigen_decomposition():
    """
    特征分解详解
    
    定义：对于方阵 A，如果存在向量 v 和标量 λ 使得 A×v = λ×v
    则称 v 是特征向量，λ 是特征值。
    
    性质：
    - 只有方阵才能进行特征分解
    - n×n 矩阵最多有 n 个特征值
    - 特征向量张成矩阵的特征空间
    
    在 LLM 中的应用：
    - 理解矩阵的谱性质
    - 主成分分析 (PCA)
    - 谱归一化
    """
    
    print("\n" + "=" * 70)
    print("2. 特征分解与特征值")
    print("=" * 70)
    
    # 示例1：计算特征值和特征向量
    print("\n示例1：计算方阵的特征值和特征向量")
    print("-" * 50)
    
    # 创建一个 2×2 矩阵
    A = np.array([[3, 1], [1, 3]])
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print(f"矩阵 A:")
    print(A)
    print(f"\n特征值: {eigenvalues}")
    print(f"\n特征向量:")
    for i, (ev, evec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        print(f"  λ_{i+1} = {ev:.2f}, v_{i+1} = {evec}")
    
    # 验证 A×v = λ×v
    print("\n验证 A×v = λ×v:")
    for i, (ev, evec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = np.dot(A, evec)
        lambda_v = ev * evec
        print(f"  A×v_{i+1} = {Av.round(3)}")
        print(f"  λ×v_{i+1}  = {lambda_v.round(3)}")
        print(f"  相等: {np.allclose(Av, lambda_v)}")
    
    # 示例2：特征值的几何意义
    print("\n示例2：特征值的几何意义")
    print("-" * 50)
    
    # 可视化特征向量的方向
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始矩阵和特征向量
    A = np.array([[3, 1], [1, 3]])
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # 绘制原点和特征向量
    origin = [0, 0]
    
    axes[0].arrow(origin[0], origin[1], eigenvectors[0,0]*2, eigenvectors[1,0]*2, 
                  head_width=0.1, head_length=0.05, fc='red', ec='red', linewidth=2)
    axes[0].arrow(origin[0], origin[1], eigenvectors[0,1]*2, eigenvectors[1,1]*2, 
                  head_width=0.1, head_length=0.05, fc='blue', ec='blue', linewidth=2)
    
    # 绘制单位圆
    theta = np.linspace(0, 2*np.pi, 100)
    x, y = np.cos(theta), np.sin(theta)
    circle = np.vstack([x, y])
    transformed = np.dot(A, circle)
    
    axes[0].plot(x, y, 'k--', alpha=0.5, label='单位圆')
    axes[0].plot(transformed[0], transformed[1], 'g-', linewidth=2, label='A×单位圆')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title(f'Eigenvalues: {eigenvalues[0]:.1f}, {eigenvalues[1]:.1f}\nEigenvectors (red, blue)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # 绘制不同矩阵的特征值分布
    matrices = [
        np.array([[2, 0], [0, 2]]),
        np.array([[3, 1], [1, 3]]),
        np.array([[1, 2], [3, 4]]),
    ]
    
    eigenvalues_list = [np.linalg.eigvals(m) for m in matrices]
    
    for i, (m, evs) in enumerate(zip(matrices, eigenvalues_list)):
        for ev in evs:
            axes[1].scatter(ev.real, ev.imag, s=100, label=f'Matrix {i+1}')
    
    axes[1].set_xlabel('实部')
    axes[1].set_ylabel('虚部')
    axes[1].set_title('复平面上的特征值分布')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/eigenvalue_analysis.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/eigenvalue_analysis.png")
    
    # 示例3：谱范数
    print("\n示例3：谱范数 - 矩阵的最大拉伸因子")
    print("-" * 50)
    
    # 谱范数 = 最大特征值的平方根
    spectral_norm = np.sqrt(max(abs(eigenvalues)))
    
    print(f"矩阵 A 的特征值: {abs(eigenvalues)}")
    print(f"谱范数 σ_max(A) = √{max(abs(eigenvalues)):.2f} = {spectral_norm:.2f}")
    print("\n谱范数的意义：")
    print("  - 表示矩阵对向量最大可能的拉伸倍数")
    print("  - 在梯度裁剪和权重初始化中很重要")
    print("  - ||Ax||_2 ≤ σ_max(A) × ||x||_2")


# ============================================================
# 3. 奇异值分解 (SVD)
# ============================================================

def explain_svd():
    """
    奇异值分解详解
    
    定义：对于任意 m×n 矩阵 A，可以分解为：
    A = U × Σ × V^T
    
    其中：
    - U 是 m×m 正交矩阵（列向量是左奇异向量）
    - Σ 是 m×n 对角矩阵（对角线是奇异值）
    - V 是 n×n 正交矩阵（列向量是右奇异向量）
    
    与特征分解的关系：
    - SVD 适用于任意矩阵（不仅是方阵）
    - 奇异值是 A×A^T 或 A^T×A 的特征值的平方根
    
    在 LLM 中的应用：
    - 矩阵近似和压缩
    - 降维（截断 SVD）
    - 潜在语义分析 (LSA)
    - 理解 LoRA 的矩阵分解思想
    """
    
    print("\n" + "=" * 70)
    print("3. 奇异值分解 (SVD)")
    print("=" * 70)
    
    # 示例1：完整 SVD
    print("\n示例1：完整奇异值分解")
    print("-" * 50)
    
    # 创建示例矩阵
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    
    # 计算 SVD
    U, S, Vt = np.linalg.svd(A)
    
    print(f"原矩阵 A ({A.shape}):")
    print(A)
    print(f"\n左奇异向量 U ({U.shape}):")
    print(U)
    print(f"\n奇异值 Σ ({S.shape}):")
    print(S)
    print(f"\n右奇异向量 V^T ({Vt.shape}):")
    print(Vt)
    
    # 验证 A = U × Σ × V^T
    A_reconstructed = np.dot(np.dot(U, np.diag(S)), Vt)
    print(f"\n验证 A = UΣV^T:")
    print(f"重构矩阵与原矩阵的差异: {np.linalg.norm(A - A_reconstructed):.2e}")
    
    # 示例2：截断 SVD 和矩阵近似
    print("\n示例2：截断 SVD - 矩阵近似和降维")
    print("-" * 50)
    
    # 使用不同数量的奇异值重构
    k_values = [1, 2, 3]
    
    print("使用前 k 个奇异值重构矩阵：")
    for k in k_values:
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - A_k)
        print(f"  k={k}: 重构误差 = {error:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, k in enumerate(k_values):
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        # 原矩阵
        if idx == 0:
            axes[0, idx].imshow(A, cmap='viridis')
            axes[0, idx].set_title('原矩阵 A', fontsize=12)
        else:
            axes[0, idx].imshow(A_k, cmap='viridis')
            axes[0, idx].set_title(f'k={k} 个奇异值重构', fontsize=12)
        
        axes[0, idx].axis('off')
        
        # 误差矩阵
        error_matrix = np.abs(A - A_k)
        im = axes[1, idx].imshow(error_matrix, cmap='hot')
        axes[1, idx].set_title(f'误差 (L2={np.linalg.norm(error_matrix):.2f})', fontsize=12)
        axes[1, idx].axis('off')
        
    plt.colorbar(im, ax=axes[1, :], orientation='vertical', shrink=0.6)
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/svd_reconstruction.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/svd_reconstruction.png")
    
    # 示例3：奇异值与信息量
    print("\n示例3：奇异值与信息量")
    print("-" * 50)
    
    # 计算奇异值的累计贡献率
    total_variance = np.sum(S**2)
    cumulative_variance = np.cumsum(S**2) / total_variance
    
    print(f"奇异值: {S}")
    print(f"奇异值平方: {S**2}")
    print(f"\n各奇异值的方差贡献率:")
    for i, (s, var) in enumerate(zip(S, S**2 / total_variance)):
        bar = '█' * int(var * 50)
        print(f"  σ_{i+1}² = {s**2:.2f}, 贡献率 = {var*100:.1f}% {bar}")
    
    print(f"\n累计方差贡献率:")
    for i, cum_var in enumerate(cumulative_variance):
        bar = '█' * int(cum_var * 50)
        print(f"  前 {i+1} 个奇异值: {cum_var*100:.1f}% {bar}")
    
    # 示例4：SVD 在文本分析中的应用
    print("\n示例4：潜在语义分析 (LSA) 示例")
    print("-" * 50)
    
    # 文档-词矩阵示例
    documents = [
        "人工智能 机器学习 深度学习",
        "机器学习 数据科学 算法",
        "深度学习 神经网络 计算机视觉",
        "数据科学 大数据 统计分析",
    ]
    
    # 简化：创建词-文档矩阵
    words = list(set(" ".join(documents).split()))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    doc_word_matrix = np.zeros((len(words), len(documents)))
    for j, doc in enumerate(documents):
        for word in doc.split():
            if word in word_to_idx:
                doc_word_matrix[word_to_idx[word], j] += 1
    
    # 对词-文档矩阵进行 SVD
    U, S, Vt = np.linalg.svd(doc_word_matrix, full_matrices=False)
    
    print(f"词汇表: {words}")
    print(f"词-文档矩阵形状: {doc_word_matrix.shape}")
    print(f"奇异值: {S}")
    
    # 使用前 2 个奇异值进行降维
    k = 2
    reduced_matrix = U[:, :k] @ np.diag(S[:k])
    
    print(f"\n降维后的词向量 (k=2):")
    for i, word in enumerate(words):
        vec = reduced_matrix[i]
        print(f"  {word}: [{vec[0]:.2f}, {vec[1]:.2f}]")


# ============================================================
# 4. 范数与距离
# ============================================================

def explain_norms_and_distances():
    """
    范数与距离详解
    
    范数是衡量向量或矩阵"大小"的函数。
    
    常用向量范数：
    - L1 范数：||x||_1 = Σ|x_i|（曼哈顿距离）
    - L2 范数：||x||_2 = √Σx_i²（欧几里得距离）
    - L∞ 范数：||x||_∞ = max|x_i|（最大范数）
    
    常用矩阵范数：
    - Frobenius 范数：||A||_F = √Σa_ij²
    - 谱范数：||A||_2 = σ_max(A)
    
    在 LLM 中的应用：
    - 权重衰减（正则化）
    - 梯度裁剪
    - 相似度计算
    - 模型参数约束
    """
    
    print("\n" + "=" * 70)
    print("4. 范数与距离")
    print("=" * 70)
    
    # 示例1：不同范数的计算
    print("\n示例1：向量范数计算")
    print("-" * 50)
    
    # 创建示例向量
    v = np.array([3, 4])
    
    # L1 范数（曼哈顿距离）
    l1_norm = np.sum(np.abs(v))
    
    # L2 范数（欧几里得距离）
    l2_norm = np.linalg.norm(v)
    
    # L∞ 范数（最大范数）
    linf_norm = np.max(np.abs(v))
    
    print(f"向量 v = {v}")
    print(f"  L1 范数 ||v||_1 = |3| + |4| = {l1_norm}")
    print(f"  L2 范数 ||v||_2 = √(3² + 4²) = {l2_norm:.2f}")
    print(f"  L∞ 范数 ||v||_∞ = max(|3|, |4|) = {linf_norm}")
    
    # 可视化不同范数的单位球
    print("\n示例2：不同范数的单位球")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (p, name) in enumerate([(1, 'L1 (Manhattan)'), (2, 'L2 (Euclidean)'), (np.inf, 'L∞ (Max)')]):
        theta = np.linspace(0, 2*np.pi, 200)
        
        # 参数化单位球
        if p == 1:
            # L1 范数单位球是菱形
            x = np.concatenate([np.linspace(-1, 1, 100), np.ones(100)])
            y = np.concatenate([1 - np.abs(np.linspace(-1, 1, 100)), np.ones(100) - 1])
            x = np.concatenate([x, x[::-1] * -1, x[::-1] * -1])
            y = np.concatenate([y, y[::-1], y[::-1] * -1])
        elif p == np.inf:
            # L∞ 范数单位球是方形
            x = np.array([-1, -1, 1, 1, -1])
            y = np.array([-1, 1, 1, -1, -1])
        else:
            x = np.cos(theta)
            y = np.sin(theta)
        
        axes[idx].fill(x, y, alpha=0.3, color='steelblue')
        axes[idx].plot(x, y, 'b-', linewidth=2)
        axes[idx].set_xlim(-1.5, 1.5)
        axes[idx].set_ylim(-1.5, 1.5)
        axes[idx].set_aspect('equal')
        axes[idx].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[idx].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        axes[idx].set_title(f'{name} Unit Ball', fontsize=12)
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/norm_balls.png', dpi=150, bbox_inches='tight')
    print("图片已保存: week1_theory_foundations/norm_balls.png")
    
    # 示例3：矩阵范数
    print("\n示例3：矩阵范数")
    print("-" * 50)
    
    # 创建示例矩阵
    A = np.array([[1, 2], [3, 4]])
    
    # Frobenius 范数
    frob_norm = np.linalg.norm(A, 'fro')
    
    # 谱范数
    spectral_norm = np.linalg.norm(A, 2)
    
    # 核范数（奇异值之和）
    singular_values = np.linalg.svd(A, compute_uv=False)
    nuclear_norm = np.sum(singular_values)
    
    print(f"矩阵 A:")
    print(A)
    print(f"\n  Frobenius 范数 ||A||_F = √Σa_ij² = {frob_norm:.2f}")
    print(f"  谱范数 ||A||_2 = σ_max(A) = {spectral_norm:.2f}")
    print(f"  核范数 ||A||_* = Σσ_i = {nuclear_norm:.2f}")
    
    # 示例4：词向量的相似度计算
    print("\n示例4：词向量相似度计算")
    print("-" * 50)
    
    # 模拟词嵌入
    word_embeddings = {
        "猫": np.array([0.9, 0.1, 0.2]),
        "狗": np.array([0.8, 0.15, 0.25]),
        "汽车": np.array([0.1, 0.9, 0.1]),
        "电脑": np.array([0.15, 0.85, 0.2]),
    }
    
    def cosine_similarity(a, b):
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def euclidean_distance(a, b):
        """计算欧几里得距离"""
        return np.linalg.norm(a - b)
    
    print("词向量余弦相似度:")
    print("       猫       狗      汽车     电脑")
    for word1 in ["猫", "狗", "汽车", "电脑"]:
        row = []
        for word2 in ["猫", "狗", "汽车", "电脑"]:
            sim = cosine_similarity(word_embeddings[word1], word_embeddings[word2])
            row.append(f"{sim:.2f}")
        print(f"{word1}  {row[0]:<8} {row[1]:<8} {row[2]:<8} {row[3]:<8}")
    
    print("\n解读：")
    print("  - '猫' 和 '狗' 余弦相似度高（都是动物）")
    print("  - '汽车' 和 '电脑' 余弦相似度高（都是科技产品）")
    print("  - 动物和科技产品之间相似度低")


# ============================================================
# 5. 矩阵运算在 LLM 中的应用
# ============================================================

def matrix_applications_in_llm():
    """
    矩阵运算在 LLM 中的实际应用
    """
    
    print("\n" + "=" * 70)
    print("5. 矩阵运算在 LLM 中的应用")
    print("=" * 70)
    
    # 示例1：嵌入层
    print("\n示例1：词嵌入层")
    print("-" * 50)
    
    vocab_size = 10000
    embed_dim = 768
    seq_len = 512
    
    # 嵌入矩阵
    embedding_matrix = np.random.randn(vocab_size, embed_dim) * 0.01
    
    # 输入序列（词索引）
    input_ids = np.array([123, 456, 789, 101, 202])
    
    # 嵌入查找（本质是索引选择）
    embedded = embedding_matrix[input_ids]
    
    print(f"词汇表大小: {vocab_size}")
    print(f"嵌入维度: {embed_dim}")
    print(f"序列长度: {len(input_ids)}")
    print(f"\n输入词索引: {input_ids}")
    print(f"嵌入后形状: {embedded.shape}")
    print(f"\n嵌入矩阵形状: {embedding_matrix.shape}")
    print(f"嵌入查找操作: O(序列长度)，不涉及大规模矩阵乘法")
    
    # 示例2：注意力计算
    print("\n示例2：注意力机制中的矩阵运算")
    print("-" * 50)
    
    batch_size = 2
    seq_len = 8
    d_model = 16
    num_heads = 4
    head_dim = d_model // num_heads
    
    # 模拟 Query, Key, Value 矩阵
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)
    
    # 投影矩阵
    W_Q = np.random.randn(d_model, d_model)
    W_K = np.random.randn(d_model, d_model)
    W_V = np.random.randn(d_model, d_model)
    W_O = np.random.randn(d_model, d_model)
    
    # 计算 Q, K, V
    Q_proj = np.einsum('bld,dd->bld', Q, W_Q)
    K_proj = np.einsum('bld,dd->bld', K, W_K)
    V_proj = np.einsum('bld,dd->bld', V, W_V)
    
    print(f"批次大小: {batch_size}")
    print(f"序列长度: {seq_len}")
    print(f"模型维度: {d_model}")
    print(f"注意力头数: {num_heads}")
    print(f"每头维度: {head_dim}")
    print(f"\nQ 投影后形状: {Q_proj.shape}")
    print(f"K 投影后形状: {K_proj.shape}")
    print(f"V 投影后形状: {V_proj.shape}")
    
    # 多头注意力
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    print(f"\n多头形状: {Q_heads.shape}")
    print(f"  (batch, num_heads, seq_len, head_dim)")
    
    # 示例3：LoRA 的矩阵分解思想
    print("\n示例3：LoRA 的低秩分解思想")
    print("-" * 50)
    
    # 全量微调需要更新的矩阵
    W_full = np.random.randn(768, 768)
    print(f"全量微调参数量: {W_full.shape[0]} × {W_full.shape[1]} = {W_full.size:,}")
    
    # LoRA 的低秩分解
    rank = 8
    A = np.random.randn(768, rank)  # 降维
    B = np.random.randn(rank, 768)  # 升维
    
    lora_params = A.size + B.size
    print(f"\nLoRA 参数:")
    print(f"  A 矩阵形状: {A.shape} = {A.size:,} 参数")
    print(f"  B 矩阵形状: {B.shape} = {B.size:,} 参数")
    print(f"  总参数量: {lora_params:,}")
    print(f"  参数减少: {lora_params / W_full.size * 100:.2f}%")
    print(f"\n核心思想: ΔW = B × A (低秩分解)")
    print(f"  - 不直接更新 W，而是学习低秩分解的 A 和 B")
    print(f"  - 推理时: W + ΔW = W + BA（可合并）")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("线性代数基础 - 大模型工程师必备数学知识")
    print("=" * 70)
    
    # 运行各个示例
    explain_matrix_basics()
    explain_eigen_decomposition()
    explain_svd()
    explain_norms_and_distances()
    matrix_applications_in_llm()
    
    print("\n" + "=" * 70)
    print("线性代数学习总结")
    print("=" * 70)
    print("""
核心概念速查表：

┌─────────────────────┬─────────────────────────────────────────────┐
│ 概念                │ 公式与应用                                    │
├─────────────────────┼─────────────────────────────────────────────┤
│ 矩阵乘法            │ (m,n) × (n,k) = (m,k)                        │
│                     │ 注意力计算的基础                               │
├─────────────────────┼─────────────────────────────────────────────┤
│ 特征分解            │ A = VΛV⁻¹                                    │
│                     │ 方阵的谱分析                                   │
├─────────────────────┼─────────────────────────────────────────────┤
│ 奇异值分解          │ A = UΣVᵀ                                     │
│                     │ 任意矩阵的分解，降维、压缩                      │
├─────────────────────┼─────────────────────────────────────────────┤
│ L2 范数             │ ||x||₂ = √Σxᵢ²                               │
│                     │ 权重衰减、梯度裁剪                             │
├─────────────────────┼─────────────────────────────────────────────┤
│ 余弦相似度          │ cos(θ) = (a·b)/(|a||b|)                      │
│                     │ 词向量相似度计算                              │
├─────────────────────┼─────────────────────────────────────────────┤
│ Frobenius 范数      │ ||A||_F = √Σaᵢⱼ²                             │
│                     │ 矩阵大小度量                                   │
└─────────────────────┴─────────────────────────────────────────────┘

与 LLM 的关系：
- 所有 Transformer 计算都是矩阵运算
- 注意力机制 = 矩阵乘法 + Softmax
- SVD 用于理解矩阵的低秩结构
- LoRA 等微调方法基于矩阵分解

下一步学习：
- 优化算法：梯度下降、Adam
- NLP 基础：分词、词嵌入
- 注意力机制：Q-K-V 抽象
    """)
