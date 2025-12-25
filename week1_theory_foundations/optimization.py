"""
优化算法基础 - 大模型工程师必备知识

本模块涵盖深度学习中最常用的优化算法。

核心知识点：
1. 梯度下降 (Gradient Descent) - 最基本的优化算法
2. 随机梯度下降 (SGD) - 随机采样加速训练
3. 动量 (Momentum) - 加速收敛，减少震荡
4. AdaGrad - 自适应学习率
5. RMSProp - 指数移动平均的梯度缩放
6. Adam - 最常用的自适应优化器
7. 学习率调度 - 动态调整学习率

为什么需要优化算法？
- 深度学习模型有数百万甚至数十亿参数
- 无法手动求解，需要数值优化
- 优化算法的效率直接影响训练速度和最终性能
- 大模型训练需要高效的优化策略
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


# ============================================================
# 1. 梯度下降 (Gradient Descent)
# ============================================================

def explain_gradient_descent():
    """
    梯度下降详解
    
    核心思想：沿着损失函数梯度的反方向更新参数
    
    更新公式：θ = θ - η × ∇L(θ)
    
    关键概念：
    - 梯度：损失函数对参数的导数，指向函数增长最快的方向
    - 学习率 (η)：控制每一步的步长
    - 收敛：参数更新趋于稳定，损失函数接近最小值
    
    问题：
    - 容易陷入局部最小值（但深度学习中通常不是大问题）
    - 对学习率敏感
    - 难以处理病态曲率
    """
    
    print("=" * 70)
    print("1. 梯度下降 (Gradient Descent)")
    print("=" * 70)
    
    # 示例1：可视化梯度下降
    print("\n示例1：在一维函数上演示梯度下降")
    print("-" * 50)
    
    # 定义损失函数 L(x) = x^4 - 3x^3 + 2
    def loss_function(x):
        return x**4 - 3*x**3 + 2
    
    def gradient(x):
        return 4*x**3 - 9*x**2
    
    # 梯度下降
    x = 4.0  # 初始位置
    learning_rate = 0.01
    path = [x]
    
    for _ in range(50):
        grad = gradient(x)
        x = x - learning_rate * grad
        path.append(x)
    
    print(f"初始位置: x = {path[0]}")
    print(f"学习率: η = {learning_rate}")
    print(f"\n梯度下降过程:")
    for i, x_val in enumerate(path[:10]):
        loss = loss_function(x_val)
        print(f"  步 {i:2d}: x = {x_val:7.4f}, L(x) = {loss:8.4f}")
    
    print(f"  ...")
    print(f"  最终: x = {path[-1]:7.4f}, L(x) = {loss_function(path[-1]):8.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失函数曲线
    x_range = np.linspace(-1, 4, 200)
    y_range = [loss_function(x) for x in x_range]
    
    axes[0].plot(x_range, y_range, 'b-', linewidth=2, label='L(x) = x⁴ - 3x³ + 2')
    axes[0].plot(path, [loss_function(x) for x in path], 'r.-', markersize=10, label='Gradient Descent Path')
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('L(x)', fontsize=12)
    axes[0].set_title('Gradient Descent on 1D Function', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 损失曲线
    losses = [loss_function(x) for x in path]
    axes[1].plot(losses, 'b-', linewidth=2)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss Curve', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/gradient_descent_1d.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/gradient_descent_1d.png")
    
    # 示例2：学习率的影响
    print("\n示例2：学习率对梯度下降的影响")
    print("-" * 50)
    
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.15]
    
    print(f"{'学习率 η':<12} {'收敛情况':<30} {'最终损失'}")
    print("-" * 60)
    
    for lr in learning_rates:
        x = 4.0
        path_temp = [x]
        
        for _ in range(100):
            grad = gradient(x)
            x = x - lr * grad
            path_temp.append(x)
        
        final_loss = loss_function(x)
        
        if lr < 0.01:
            status = "太慢，收敛困难"
        elif lr < 0.05:
            status = "适中，快速收敛"
        elif lr < 0.1:
            status = "良好，最优收敛"
        elif lr < 0.15:
            status = "震荡，可能发散"
        else:
            status = "发散"
        
        print(f"{lr:<12} {status:<30} {final_loss:.4f}")


# ============================================================
# 2. 随机梯度下降 (SGD)
# ============================================================

def explain_sgd():
    """
    随机梯度下降详解
    
    问题：批量梯度下降在每次迭代中需要计算所有样本的梯度，计算代价高。
    
    解决方案：每次只用一个样本（或一小批样本）来估计梯度。
    
    优点：
    - 计算快，适合大规模数据
    - 引入随机性，有助于跳出局部最小值
    - 在线学习能力
    
    缺点：
    - 梯度估计有噪声，收敛路径震荡
    - 可能需要更多的迭代次数
    
    小批量 SGD：
    - 平衡计算效率和梯度估计准确性
    - 通常 batch_size = 32, 64, 128, 256
    """
    
    print("\n" + "=" * 70)
    print("2. 随机梯度下降 (SGD)")
    print("=" * 70)
    
    # 示例1：模拟小批量 SGD
    print("\n示例1：小批量 SGD 原理")
    print("-" * 50)
    
    # 模拟一个简单的数据集
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 2)
    true_w = np.array([2.0, -1.5])
    true_b = 0.5
    y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1
    
    # 损失函数：均方误差
    def compute_loss(w, b, X, y):
        predictions = X @ w + b
        return np.mean((predictions - y) ** 2)
    
    # 梯度计算
    def compute_gradient(w, b, X, y):
        n = len(y)
        predictions = X @ w + b
        errors = predictions - y
        grad_w = (2/n) * X.T @ errors
        grad_b = (2/n) * np.sum(errors)
        return grad_w, grad_b
    
    # 批量梯度下降
    w_batch = np.zeros(2)
    b_batch = 0
    lr = 0.1
    
    for i in range(100):
        grad_w, grad_b = compute_gradient(w_batch, b_batch, X, y)
        w_batch = w_batch - lr * grad_w
        b_batch = b_batch - lr * grad_b
    
    loss_batch = compute_loss(w_batch, b_batch, X, y)
    
    # 小批量 SGD
    w_sgd = np.zeros(2)
    b_sgd = 0
    batch_size = 32
    
    np.random.seed(42)
    loss_sgd = []
    
    for i in range(1000):
        # 随机选择小批量
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        
        grad_w, grad_b = compute_gradient(w_sgd, b_sgd, X_batch, y_batch)
        w_sgd = w_sgd - lr * grad_w
        b_sgd = b_sgd - lr * grad_b
        
        if i % 100 == 0:
            loss_sgd.append(compute_loss(w_sgd, b_sgd, X, y))
    
    print(f"数据集大小: {n_samples} 样本")
    print(f"特征维度: {X.shape[1]}")
    print(f"批量大小: {batch_size}")
    print(f"\n真实参数: w = {true_w}, b = {true_b}")
    print(f"\n批量梯度下降 (100次迭代):")
    print(f"  估计参数: w = {w_batch.round(4)}, b = {b_batch:.4f}")
    print(f"  最终损失: {loss_batch:.6f}")
    print(f"\n小批量 SGD (1000次迭代):")
    print(f"  估计参数: w = {w_sgd.round(4)}, b = {b_sgd:.4f}")
    print(f"  最终损失: {loss_sgd[-1]:.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 批量 SGD 损失曲线
    axes[0].plot(loss_sgd, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch (×100)', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Mini-batch SGD Loss Curve', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # 拟合结果
    x_test = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_true = true_w[0] * x_test + true_w[1] * 0 + true_b
    y_batch_fit = w_batch[0] * x_test + w_batch[1] * 0 + b_batch
    y_sgd_fit = w_sgd[0] * x_test + w_sgd[1] * 0 + b_sgd
    
    axes[1].scatter(X[:, 0], y, alpha=0.3, s=10, label='Data')
    axes[1].plot(x_test, y_true, 'g-', linewidth=2, label='True Model')
    axes[1].plot(x_test, y_batch_fit, 'r--', linewidth=2, label='Batch GD')
    axes[1].plot(x_test, y_sgd_fit, 'b:', linewidth=2, label='SGD')
    axes[1].set_xlabel('X₁', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].set_title('Model Fitting Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/sgd_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/sgd_comparison.png")


# ============================================================
# 3. 动量 (Momentum)
# ============================================================

def explain_momentum():
    """
    动量 (Momentum) 详解
    
    核心思想：模仿物理中的动量，积累历史梯度信息来加速收敛。
    
    更新公式：
    v = γ × v + η × ∇L(θ)
    θ = θ - v
    
    其中：
    - v 是速度向量
    - γ 是动量系数（通常 0.9 或 0.99）
    - η 是学习率
    
    优点：
    - 加速收敛
    - 减少震荡
    - 有助于跳出局部最小值
    - 处理病态曲率
    """
    
    print("\n" + "=" * 70)
    print("3. 动量 (Momentum)")
    print("=" * 70)
    
    # 示例1：对比 SGD 和 Momentum
    print("\n示例1：SGD vs Momentum")
    print("-" * 50)
    
    # 定义二维损失函数
    def loss_2d(w1, w2):
        return w1**2 + 10*w2**2
    
    def grad_2d(w1, w2):
        return np.array([2*w1, 20*w2])
    
    # SGD
    w_sgd = np.array([4.0, 1.0])
    lr = 0.05
    path_sgd = [w_sgd.copy()]
    
    for _ in range(50):
        grad = grad_2d(*w_sgd)
        w_sgd = w_sgd - lr * grad
        path_sgd.append(w_sgd.copy())
    
    # Momentum
    w_mom = np.array([4.0, 1.0])
    v = np.zeros(2)
    gamma = 0.9
    path_mom = [w_mom.copy()]
    
    for _ in range(50):
        grad = grad_2d(*w_mom)
        v = gamma * v + lr * grad
        w_mom = w_mom - v
        path_mom.append(w_mom.copy())
    
    print(f"初始位置: w = [4.0, 1.0]")
    print(f"学习率: η = {lr}, 动量系数: γ = {gamma}")
    print(f"\n{'方法':<15} {'最终位置':<20} {'最终损失':<12}")
    print("-" * 50)
    print(f"{'SGD':<15} {f'[{w_sgd[0]:.4f}, {w_sgd[1]:.4f}]':<20} {loss_2d(*w_sgd):.6f}")
    print(f"{'Momentum':<15} {f'[{w_mom[0]:.4f}, {w_mom[1]:.4f}]':<20} {loss_2d(*w_mom):.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 损失函数等高线
    w1_range = np.linspace(-5, 5, 100)
    w2_range = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = loss_2d(W1, W2)
    
    axes[0].contour(W1, W2, Z, levels=30, cmap='viridis')
    axes[0].plot([p[0] for p in path_sgd], [p[1] for p in path_sgd], 
                 'r.-', markersize=10, linewidth=2, label='SGD')
    axes[0].plot([p[0] for p in path_mom], [p[1] for p in path_mom], 
                 'b.-', markersize=10, linewidth=2, label='Momentum')
    axes[0].scatter([0], [0], c='red', s=200, marker='*', label='Minimum')
    axes[0].set_xlabel('w₁', fontsize=12)
    axes[0].set_ylabel('w₂', fontsize=12)
    axes[0].set_title('SGD vs Momentum Trajectory', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 损失曲线对比
    losses_sgd = [loss_2d(*p) for p in path_sgd]
    losses_mom = [loss_2d(*p) for p in path_mom]
    
    axes[1].semilogy(losses_sgd, 'r-', linewidth=2, label='SGD')
    axes[1].semilogy(losses_mom, 'b-', linewidth=2, label='Momentum')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('Loss Curve Comparison', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/momentum_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/momentum_comparison.png")


# ============================================================
# 4. Adam 优化器
# ============================================================

def explain_adam():
    """
    Adam (Adaptive Moment Estimation) 详解
    
    核心思想：结合动量和自适应学习率的优点。
    
    更新步骤：
    1. 计算梯度：g = ∇L(θ)
    2. 更新一阶矩估计（动量）：m = β₁ × m + (1 - β₁) × g
    3. 更新二阶矩估计（自适应）：v = β₂ × v + (1 - β₂) × g²
    4. 偏差校正：m̂ = m / (1 - β₁^t), v̂ = v / (1 - β₂^t)
    5. 更新参数：θ = θ - η × m̂ / (√v̂ + ε)
    
    默认超参数：
    - β₁ = 0.9（一阶矩衰减）
    - β₂ = 0.999（二阶矩衰减）
    - ε = 1e-8（数值稳定）
    - η = 0.001（学习率）
    
    为什么 Adam 如此流行？
    - 自适应学习率，对超参数不敏感
    - 结合动量优点，收敛快
    - 很少需要调参
    - 在大多数任务上效果良好
    """
    
    print("\n" + "=" * 70)
    print("4. Adam 优化器")
    print("=" * 70)
    
    # 示例1：实现简单的 Adam
    print("\n示例1：从零实现 Adam")
    print("-" * 50)
    
    def simple_adam(grad_func, init_theta, lr=0.01, n_iterations=100,
                    beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        简化的 Adam 实现
        """
        theta = init_theta.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        path = [theta.copy()]
        
        for t in range(1, n_iterations + 1):
            grad = grad_func(theta)
            
            # 更新一阶和二阶矩估计
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # 偏差校正
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # 更新参数
            theta = theta - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            path.append(theta.copy())
        
        return theta, path
    
    # 测试函数
    def rosenbrock_grad(theta):
        """Rosenbrock 函数的梯度"""
        x, y = theta
        grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
        grad_y = 200 * (y - x**2)
        return np.array([grad_x, grad_y])
    
    def rosenbrock(theta):
        x, y = theta
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    # 使用 Adam 优化
    init_theta = np.array([-1.0, 1.0])
    final_theta, path = simple_adam(rosenbrock_grad, init_theta, lr=0.001, n_iterations=2000)
    
    print(f"测试函数: Rosenbrock 函数")
    print(f"  f(x,y) = (1-x)² + 100(y-x²)²")
    print(f"  全局最小: (1, 1), f(1,1) = 0")
    print(f"\n初始位置: {init_theta}")
    print(f"Adam 优化后: {final_theta.round(4)}")
    print(f"最终损失: {rosenbrock(final_theta):.6f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失函数等高线
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock([X, Y])
    
    path = np.array(path)
    axes[0].contour(X, Y, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
    axes[0].plot(path[:, 0], path[:, 1], 'r.-', markersize=2, linewidth=1, alpha=0.7)
    axes[0].scatter([1], [1], c='red', s=200, marker='*', zorder=5)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-1, 3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Adam Trajectory on Rosenbrock')
    
    # 损失曲线
    losses = [rosenbrock(t) for t in path]
    axes[1].semilogy(losses, 'b-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Adam Loss Curve')
    axes[1].grid(alpha=0.3)
    
    # 各优化器对比
    # 重新实现 SGD 和 Momentum
    def sgd(grad_func, init_theta, lr, n_iter):
        theta = init_theta.copy()
        path = [theta.copy()]
        for _ in range(n_iter):
            grad = grad_func(theta)
            theta = theta - lr * grad
            path.append(theta.copy())
        return theta, path
    
    def momentum(grad_func, init_theta, lr, n_iter, beta=0.9):
        theta = init_theta.copy()
        v = np.zeros_like(theta)
        path = [theta.copy()]
        for _ in range(n_iter):
            grad = grad_func(theta)
            v = beta * v + lr * grad
            theta = theta - v
            path.append(theta.copy())
        return theta, path
    
    init = np.array([-1.0, 1.0])
    _, path_sgd = sgd(rosenbrock_grad, init, 0.001, 2000)
    _, path_mom = momentum(rosenbrock_grad, init, 0.001, 2000)
    
    losses_sgd = [rosenbrock(t) for t in path_sgd]
    losses_mom = [rosenbrock(t) for t in path_mom]
    losses_adam = [rosenbrock(t) for t in path]
    
    axes[2].semilogy(losses_sgd, 'r-', linewidth=2, label='SGD', alpha=0.7)
    axes[2].semilogy(losses_mom, 'g-', linewidth=2, label='Momentum', alpha=0.7)
    axes[2].semilogy(losses_adam, 'b-', linewidth=2, label='Adam')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss (log scale)')
    axes[2].set_title('Optimizer Comparison')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/adam_optimizer.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/adam_optimizer.png")
    
    # 示例2：Adam 的偏差校正
    print("\n示例2：Adam 的偏差校正演示")
    print("-" * 50)
    
    t = np.arange(1, 21)
    beta1, beta2 = 0.9, 0.999
    
    uncorrected_m = beta1 ** t
    corrected_m = 1 - beta1 ** t
    
    print("迭代次数 t | 未校正 m̂=m/(1-β₁^t) | 已校正")
    print("-" * 50)
    for i in [1, 5, 10, 20]:
        print(f"  t={i:2d}   | {uncorrected_m[i-1]:.4f}              | {corrected_m[i-1]:.4f}")
    
    print("\n解读：")
    print("  - 初期迭代（t 很小）时，偏差校正很重要")
    print("  - 随着迭代次数增加，校正因子的影响减小")
    print("  - 如果不进行校正，前几次更新的梯度会被放大")


# ============================================================
# 5. 学习率调度
# ============================================================

def explain_learning_rate_scheduling():
    """
    学习率调度详解
    
    核心思想：在训练过程中动态调整学习率。
    
    常用策略：
    1. Step Decay：每隔固定步数降低学习率
    2. Exponential Decay：指数衰减
    3. Cosine Annealing：余弦退火
    4. Warmup：初期逐步增加学习率
    5. Reduce on Plateau：损失停滞时降低学习率
    
    在大模型训练中的重要性：
    - 预训练通常使用 Cosine Annealing + Warmup
    - 微调常用 Cosine 或 Step Decay
    - 学习率对最终性能影响很大
    """
    
    print("\n" + "=" * 70)
    print("5. 学习率调度")
    print("=" * 70)
    
    # 示例1：不同调度策略
    print("\n示例1：不同学习率调度策略")
    print("-" * 50)
    
    n_steps = 1000
    initial_lr = 0.01
    
    # 1. Step Decay
    def step_decay(step, decay_rate=0.5, decay_steps=100):
        return initial_lr * (decay_rate ** (step // decay_steps))
    
    # 2. Exponential Decay
    def exp_decay(step, k=0.005):
        return initial_lr * np.exp(-k * step)
    
    # 3. Cosine Annealing
    def cosine_annealing(step, T_max=1000, eta_min=0):
        return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * step / T_max)) / 2
    
    # 4. Warmup + Cosine
    def warmup_cosine(step, warmup_steps=100, T_max=1000, eta_min=0):
        if step < warmup_steps:
            return initial_lr * step / warmup_steps
        else:
            return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * (step - warmup_steps) / (T_max - warmup_steps))) / 2
    
    steps = np.arange(n_steps)
    
    lrs = {
        'Step Decay': [step_decay(s) for s in steps],
        'Exponential': [exp_decay(s) for s in steps],
        'Cosine Annealing': [cosine_annealing(s) for s in steps],
        'Warmup + Cosine': [warmup_cosine(s) for s in steps],
    }
    
    print(f"初始学习率: {initial_lr}")
    print(f"总迭代次数: {n_steps}")
    print(f"\n{'策略':<20} {'最终学习率':<15} {'特点'}")
    print("-" * 60)
    print(f"{'Step Decay':<20} {lrs['Step Decay'][-1]:<15.6f} 简单，适合传统网络")
    print(f"{'Exponential':<20} {lrs['Exponential'][-1]:<15.6f} 平滑衰减")
    print(f"{'Cosine Annealing':<20} {lrs['Cosine Annealing'][-1]:<15.6f} 理论最优收敛")
    print(f"{'Warmup + Cosine':<20} {lrs['Warmup + Cosine'][-1]:<15.6f} 大模型训练标配")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for idx, (name, lr_values) in enumerate(lrs.items()):
        ax = axes[idx // 2, idx % 2]
        ax.plot(steps, lr_values, color=colors[idx], linewidth=2)
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'{name}', fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, initial_lr * 1.1)
        
        # 标记关键点
        if name == 'Warmup + Cosine':
            ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
            ax.text(105, initial_lr * 0.5, 'Warmup\nEnd', fontsize=10)
    
    plt.suptitle('Learning Rate Scheduling Strategies', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('week1_theory_foundations/learning_rate_scheduling.png', dpi=150, bbox_inches='tight')
    print("\n图片已保存: week1_theory_foundations/learning_rate_scheduling.png")
    
    # 示例2：学习率对训练的影响
    print("\n示例2：学习率对训练的影响")
    print("-" * 50)
    
    def train_with_lr(lr, n_epochs=100):
        """模拟不同学习率的训练"""
        x = 4.0
        losses = []
        for _ in range(n_epochs):
            # 损失函数 L(x) = x^4 - 3x^3 + 2
            loss = x**4 - 3*x**3 + 2
            grad = 4*x**3 - 9*x**2
            x = x - lr * grad
            losses.append(loss)
        return losses
    
    learning_rates = [0.001, 0.005, 0.01, 0.02]
    
    print(f"初始位置: x = 4.0")
    print(f"目标: 找到全局最小值 x ≈ 2.25")
    print(f"\n{'学习率':<12} {'是否收敛':<12} {'收敛速度'}")
    print("-" * 40)
    
    for lr in learning_rates:
        losses = train_with_lr(lr)
        final_x = 4.0
        for _ in range(100):
            grad = 4*final_x**3 - 9*final_x**2
            final_x = final_x - lr * grad
        
        converged = "✓ 收敛" if abs(final_x - 2.25) < 0.1 else "✗ 发散/震荡"
        speed = "快" if losses[-1] < 0.1 else ("中" if losses[-1] < 1 else "慢")
        
        print(f"{lr:<12} {converged:<12} {speed}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("优化算法基础 - 大模型工程师必备知识")
    print("=" * 70)
    
    # 运行各个示例
    explain_gradient_descent()
    explain_sgd()
    explain_momentum()
    explain_adam()
    explain_learning_rate_scheduling()
    
    print("\n" + "=" * 70)
    print("优化算法学习总结")
    print("=" * 70)
    print("""
核心概念速查表：

┌─────────────────────┬─────────────────────────────────────────────┐
│ 优化器              │ 公式与特点                                   │
├─────────────────────┼─────────────────────────────────────────────┤
│ SGD                 │ θ = θ - η∇L(θ)                              │
│                     │ 简单，噪声大，需要调参                        │
├─────────────────────┼─────────────────────────────────────────────┤
│ Momentum            │ v = γv + η∇L(θ), θ = θ - v                  │
│                     │ 加速收敛，减少震荡                            │
├─────────────────────┼─────────────────────────────────────────────┤
│ Adam                │ m, v 自适应，θ = θ - η·m̂/√v̂ + ε            │
│                     │ 最常用，自适应学习率                          │
├─────────────────────┼─────────────────────────────────────────────┤
│ AdamW               │ Adam + 权重衰减分离                          │
│                     │ LLM 训练推荐                                  │
└─────────────────────┴─────────────────────────────────────────────┘

Adam 默认超参数：
- 学习率 η = 0.001
- β₁ = 0.9（一阶矩）
- β₂ = 0.999（二阶矩）
- ε = 1e-8

学习率调度策略：
1. Warmup：前几百步逐步增加学习率
2. Cosine Annealing：余弦曲线衰减
3. 典型配置：Warmup → Cosine → 线性衰减

与 LLM 的关系：
- 训练大模型通常使用 AdamW
- 需要配合学习率调度（特别是 Warmup）
- 梯度裁剪防止梯度爆炸
- 混合精度训练加速

下一步学习：
- NLP 基础：分词、词嵌入
- 注意力机制：Q-K-V 抽象
- Transformer 架构
    """)
