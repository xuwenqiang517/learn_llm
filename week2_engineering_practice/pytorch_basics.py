"""
PyTorch基础教程 - Week2 工程实践

本模块涵盖PyTorch框架的核心概念和操作，包括：
1. 张量(Tensor)创建、操作和变形
2. 自动求导(Autograd)机制
3. 神经网络模块(nn.Module)构建
4. 损失函数和优化器使用
5. 完整训练循环实现
6. GPU加速和混合精度训练基础

Author: learn_llm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import time
import math


class TensorOperations:
    """
    张量操作类 - 演示PyTorch张量的各种操作
    
    张量是PyTorch的核心数据结构，是NumPy数组的GPU加速版本。
    在大模型中，张量用于表示：
    - 输入文本的token嵌入 (batch_size, seq_len, hidden_size)
    - 模型权重和参数
    - 中间激活值和梯度
    """
    
    @staticmethod
    def create_tensors():
        """演示不同方式创建张量"""
        
        # 方式1: 从Python列表创建
        tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        print(f"从列表创建: {tensor_from_list.shape}")
        
        # 方式2: 使用特定函数创建
        zeros = torch.zeros(2, 3)  # 全零张量
        ones = torch.ones(2, 3)    # 全一张量
        arange = torch.arange(0, 10, 2)  # 等差序列
        linspace = torch.linspace(0, 1, 5)  # 等分序列
        
        # 方式3: 从NumPy数组创建（共享内存）
        np_array = np.array([[1, 2], [3, 4]])
        tensor_from_np = torch.from_numpy(np_array)
        # 修改np_array会影响tensor_from_np
        tensor_clone = torch.from_numpy(np_array.copy())  # 独立副本
        
        # 方式4: 在指定设备上创建
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_gpu = torch.ones(2, 2, device=device)
        
        # 方式5: 随机张量
        rand = torch.rand(2, 3)        # [0, 1)均匀分布
        randn = torch.randn(2, 3)      # 标准正态分布
        randint = torch.randint(0, 10, (2, 3))  # 整数均匀分布
        
        return {
            'zeros': zeros, 'ones': ones, 'arange': arange,
            'linspace': linspace, 'rand': rand, 'randn': randn
        }
    
    @staticmethod
    def tensor_operations_demo():
        """演示张量基本运算"""
        
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        
        # 逐元素运算
        add = a + b
        multiply = a * b  # 逐元素相乘
        matmul = a @ b    # 矩阵乘法
        
        # 维度操作
        tensor_3d = torch.randn(2, 3, 4)
        # view/reshape: 改变张量形状（注意内存连续性）
        flat = tensor_3d.view(-1)  # -1表示自动推断维度
        reshaped = tensor_3d.reshape(2, 12)
        
        # squeeze/unsqueeze: 移除/添加维度
        squeezed = tensor_3d.squeeze(0)  # 移除第0维（如果该维为1）
        unsqueezed = tensor_3d.unsqueeze(1)  # 在第1维前插入新维度
        
        # permute/transpose: 维度重排
        permuted = tensor_3d.permute(2, 0, 1)  # 重新排列维度
        transposed = tensor_3d.transpose(0, 1)  # 交换两个维度
        
        # 索引和切片
        first_dim = tensor_3d[:, 0, :]  # 第一个batch的所有序列的第一个token
        
        # 聚合操作
        mean_val = tensor_3d.mean()
        sum_val = tensor_3d.sum(dim=1)  # 按维度1求和
        max_val = tensor_3d.max(dim=-1)  # 最大值及索引
        
        print(f"矩阵乘法结果:\n{matmul}")
        print(f"形状变换 - 原形状: {tensor_3d.shape}, 展平后: {flat.shape}")
        print(f"按维度求和结果形状: {sum_val.shape}")
        
        return add, multiply, matmul
    
    @staticmethod
    def tensor_indexing_advanced():
        """高级索引和切片技巧"""
        
        # 创建示例张量: (batch=4, seq_len=8, hidden=16)
        x = torch.randn(4, 8, 16)
        
        # 基础索引
        batch_0 = x[0]           # 第一批次
        seq_0_to_3 = x[0, :4]    # 第一批次的前4个位置
        hidden_0_to_7 = x[0, :, :8]  # 第一批次的所有位置的前8维
        
        # 高级索引
        indices = torch.tensor([0, 2])
        selected_batches = x[indices]  # 选择指定批次
        
        # 布尔掩码索引
        mask = x > 0.5
        masked_values = x[mask]  # 只保留大于0.5的值
        
        # 使用where进行条件选择
        condition = torch.randn(4, 8) > 0
        result = torch.where(condition, x, torch.zeros_like(x))
        
        # gather和scatter（重要用于输出选择）
        # 假设我们有一个token分类任务，需要从hidden states中选择特定位置
        positions = torch.tensor([[1, 2], [3, 4], [0, 1], [2, 3]])
        batch_indices = torch.arange(4).unsqueeze(1).expand(-1, 2)
        gathered = x[batch_indices, positions]
        
        # index_select: 按索引从指定维度选择
        selected_indices = torch.tensor([0, 2, 4])
        selected = torch.index_select(x, dim=1, index=selected_indices)
        
        return gathered
    
    @staticmethod
    def tensor_dtype_conversion():
        """数据类型转换和精度管理"""
        
        # PyTorch默认使用float32
        tensor_float32 = torch.randn(3, 3)
        print(f"默认数据类型: {tensor_float32.dtype}")
        
        # 转换为float16（用于混合精度训练）
        tensor_float16 = tensor_float32.half()
        print(f"Float16: {tensor_float16.dtype}")
        
        # 转换为float64（用于需要高精度的计算）
        tensor_float64 = tensor_float32.double()
        print(f"Float64: {tensor_float64.dtype}")
        
        # 转换为int（用于分类标签等）
        tensor_int = tensor_float32.long()
        print(f"Long: {tensor_int.dtype}")
        
        # 模型参数通常使用float32
        # 嵌入层可能使用float16以节省显存
        # 计算梯度时float32是稳定的
        
        # 查看当前设备的默认精度
        print(f"当前设备: {tensor_float32.device}")
        
        # 确保模型和数据在相同设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_on_device = tensor_float32.to(device)
        print(f"转换到设备: {model_on_device.device}")


class AutogradMechanism:
    """
    自动求导机制类 - 演示PyTorch的autograd系统
    
    Autograd是PyTorch的核心特性，允许自动计算梯度。
    通过设置requires_grad=True来追踪张量的操作历史。
    
    关键概念:
    - requires_grad: 是否追踪该张量的梯度
    - grad_fn: 创建该张量的函数（用于反向传播）
    - backward(): 执行反向传播计算梯度
    - grad: 存储计算得到的梯度值
    """
    
    @staticmethod
    def basic_autograd():
        """基础自动求导演示"""
        
        # 创建需要梯度的张量
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        
        # 执行计算（PyTorch会自动构建计算图）
        z = x ** 2 + 2 * y + 1
        
        # 反向传播
        z.backward()
        
        # 获取梯度
        print(f"z = {z.item()}")
        print(f"dz/dx = {x.grad}")
        print(f"dz/dy = {y.grad}")
        
        # 验证: dz/dx = 2x = 4, dz/dy = 2
        
        return x.grad, y.grad
    
    @staticmethod
    def autograd_graph():
        """理解计算图结构"""
        
        # 创建叶子节点
        a = torch.tensor([2.0, 3.0], requires_grad=True)
        b = torch.tensor([1.0, 2.0], requires_grad=True)
        
        # 构建计算图
        c = a * b  # 乘法节点
        d = c.sum()  # 求和节点
        
        # 查看计算图
        print(f"a是叶子节点: {a.is_leaf}")
        print(f"c的grad_fn: {c.grad_fn}")
        print(f"d的grad_fn: {d.grad_fn}")
        
        # 反向传播
        d.backward()
        
        # 梯度流动
        print(f"a.grad: {a.grad}")
        print(f"b.grad: {b.grad}")
        
        # 计算图保持机制
        # 默认情况下，backward()后会释放计算图
        # 如果需要多次backward，需要 retain_graph=True
        e = a.sum()
        e.backward(retain_graph=True)
        print(f"再次反向传播后的a.grad: {a.grad}")
        
        return a.grad
    
    @staticmethod
    def autograd_gradients():
        """梯度的高级操作"""
        
        # 创建向量输出函数的梯度
        x = torch.randn(3, requires_grad=True)
        
        # y是向量，backward需要gradient参数
        y = x ** 2
        y.backward(torch.tensor([1.0, 0.5, 0.25]))
        
        print(f"dy/dx (带权重): {x.grad}")
        
        # 使用torch.autograd.grad计算梯度（更灵活）
        x = torch.tensor(2.0, requires_grad=True)
        y = x ** 3 + 2 * x
        
        # 计算高阶导数
        first_grad = torch.autograd.grad(y, x, create_graph=True)
        print(f"一阶导数 dy/dx = {first_grad[0]}")
        
        # 二阶导数
        second_grad = torch.autograd.grad(first_grad[0], x)
        print(f"二阶导数 d²y/dx² = {second_grad[0]}")
        
        # 使用gradcheck验证梯度计算（数值梯度 vs 计算图梯度）
        def func(x):
            return x ** 2 + torch.sin(x)
        
        x_test = torch.randn(5, requires_grad=True)
        gradcheck_result = torch.autograd.gradcheck(func, x_test)
        print(f"梯度检验通过: {gradcheck_result}")
        
        return first_grad[0], second_grad[0]
    
    @staticmethod
    def disable_grad():
        """控制梯度计算"""
        
        x = torch.randn(3, requires_grad=True)
        y = x ** 2
        
        # 方法1: 使用torch.no_grad()上下文
        with torch.no_grad():
            z = y * 2
            print(f"在no_grad中: {z.requires_grad}")
        
        # 方法2: 使用detach()分离张量
        detached = y.detach()
        print(f"分离后: {detached.requires_grad}")
        
        # 方法3: 设置requires_grad=False
        y_no_grad = y.requires_grad_(False)
        
        # 在推理时使用no_grad可以节省内存和计算
        # detach()用于将需要梯度的张量从计算图中分离
        
        return detached


class NeuralNetworkModule(nn.Module):
    """
    神经网络模块类 - 演示nn.Module的使用
    
    nn.Module是所有神经网络模块的基类。
    它提供了参数管理、模型保存/加载、设备管理等功能。
    
    重要方法:
    - __init__: 初始化网络层
    - forward: 定义前向传播
    - parameters(): 返回所有参数
    - to(device): 移动到指定设备
    - train()/eval(): 设置训练/评估模式
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        初始化神经网络
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出层维度
        """
        super().__init__()
        
        # 方式1: 使用nn.Linear（推荐）
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Dropout层（正则化）
        self.dropout = nn.Dropout(0.2)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播定义
        
        Args:
            x: 输入张量，形状 (batch_size, input_size)
            
        Returns:
            输出张量，形状 (batch_size, output_size)
        """
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        return x
    
    def initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AdvancedNeuralNetwork(nn.Module):
    """
    高级神经网络示例 - 展示更复杂的架构
    
    包含:
    - 残差连接
    - 多头注意力（简化版）
    - 层归一化
    """
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, 1)
        
    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: 输入序列 (batch_size, seq_len, d_model)
            src_key_padding_mask: 填充位置掩码 (batch_size, seq_len)
        """
        output = self.transformer_encoder(
            src, 
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.fc(output)
        return output


class TrainingPipeline:
    """
    完整训练管道类 - 演示标准训练循环
    
    包含:
    - 数据准备
    - 模型初始化
    - 训练循环
    - 验证评估
    - 模型保存
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            
            total_loss += criterion(output, target).item()
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int, lr: float = 0.001):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
        """
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print(f"开始在{self.device}上训练...")
        print(f"总epochs: {epochs}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_acc:.4f}')
            
            # 保存最佳模型
            if val_acc >= max(self.val_losses):
                self.save_checkpoint('best_model.pth')
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, filepath: str):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到 {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"检查点已从 {filepath} 加载")


class OptimizerDemo:
    """
    优化器演示类 - 展示不同优化器的使用和特点
    """
    
    @staticmethod
    def compare_optimizers():
        """比较不同优化器"""
        
        # 创建简单的测试函数
        def rosenbrock(x):
            return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # 初始点
        x0 = torch.tensor([-3.0, -4.0], requires_grad=True)
        
        # 测试不同优化器
        optimizers = {
            'SGD': lambda: optim.SGD([x0.clone().requires_grad_(True)], lr=0.001),
            'Momentum': lambda: optim.SGD([x0.clone().requires_grad_(True)], lr=0.01, momentum=0.9),
            'Adam': lambda: optim.Adam([x0.clone().requires_grad_(True)], lr=0.1),
            'AdamW': lambda: optim.AdamW([x0.clone().requires_grad_(True)], lr=0.1, weight_decay=0.01)
        }
        
        results = {}
        
        for name, opt_factory in optimizers.items():
            optimizer = opt_factory()
            x = x0.clone().requires_grad_(True)
            losses = []
            
            for i in range(100):
                optimizer.zero_grad()
                loss = rosenbrock(x)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            results[name] = {
                'final_loss': losses[-1],
                'loss_curve': losses
            }
            print(f"{name}: 最终损失 = {losses[-1]:.4f}")
        
        return results
    
    @staticmethod
    def learning_rate_scheduling():
        """学习率调度策略"""
        
        # Step Decay
        step_scheduler = optim.lr_scheduler.StepLR(
            optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=1.0),
            step_size=30, gamma=0.1
        )
        
        # Exponential Decay
        exp_scheduler = optim.lr_scheduler.ExponentialLR(
            optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=1.0),
            gamma=0.95
        )
        
        # Cosine Annealing
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optim.SGD([torch.tensor(1.0, requires_grad=True)], lr=1.0),
            T_max=100
        )
        
        # Warmup + Cosine (大模型训练常用)
        def warmup_cosine_lr(step):
            warmup_steps = 1000
            total_steps = 10000
            
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        # 打印学习率变化
        print("学习率调度示例 (前10个step):")
        for step in range(10):
            lr = warmup_cosine_lr(step * 100)
            print(f"Step {step * 100}: lr = {lr:.6f}")


class GPUMemoryManagement:
    """
    GPU内存管理类 - 展示大模型训练中的内存优化技巧
    """
    
    @staticmethod
    def check_gpu_memory():
        """检查GPU内存使用情况"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            print(f"GPU: {torch.cuda.get_device_name(device)}")
            print(f"总显存: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
            print(f"已分配显存: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"缓存显存: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        else:
            print("CUDA不可用")
    
    @staticmethod
    @torch.no_grad()
    def memory_efficient_inference(model, input_tensor):
        """
        内存高效推理
        
        技巧:
        1. 使用torch.no_grad()避免计算梯度
        2. 梯度检查点(Gradient Checkpointing)节省显存
        3. 混合精度推理
        """
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        return output
    
    @staticmethod
    def gradient_checkpointing_example():
        """
        梯度检查点演示
        
        梯度检查点通过在前向传播时不保存所有中间激活值，
        而是在反向传播时重新计算来节省显存。
        以计算换空间的策略。
        """
        from torch.utils.checkpoint import checkpoint, checkpoint_sequential
        
        class CheckpointedModel(nn.Module):
            def __init__(self, num_layers=12):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(768, 768) for _ in range(num_layers)
                ])
                self.layer_norm = nn.LayerNorm(768)
            
            def forward(self, x):
                # 对每个层使用检查点
                for i, layer in enumerate(self.layers):
                    x = checkpoint(layer, x) if i % 2 == 0 else layer(x)
                return self.layer_norm(x)
        
        return CheckpointedModel()


class MixedPrecisionTraining:
    """
    混合精度训练类 - 使用FP16减少显存和提高训练速度
    
    混合精度训练通过以下方式加速训练:
    1. 主权重保持为FP32（数值稳定性）
    2. 前向/反向传播使用FP16（计算加速）
    3. 梯度累积到FP32（避免梯度消失）
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        # 缩放器用于梯度缩放
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, data, target, optimizer, criterion):
        """混合精度训练步骤"""
        data = data.to(self.device)
        target = target.to(self.device)
        
        # 使用 autocast 上下文管理器
        with torch.cuda.amp.autocast():
            output = self.model(data)
            loss = criterion(output, target)
        
        # 缩放梯度并反向传播
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数（自动处理精度转换）
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()


def demo_pytorch_basics():
    """
    PyTorch基础演示函数
    
    运行此函数展示所有PyTorch基础概念
    """
    print("=" * 60)
    print("PyTorch 基础教程演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 张量操作
    print("\n1. 张量操作演示")
    TensorOperations.create_tensors()
    TensorOperations.tensor_operations_demo()
    TensorOperations.tensor_indexing_advanced()
    TensorOperations.tensor_dtype_conversion()
    
    # 2. 自动求导
    print("\n2. 自动求导演示")
    AutogradMechanism.basic_autograd()
    AutogradMechanism.autograd_graph()
    AutogradMechanism.autograd_gradients()
    AutogradMechanism.disable_grad()
    
    # 3. 神经网络
    print("\n3. 神经网络演示")
    model = NeuralNetworkModule(input_size=784, hidden_size=256, output_size=10)
    model.initialize_weights()
    
    # 测试前向传播
    dummy_input = torch.randn(32, 784)
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 4. 优化器比较
    print("\n4. 优化器演示")
    OptimizerDemo.compare_optimizers()
    OptimizerDemo.learning_rate_scheduling()
    
    # 5. GPU内存管理
    print("\n5. GPU内存管理")
    GPUMemoryManagement.check_gpu_memory()
    
    print("\n" + "=" * 60)
    print("PyTorch 基础教程演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_pytorch_basics()
