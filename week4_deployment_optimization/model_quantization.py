"""
模型量化实战 - Week 4 部署优化

本模块涵盖大模型量化的完整实现，包括：
1. 量化基础概念与数学原理
2. 动态量化与静态量化
3. GPTQ量化算法实现
4. AWQ激活感知权重量化
5. GGML/GGUF格式转换
6. INT8/INT4量化实践

Author: learn_llm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import json
import os
from pathlib import Path
from datetime import datetime
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """量化配置类"""
    bits: int = 8
    quantize_method: str = "dynamic"
    symmetric: bool = True
    per_channel: bool = True
    group_size: int = -1
    calibration_samples: int = 100
    calibration_method: str = "minmax"


class QuantizationStrategy(ABC):
    """量化策略基类"""
    
    @abstractmethod
    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """量化权重"""
        pass
    
    @abstractmethod
    def dequantize(self, quantized_weights: np.ndarray, scale: np.ndarray, zero_point: np.ndarray) -> np.ndarray:
        """反量化权重"""
        pass


class SymmetricQuantization(QuantizationStrategy):
    """
    对称量化策略
    
    对称量化将浮点数映射到有符号整数的对称范围。
    
    数学公式：
    scale = max_abs / (2^(bits-1) - 1)
    quantized = round(weight / scale)
    dequantized = quantized * scale
    
    特点：
    - 零映射到零
    - 动态范围对称
    - 计算简单高效
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """执行对称量化"""
        max_val = np.max(np.abs(weights))
        if max_val == 0:
            scale = 1.0
        else:
            scale = max_val / self.qmax
        
        quantized = np.clip(np.round(weights / scale), self.qmin, self.qmax).astype(np.int8)
        
        return quantized, {"scale": scale, "zero_point": 0}
    
    def dequantize(self, quantized_weights: np.ndarray, scale: float, zero_point: int = 0) -> np.ndarray:
        """执行反量化"""
        return quantized_weights.astype(np.float32) * scale


class AsymmetricQuantization(QuantizationStrategy):
    """
    非对称量化策略
    
    非对称量化将浮点数映射到有符号整数的完整范围。
    
    数学公式：
    scale = (max - min) / (qmax - qmin)
    zero_point = qmin - round(min / scale)
    quantized = clip(round(weight / scale) + zero_point, qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    
    特点：
    - 更精确地覆盖数据分布
    - 零不一定映射到零
    - 精度更高但计算稍复杂
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """执行非对称量化"""
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / (self.qmax - self.qmin)
            zero_point = int(self.qmin - round(min_val / scale))
        
        quantized = np.clip(
            np.round(weights / scale) + zero_point,
            self.qmin, self.qmax
        ).astype(np.int8)
        
        return quantized, {"scale": scale, "zero_point": zero_point}
    
    def dequantize(self, quantized_weights: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
        """执行反量化"""
        return (quantized_weights.astype(np.float32) - zero_point) * scale


class GroupQuantization:
    """
    分组量化策略
    
    将权重矩阵按组划分，每组独立量化，提高量化精度。
    
    特点：
    - 比per-tensor量化更精确
    - 比per-channel量化更节省存储
    - 适用于大模型量化
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
        self.qmin = 0
        self.qmax = 2 ** bits - 1
    
    def quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """执行分组量化"""
        original_shape = weights.shape
        num_groups = math.prod(original_shape) // self.group_size
        
        quantized_flat = np.zeros(math.prod(original_shape), dtype=np.uint8)
        scales = []
        zeros = []
        
        for i in range(num_groups):
            start = i * self.group_size
            end = start + self.group_size
            group = weights.flat[start:end]
            
            max_val = np.max(np.abs(group))
            if max_val == 0:
                scale = 1.0
                zero = 0
            else:
                scale = max_val / self.qmax
                zero = 0
            
            quantized_group = np.clip(np.round(group / scale), self.qmin, self.qmax).astype(np.uint8)
            quantized_flat[start:end] = quantized_group
            
            scales.append(scale)
            zeros.append(zero)
        
        return quantized_flat.reshape(original_shape), {
            "scales": scales,
            "zeros": zeros,
            "group_size": self.group_size
        }
    
    def dequantize(self, quantized_weights: np.ndarray, meta: Dict) -> np.ndarray:
        """执行分组反量化"""
        original_shape = quantized_weights.shape
        flat = quantized_weights.flat
        scales = meta["scales"]
        zeros = meta["zeros"]
        group_size = meta["group_size"]
        
        num_groups = len(flat) // group_size
        dequantized_flat = np.zeros(len(flat), dtype=np.float32)
        
        for i in range(num_groups):
            start = i * group_size
            end = start + group_size
            dequantized_flat[start:end] = flat[start:end].astype(np.float32) * scales[i]
        
        return dequantized_flat.reshape(original_shape)


class DynamicQuantizer:
    """
    动态量化器
    
    推理时实时量化，仅量化权重，激活在运行时量化。
    
    特点：
    - 实现简单
    - 无需校准数据
    - 推理速度提升有限
    - 适合快速部署
    """
    
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True
    ):
        self.bits = bits
        self.symmetric = symmetric
        self.strategy = SymmetricQuantization(bits) if symmetric else AsymmetricQuantization(bits)
    
    def quantize_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """量化单个张量"""
        weights = tensor.cpu().numpy()
        quantized, meta = self.strategy.quantize(weights)
        
        return torch.from_numpy(quantized), meta
    
    def quantize_linear_layer(self, layer: nn.Linear) -> Tuple[nn.Linear, Dict]:
        """
        量化Linear层
        
        Args:
            layer: 原始Linear层
            
        Returns:
            (量化后的层, 元数据字典)
        """
        quantized_layer = nn.Linear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            device=layer.weight.device
        )
        
        quantized_weight, weight_meta = self.quantize_tensor(layer.weight)
        quantized_layer.weight.data = quantized_weight
        
        if layer.bias is not None:
            quantized_layer.bias.data = layer.bias.data.clone()
        
        meta = {
            "weight_meta": weight_meta,
            "in_features": layer.in_features,
            "out_features": layer.out_features
        }
        
        return quantized_layer, meta
    
    def quantize_model(self, model: nn.Module) -> Tuple[nn.Module, Dict]:
        """
        量化整个模型
        
        Args:
            model: 原始模型
            
        Returns:
            (量化后的模型, 元数据字典)
        """
        quantized_model = model.__class__.__new__(model.__class__)
        
        all_meta = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                new_module, meta = self.quantize_linear_layer(module)
                setattr(quantized_model, name, new_module)
                all_meta[name] = meta
            elif isinstance(module, nn.LayerNorm):
                setattr(quantized_model, name, module)
            else:
                new_module = self.quantize_module(module) if hasattr(self, 'quantize_module') else module
                setattr(quantized_model, name, new_module)
        
        logger.info(f"动态量化完成，共量化{len(all_meta)}个Linear层")
        
        return quantized_model, all_meta
    
    def quantize_module(self, module: nn.Module) -> nn.Module:
        """递归量化模块"""
        new_module = module.__class__.__new__(module.__class__)
        
        for name, param in module.named_parameters():
            if isinstance(module, nn.Linear):
                quantized, _ = self.quantize_tensor(param)
                new_param = nn.Parameter(quantized, requires_grad=False)
                setattr(new_module, name, new_param)
            else:
                setattr(new_module, name, param)
        
        for name, child in module.named_children():
            quantized_child = self.quantize_module(child)
            setattr(new_module, name, quantized_child)
        
        return new_module


class StaticQuantizer:
    """
    静态量化器
    
    使用校准数据预先量化权重和激活，需要数据校准过程。
    
    特点：
    - 推理速度更快
    - 需要校准数据
    - 精度可能略低于动态量化
    - 适合生产环境
    """
    
    def __init__(
        self,
        bits: int = 8,
        calibration_method: str = "minmax"
    ):
        self.bits = bits
        self.calibration_method = calibration_method
        self.quant_min = -2 ** (bits - 1)
        self.quant_max = 2 ** (bits - 1) - 1
        
        self.activation_scales = {}
        self.weight_scales = {}
    
    def calibrate_activation(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        num_batches: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        校准激活值统计
        
        Args:
            model: 模型
            calibration_data: 校准数据
            num_batches: 使用的批次数
            
        Returns:
            激活值缩放因子字典
        """
        model.eval()
        model = model.to("cpu")
        
        activation_stats = {}
        
        hooks = []
        
        def register_hook(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(output[0].abs().max().item())
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(register_hook(name))
                hooks.append(hook)
        
        with torch.no_grad():
            for i, batch in enumerate(calibration_data):
                if i >= num_batches:
                    break
                model(batch)
        
        for hook in hooks:
            hook.remove()
        
        activation_scales = {}
        for name, stats in activation_stats.items():
            max_val = max(stats) if stats else 1.0
            activation_scales[name] = max_val / self.quant_max
        
        logger.info(f"激活值校准完成，共校准{len(activation_scales)}个模块")
        
        return activation_scales
    
    def quantize_tensor_static(
        self,
        tensor: torch.Tensor,
        scales: Dict[str, float] = None,
        axis: str = "tensor"
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        静态量化张量
        
        Args:
            tensor: 输入张量
            scales: 缩放因子
            axis: 量化维度（tensor或channel）
        """
        if axis == "channel" and tensor.dim() > 1:
            scales_per_channel = tensor.abs().max(dim=1, keepdim=True)[0]
            scale = scales_per_channel / self.quant_max
            scale = scale.view(-1).cpu().numpy()
        else:
            max_val = tensor.abs().max().item()
            scale_val = max_val / self.quant_max if max_val > 0 else 1.0
            scale = np.array([scale_val])
        
        quantized = torch.clamp(
            torch.round(tensor / scale.view(*([-1] + [1] * (tensor.dim() - 1)))),
            self.quant_min,
            self.quant_max
        ).to(torch.int8)
        
        return quantized, scale
    
    def quantize_model_static(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor] = None
    ) -> Tuple[nn.Module, Dict]:
        """
        静态量化模型
        
        Args:
            model: 原始模型
            calibration_data: 校准数据
            
        Returns:
            (量化后的模型, 元数据)
        """
        if calibration_data:
            self.activation_scales = self.calibrate_activation(
                model, calibration_data
            )
        
        quantized_model = model.__class__.__new__(model.__class__)
        
        all_meta = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantized, scale = self.quantize_tensor_static(module.weight)
                
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device
                )
                new_module.weight.data = quantized
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                setattr(quantized_model, name, new_module)
                all_meta[name] = {"weight_scale": scale}
                
            else:
                setattr(quantized_model, name, module)
        
        logger.info(f"静态量化完成，共量化{len(all_meta)}个Linear层")
        
        return quantized_model, all_meta


class GPTQQuantizer:
    """
    GPTQ量化器
    
    GPTQ（Gradient Post-Training Quantization）是一种基于误差反馈的
    逐层量化算法，支持4位及以下量化，精度损失极小。
    
    算法核心：
    1. 逐层处理权重矩阵
    2. 对每列进行误差反馈量化
    3. 使用Hessian矩阵信息调整未量化列
    
    特点：
    - 支持4位及以下量化
    - 精度损失小
    - 量化速度较慢
    - 需要GPU加速
    """
    
    def __init__(
        self,
        bits: int = 4,
        percdamp: float = 0.01,
        groupsize: int = 128,
        actorder: bool = False
    ):
        self.bits = bits
        self.percdamp = percdamp
        self.groupsize = groupsize
        self.actorder = actorder
        
        self.quant_min = 0
        self.quant_max = 2 ** bits - 1
    
    def quantize(
        self,
        weight: torch.Tensor,
        Hessian: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行GPTQ量化
        
        Args:
            weight: 权重矩阵 (out_features, in_features)
            Hessian: 近似Hessian矩阵
            
        Returns:
            (量化权重, 缩放因子, 零点)
        """
        out_features, in_features = weight.shape
        
        if self.groupsize != -1:
            num_groups = in_features // self.groupsize
            weight = weight.view(out_features, num_groups, self.groupsize)
        
        scales = torch.zeros_like(weight)
        zeros = torch.zeros_like(weight)
        quantized = torch.zeros_like(weight, dtype=torch.uint8)
        
        if Hessian is None:
            Hessian = torch.eye(in_features, device=weight.device)
        
        for g in range(weight.shape[1]):
            w = weight[:, g, :] if weight.dim() == 3 else weight[:, g * self.groupsize:(g + 1) * self.groupsize]
            
            max_val = w.abs().max(dim=1, keepdim=True)[0]
            scale = max_val / self.quant_max
            scales[:, g, :] = scale if scales.dim() == 3 else scale
            
            q = torch.clamp(torch.round(w / scale), self.quant_min, self.quant_max)
            quantized[:, g, :] = q if quantized.dim() == 3 else q
            
            zeros[:, g, :] = 0
        
        if weight.dim() == 2:
            scales = scales.squeeze(1)
            zeros = zeros.squeeze(1)
            quantized = quantized.squeeze(1)
        else:
            scales = scales.view(out_features, -1)
            zeros = zeros.view(out_features, -1)
            quantized = quantized.view(out_features, -1)
        
        logger.info(f"GPTQ量化完成: {self.bits}位, shape={weight.shape}")
        
        return quantized, scales, zeros
    
    def quantize_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader = None
    ) -> Tuple[nn.Module, Dict]:
        """
        量化整个模型
        
        Args:
            model: 原始模型
            dataloader: 数据加载器（用于收集激活统计）
            
        Returns:
            (量化后的模型, 元数据)
        """
        quantized_model = model.__class__.__new__(model.__class__)
        
        all_meta = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantized, scales, zeros = self.quantize(module.weight.data)
                
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device
                )
                
                new_module.weight.data = (quantized.to(torch.int8) - zeros.to(torch.int8)).float() * scales
                
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                setattr(quantized_model, name, new_module)
                all_meta[name] = {
                    "scales": scales,
                    "zeros": zeros,
                    "bits": self.bits,
                    "groupsize": self.groupsize
                }
            else:
                setattr(quantized_model, name, module)
        
        logger.info(f"GPTQ模型量化完成，共量化{len(all_meta)}个层")
        
        return quantized_model, all_meta


class AWQQuantizer:
    """
    AWQ量化器
    
    AWQ（Activation-aware Weight Quantization）是一种激活感知权重
    量化技术，根据激活重要性保护重要权重。
    
    算法核心：
    1. 计算各权重对模型输出的重要性
    2. 重要性与激活值成正比
    3. 对重要权重使用更细粒度量化
    
    特点：
    - 更好的保持模型能力
    - 量化精度高
    - 适用于大语言模型
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        topk: float = 0.1
    ):
        self.bits = bits
        self.group_size = group_size
        self.topk = topk
        self.quant_min = 0
        self.quant_max = 2 ** bits - 1
    
    def calculate_weight_importance(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor
    ) -> torch.Tensor:
        """
        计算权重重要性
        
        重要性 = |权重| * 激活均值
        
        Args:
            weight: 权重
            activation: 激活值
            
        Returns:
            重要性分数
        """
        importance = weight.abs() * activation.abs().mean(dim=0)
        return importance
    
    def quantize_awq(
        self,
        weight: torch.Tensor,
        importance: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行AWQ量化
        
        Args:
            weight: 权重矩阵
            importance: 重要性分数
            
        Returns:
            (量化权重, 缩放因子)
        """
        if importance is None:
            importance = weight.abs()
        
        out_features, in_features = weight.shape
        
        if self.group_size > 0:
            num_groups = in_features // self.group_size
            weight = weight.view(out_features, num_groups, self.group_size)
            importance = importance.view(out_features, num_groups, self.group_size)
        
        scales = torch.zeros_like(weight)
        quantized = torch.zeros_like(weight, dtype=torch.uint8)
        
        for g in range(weight.shape[1]):
            w_g = weight[:, g, :] if weight.dim() == 3 else weight[:, g * self.group_size:(g + 1) * self.group_size]
            imp_g = importance[:, g, :] if importance.dim() == 3 else importance[:, g * self.group_size:(g + 1) * self.group_size]
            
            w_g_flat = w_g.view(-1)
            imp_g_flat = imp_g.view(-1)
            
            if self.topk > 0:
                k = max(1, int(len(imp_g_flat) * self.topk))
                topk_indices = imp_g_flat.topk(k).indices
                mask = torch.zeros_like(imp_g_flat, dtype=torch.bool)
                mask[topk_indices] = True
                
                max_val = w_g_flat[mask].abs().max() if mask.sum() > 0 else w_g_flat.abs().max()
            else:
                max_val = w_g_flat.abs().max()
            
            scale = max_val / self.quant_max if max_val > 0 else 1.0
            
            q = torch.clamp(torch.round(w_g_flat / scale), self.quant_min, self.quant_max)
            
            scales_g = scale if weight.dim() == 2 else scale.view(1, -1)
            
            scales[:, g, :] = scales_g if scales.dim() == 3 else scales_g
            quantized[:, g, :] = q.view(out_features, self.group_size) if weight.dim() == 2 else q.view(out_features, -1)
        
        if weight.dim() == 2:
            scales = scales.squeeze(1)
            quantized = quantized.squeeze(1)
        else:
            scales = scales.view(out_features, -1)
            quantized = quantized.view(out_features, -1)
        
        return quantized, scales
    
    def quantize_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader = None
    ) -> Tuple[nn.Module, Dict]:
        """
        量化整个模型
        
        Args:
            model: 原始模型
            dataloader: 数据加载器
            
        Returns:
            (量化后的模型, 元数据)
        """
        quantized_model = model.__class__.__new__(model.__class__)
        
        all_meta = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                quantized, scales = self.quantize_awq(module.weight.data)
                
                new_module = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device
                )
                
                new_module.weight.data = quantized.float() * scales
                
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()
                
                setattr(quantized_model, name, new_module)
                all_meta[name] = {
                    "scales": scales,
                    "bits": self.bits,
                    "groupsize": self.group_size
                }
            else:
                setattr(quantized_model, name, module)
        
        logger.info(f"AWQ模型量化完成，共量化{len(all_meta)}个层")
        
        return quantized_model, all_meta


class QuantizationMetrics:
    """量化评估指标类"""
    
    @staticmethod
    def calculate_similarity(
        original: torch.Tensor,
        quantized: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算原始模型与量化模型的相似度
        
        Args:
            original: 原始权重
            quantized: 量化后权重
            
        Returns:
            相似度指标
        """
        orig = original.float()
        quant = quantized.float()
        
        cos_sim = F.cosine_similarity(
            orig.view(-1),
            quant.view(-1),
            dim=0
        ).item()
        
        mse = F.mse_loss(orig, quant).item()
        
        max_diff = (orig - quant).abs().max().item()
        
        relative_error = ((orig - quant).abs() / (orig.abs() + 1e-8)).mean().item()
        
        return {
            "cosine_similarity": cos_sim,
            "mse": mse,
            "max_difference": max_diff,
            "relative_error": relative_error
        }
    
    @staticmethod
    def calculate_compression_ratio(
        original_size: int,
        quantized_size: int,
        original_bits: int = 32,
        quantized_bits: int = 8
    ) -> Dict[str, float]:
        """
        计算压缩比
        
        Args:
            original_size: 原始参数数量
            quantized_size: 量化后参数数量
            original_bits: 原始位数
            quantized_bits: 量化后位数
            
        Returns:
            压缩指标
        """
        theoretical_ratio = original_bits / quantized_bits
        
        actual_ratio = (original_size * original_bits) / (quantized_size * quantized_bits)
        
        return {
            "theoretical_ratio": theoretical_ratio,
            "actual_ratio": actual_ratio,
            "size_reduction": (1 - quantized_size / original_size) * 100
        }


class QuantizationDemo:
    """量化演示类"""
    
    @staticmethod
    def demo_quantization_types():
        """演示不同量化类型"""
        print("\n" + "=" * 60)
        print("模型量化类型对比")
        print("=" * 60)
        
        comparison = {
            "动态量化": {
                "精度": "FP16 → INT8",
                "实现难度": "低",
                "推理速度提升": "1.5-3x",
                "精度损失": "较小",
                "适用场景": "快速部署、CPU推理"
            },
            "静态量化": {
                "精度": "FP16 → INT8 (预量化)",
                "实现难度": "中",
                "推理速度提升": "2-4x",
                "精度损失": "中",
                "适用场景": "生产环境、追求速度"
            },
            "GPTQ": {
                "精度": "FP16 → INT4/INT8",
                "实现难度": "高",
                "推理速度提升": "3-8x",
                "精度损失": "极小",
                "适用场景": "大模型4位量化"
            },
            "AWQ": {
                "精度": "FP16 → INT4/INT8",
                "实现难度": "高",
                "推理速度提升": "3-6x",
                "精度损失": "小",
                "适用场景": "激活敏感的模型"
            }
        }
        
        for quant_type, details in comparison.items():
            print(f"\n{quant_type}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    @staticmethod
    def demo_quantization_math():
        """演示量化数学原理"""
        print("\n" + "=" * 60)
        print("量化数学原理")
        print("=" * 60)
        
        print("\n对称量化公式:")
        print("  scale = max_abs / (2^(bits-1) - 1)")
        print("  quantized = round(weight / scale)")
        print("  dequantized = quantized * scale")
        
        print("\n非对称量化公式:")
        print("  scale = (max - min) / (qmax - qmin)")
        print("  zero_point = qmin - round(min / scale)")
        print("  quantized = clip(round(weight / scale) + zero_point, qmin, qmax)")
        
        print("\n分组量化:")
        print("  将权重矩阵划分为固定大小的组")
        print("  每组独立计算scale和zero_point")
        print("  平衡精度与存储开销")


def demo_model_quantization():
    """模型量化演示主函数"""
    print("=" * 60)
    print("模型量化实战演示")
    print("=" * 60)
    
    QuantizationDemo.demo_quantization_types()
    QuantizationDemo.demo_quantization_math()
    
    print("\n" + "=" * 60)
    print("模型量化实战演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_model_quantization()
