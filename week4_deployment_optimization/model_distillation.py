"""
知识蒸馏实战 - Week 4 部署优化

本模块涵盖大模型知识蒸馏的完整实现，包括：
1. 知识蒸馏基础原理与数学公式
2. 传统蒸馏与对比学习蒸馏
3. 中间层特征蒸馏
4. 自蒸馏与渐进式蒸馏
5. 任务特定蒸馏实战
6. 蒸馏模型评估与优化

Author: learn_llm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """蒸馏配置类"""
    temperature: float = 2.0
    alpha: float = 0.5
    beta: float = 0.5
    hard_label_weight: float = 0.5
    soft_label_weight: float = 0.5
    feature_layer: int = -1
    feat_dim: int = 768
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 16


class KnowledgeDistillationBase:
    """
    基础知识蒸馏类
    
    知识蒸馏的核心思想是将大模型（教师模型）的知识迁移到小模型（学生模型）。
    
    数学公式：
    L = α * L_soft + (1-α) * L_hard
    
    其中：
    L_soft = KL(p_T || p_S) * T²
    L_hard = CE(y_true, p_S)
    p_T = softmax(z_T / T), p_S = softmax(z_S / T)
    
    温度参数T的作用：
    - T=1: 标准softmax
    - T>1: 分布更平滑，类别间关系更明显
    - T<1: 分布更尖锐，置信度更高
    """
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.teacher_model = None
        self.student_model = None
    
    @staticmethod
    def soft_cross_entropy(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        计算软交叉熵损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            temperature: 温度参数
            
        Returns:
            软交叉熵损失
        """
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (temperature ** 2)
        
        return loss
    
    @staticmethod
    def hard_cross_entropy(
        student_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算硬标签交叉熵损失
        
        Args:
            student_logits: 学生模型logits
            labels: 真实标签
            
        Returns:
            硬标签交叉熵损失
        """
        return F.cross_entropy(student_logits, labels)
    
    @staticmethod
    def js_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        计算JS散度损失
        
        JS散度是对称且有界的，比KL散度更稳定
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            temperature: 温度参数
            
        Returns:
            JS散度损失
        """
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        m = 0.5 * (student_probs + teacher_probs)
        
        kl_student_m = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            m,
            reduction='batchmean',
            log_target=False
        )
        
        kl_teacher_m = F.kl_div(
            F.log_softmax(teacher_logits / temperature, dim=-1),
            m,
            reduction='batchmean',
            log_target=False
        )
        
        return 0.5 * (kl_student_m + kl_teacher_m) * (temperature ** 2)
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = None,
        temperature: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏总损失
        
        Args:
            student_logits: 学生模型logits
            teacher_logits: 教师模型logits
            labels: 真实标签
            alpha: 软标签权重
            temperature: 温度参数
            
        Returns:
            包含各部分损失的字典
        """
        alpha = alpha if alpha is not None else self.config.alpha
        temperature = temperature if temperature is not None else self.config.temperature
        
        soft_loss = self.soft_cross_entropy(
            student_logits, teacher_logits, temperature
        )
        
        hard_loss = self.hard_cross_entropy(student_logits, labels)
        
        total_loss = (
            self.config.soft_label_weight * soft_loss +
            self.config.hard_label_weight * hard_loss
        )
        
        return {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_loss
        }


class FeatureDistillation:
    """
    特征蒸馏类
    
    利用教师模型的中间层表示作为监督信号，
    进一步提升学生模型的能力。
    
    蒸馏方式：
    - MSE Loss: 直接匹配特征向量
    - Cosine Loss: 匹配特征方向
    - Projector: 使用投影层适配特征维度
    """
    
    def __init__(self, feat_dim: int = 768, hidden_dim: int = 256):
        """
        初始化特征蒸馏器
        
        Args:
            feat_dim: 特征维度
            hidden_dim: 投影层隐藏维度
        """
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim)
        )
        
        self.criterion = nn.MSELoss()
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            student_features: 学生模型特征
            teacher_features: 教师模型特征
            
        Returns:
            特征蒸馏损失
        """
        student_proj = self.projector(student_features)
        teacher_features = teacher_features.detach()
        
        loss = self.criterion(student_proj, teacher_features)
        
        return loss
    
    def feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算带注意力掩码的特征损失
        
        Args:
            student_features: 学生特征
            teacher_features: 教师特征
            attention_mask: 注意力掩码
            
        Returns:
            带掩码的特征损失
        """
        if attention_mask is not None:
            student_features = student_features * attention_mask.unsqueeze(-1)
            teacher_features = teacher_features * attention_mask.unsqueeze(-1)
            
            mask_sum = attention_mask.sum(dim=-1, keepdim=True) + 1e-8
            student_features = student_features / mask_sum.unsqueeze(-1)
            teacher_features = teacher_features / mask_sum.unsqueeze(-1)
        
        return self.forward(student_features, teacher_features)


class ContrastiveDistillation:
    """
    对比学习蒸馏类
    
    利用对比学习思想进行知识蒸馏，
    增强学生模型对样本间关系的理解。
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        初始化对比蒸馏器
        
        Args:
            temperature: 对比学习温度参数
        """
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_info_nce(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        batch_size: int = None
    ) -> torch.Tensor:
        """
        计算InfoNCE损失
        
        Args:
            student_embeddings: 学生模型嵌入
            teacher_embeddings: 教师模型嵌入
            batch_size: 批次大小
            
        Returns:
            InfoNCE损失
        """
        batch_size = batch_size or student_embeddings.size(0)
        
        student_embeddings = F.normalize(student_embeddings, dim=1)
        teacher_embeddings = F.normalize(teacher_embeddings, dim=1)
        
        similarity_matrix = torch.matmul(student_embeddings, teacher_embeddings.T) / self.temperature
        
        labels = torch.arange(batch_size, device=student_embeddings.device)
        
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
    
    def contrastive_distillation_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        hard_labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算对比蒸馏损失
        
        Args:
            student_embeddings: 学生嵌入
            teacher_embeddings: 教师嵌入
            hard_labels: 硬标签
            
        Returns:
            损失字典
        """
        info_nce_loss = self.compute_info_nce(
            student_embeddings, teacher_embeddings
        )
        
        if hard_labels is not None:
            return {
                "contrastive_loss": info_nce_loss,
                "total_loss": info_nce_loss
            }
        
        return {
            "contrastive_loss": info_nce_loss,
            "total_loss": info_nce_loss
        }


class SelfDistillation:
    """
    自蒸馏类
    
    使用模型自身的历史预测或EMA权重进行蒸馏，
    无需额外的教师模型。
    """
    
    def __init__(self, model: nn.Module, alpha: float = 0.5, tau: float = 1.0):
        """
        初始化自蒸馏器
        
        Args:
            model: 待蒸馏模型
            alpha: 蒸馏权重
            tau: 温度参数
        """
        self.model = model
        self.alpha = alpha
        self.tau = tau
        
        self.ema_model = ExponentialMovingAverage(model, decay=0.999)
        
        self.previous_predictions = []
        self.max_predictions = 5
    
    def update_ema(self):
        """更新EMA模型"""
        self.ema_model.update()
    
    def self_distillation_loss(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算自蒸馏损失
        
        Args:
            logits: 当前模型logits
            hard_labels: 硬标签
            
        Returns:
            损失字典
        """
        with torch.no_grad():
            ema_logits = self.ema_model.module.logits if hasattr(self.ema_model.module, 'logits') else None
            if ema_logits is None:
                ema_logits = self.ema_model(logits)
        
        soft_loss = self.soft_cross_entropy(logits, ema_logits, self.tau)
        hard_loss = F.cross_entropy(logits, hard_labels)
        
        total_loss = (
            self.alpha * soft_loss +
            (1 - self.alpha) * hard_loss
        )
        
        return {
            "total_loss": total_loss,
            "soft_loss": soft_loss,
            "hard_loss": hard_loss
        }
    
    def soft_cross_entropy(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """软交叉熵"""
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        return F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (temperature ** 2)


class ExponentialMovingAverage:
    """指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self._register()
    
    def _register(self):
        """注册模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    (1.0 - self.decay) * param.data +
                    self.decay * self.shadow[name]
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class TaskSpecificDistillation:
    """
    任务特定蒸馏类
    
    针对不同任务定制蒸馏策略
    """
    
    @staticmethod
    def distill_classification(
        student_model: nn.Module,
        teacher_model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        分类任务蒸馏
        
        Args:
            student_model: 学生模型
            teacher_model: 教师模型
            dataloader: 数据加载器
            optimizer: 优化器
            device: 设备
            temperature: 温度参数
            alpha: 蒸馏权重
            
        Returns:
            训练指标
        """
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ).logits
            
            student_logits = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
            
            soft_loss = KnowledgeDistillationBase.soft_cross_entropy(
                student_logits, teacher_logits, temperature
            )
            hard_loss = F.cross_entropy(student_logits, labels)
            
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = student_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": total_correct / total_samples
        }
    
    @staticmethod
    def distill_question_answering(
        student_model: nn.Module,
        teacher_model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        temperature: float = 2.0,
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        问答任务蒸馏
        
        使用token-level蒸馏和span-level蒸馏
        """
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0
        total_samples = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            with torch.no_grad():
                teacher_output = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_start_logits = teacher_output.start_logits
                teacher_end_logits = teacher_output.end_logits
            
            student_output = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_start_logits = student_output.start_logits
            student_end_logits = student_output.end_logits
            
            soft_start_loss = KnowledgeDistillationBase.soft_cross_entropy(
                student_start_logits, teacher_start_logits, temperature
            )
            soft_end_loss = KnowledgeDistillationBase.soft_cross_entropy(
                student_end_logits, teacher_end_logits, temperature
            )
            
            hard_start_loss = F.cross_entropy(student_start_logits, start_positions)
            hard_end_loss = F.cross_entropy(student_end_logits, end_positions)
            
            loss = alpha * (soft_start_loss + soft_end_loss) / 2 + \
                   (1 - alpha) * (hard_start_loss + hard_end_loss) / 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_samples += input_ids.size(0)
        
        return {
            "loss": total_loss / len(dataloader)
        }


class DistillationTrainer:
    """
    蒸馏训练器类 - 完整蒸馏流程封装
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        device: str = "cuda"
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=config.learning_rate
        )
        
        self.feature_distiller = FeatureDistillation(
            feat_dim=config.feat_dim,
            hidden_dim=config.feat_dim // 4
        )
        
        self.contrastive_distiller = ContrastiveDistillation()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        use_feature_distillation: bool = True,
        use_contrastive: bool = False
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.student_model.train()
        self.teacher_model.eval()
        
        total_loss = 0
        total_soft_loss = 0
        total_hard_loss = 0
        total_feat_loss = 0
        total_samples = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            with torch.no_grad():
                teacher_output = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_output.logits
                teacher_hidden = teacher_output.hidden_states[self.config.feature_layer]
            
            student_output = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            student_logits = student_output.logits
            student_hidden = student_output.hidden_states[self.config.feature_layer]
            
            distillation_result = KnowledgeDistillationBase.soft_cross_entropy(
                student_logits, teacher_logits, self.config.temperature
            )
            hard_loss = F.cross_entropy(student_logits, labels) if labels is not None else 0
            
            soft_loss = distillation_result
            
            loss_dict = {
                "soft_loss": soft_loss,
                "hard_loss": hard_loss
            }
            
            if use_feature_distillation:
                feat_loss = self.feature_distiller(
                    student_hidden, teacher_hidden, attention_mask
                )
                loss_dict["feature_loss"] = feat_loss
            else:
                feat_loss = 0
            
            if use_contrastive:
                contrast_loss = self.contrastive_distiller.compute_info_nce(
                    student_hidden[:, 0, :],
                    teacher_hidden[:, 0, :]
                )
                loss_dict["contrastive_loss"] = contrast_loss
            else:
                contrast_loss = 0
            
            total_loss_value = (
                self.config.soft_label_weight * soft_loss +
                self.config.hard_label_weight * hard_loss +
                feat_loss * self.config.beta +
                contrast_loss * self.config.beta
            )
            
            self.optimizer.zero_grad()
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_loss_value.item()
            total_soft_loss += soft_loss.item()
            total_hard_loss += hard_loss.item() if hard_loss else 0
            total_feat_loss += feat_loss.item() if isinstance(feat_loss, torch.Tensor) else feat_loss
            total_samples += input_ids.size(0)
        
        return {
            "loss": total_loss / len(train_loader),
            "soft_loss": total_soft_loss / len(train_loader),
            "hard_loss": total_hard_loss / len(train_loader),
            "feature_loss": total_feat_loss / len(train_loader)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader = None,
        num_epochs: int = None,
        use_feature_distillation: bool = True,
        use_contrastive: bool = False
    ) -> Dict[str, List[float]]:
        """完整训练流程"""
        num_epochs = num_epochs or self.config.epochs
        
        history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": []
        }
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(
                train_loader,
                use_feature_distillation,
                use_contrastive
            )
            
            history["train_loss"].append(train_metrics["loss"])
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Loss: {train_metrics['loss']:.4f}, "
                       f"Soft Loss: {train_metrics['soft_loss']:.4f}")
            
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                history["eval_loss"].append(eval_metrics.get("loss", 0))
                history["eval_accuracy"].append(eval_metrics.get("accuracy", 0))
                
                logger.info(f"Eval Loss: {eval_metrics.get('loss', 0):.4f}, "
                           f"Eval Acc: {eval_metrics.get('accuracy', 0):.4f}")
        
        return history
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.student_model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
            
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()
                
                preds = logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
            
            total_samples += input_ids.size(0)
        
        return {
            "loss": total_loss / len(eval_loader) if total_loss > 0 else 0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0
        }
    
    def save_student(self, output_dir: str):
        """保存学生模型"""
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.student_model.state_dict(), f"{output_dir}/student_model.pt")
        logger.info(f"学生模型已保存到: {output_dir}")


class DistillationMetrics:
    """蒸馏评估指标类"""
    
    @staticmethod
    def calculate_distillation_quality(
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算蒸馏质量指标
        
        Args:
            teacher_logits: 教师模型logits
            student_logits: 学生模型logits
            labels: 真实标签
            
        Returns:
            蒸馏质量指标
        """
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        
        prediction_agreement = (teacher_logits.argmax(dim=-1) == student_logits.argmax(dim=-1)).float().mean().item()
        
        kl_div = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ).item()
        
        teacher_acc = (teacher_logits.argmax(dim=-1) == labels).float().mean().item()
        student_acc = (student_logits.argmax(dim=-1) == labels).float().mean().item()
        
        return {
            "prediction_agreement": prediction_agreement,
            "kl_divergence": kl_div,
            "teacher_accuracy": teacher_acc,
            "student_accuracy": student_acc,
            "accuracy_gap": teacher_acc - student_acc
        }
    
    @staticmethod
    def compare_with_baseline(
        baseline_logits: torch.Tensor,
        distilled_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        与基线模型对比
        
        Args:
            baseline_logits: 基线模型logits
            distilled_logits: 蒸馏模型logits
            labels: 真实标签
            
        Returns:
            对比指标
        """
        baseline_acc = (baseline_logits.argmax(dim=-1) == labels).float().mean().item()
        distilled_acc = (distilled_logits.argmax(dim=-1) == labels).float().mean().item()
        
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        distilled_probs = F.softmax(distilled_logits, dim=-1)
        
        prediction_change = (baseline_logits.argmax(dim=-1) != distilled_logits.argmax(dim=-1)).float().mean().item()
        
        return {
            "baseline_accuracy": baseline_acc,
            "distilled_accuracy": distilled_acc,
            "improvement": distilled_acc - baseline_acc,
            "prediction_change_rate": prediction_change
        }


class DistillationDemo:
    """蒸馏演示类"""
    
    @staticmethod
    def demo_distillation_concept():
        """演示蒸馏概念"""
        print("\n" + "=" * 60)
        print("知识蒸馏核心概念")
        print("=" * 60)
        
        print("\n蒸馏损失函数:")
        print("  L = α * L_soft + (1-α) * L_hard")
        print("  L_soft = KL(p_T || p_S) * T²")
        print("  L_hard = CE(y_true, p_S)")
        
        print("\n温度参数T的作用:")
        print("  T=1: 标准softmax")
        print("  T>1: 分布更平滑，揭示类别间关系")
        print("  T<1: 分布更尖锐")
        
        print("\n蒸馏类型:")
        print("  1. 输出层蒸馏：匹配logits分布")
        print("  2. 特征层蒸馏：匹配中间表示")
        print("  3. 关系蒸馏：匹配样本间关系")
        print("  4. 自蒸馏：使用模型自身历史")
    
    @staticmethod
    def demo_distillation_types():
        """演示蒸馏类型对比"""
        print("\n" + "=" * 60)
        print("蒸馏类型对比")
        print("=" * 60)
        
        comparison = {
            "输出层蒸馏": {
                "原理": "匹配教师和学生输出的概率分布",
                "优点": "简单直接，效果稳定",
                "缺点": "信息有限，仅传递输出层知识",
                "适用": "分类任务快速蒸馏"
            },
            "特征层蒸馏": {
                "原理": "使用投影层匹配中间层表示",
                "优点": "传递更丰富的知识",
                "缺点": "需要处理维度不匹配",
                "适用": "需要深层理解的任务"
            },
            "对比蒸馏": {
                "原理": "利用对比学习对齐嵌入空间",
                "优点": "增强样本间关系理解",
                "缺点": "计算开销较大",
                "适用": "需要语义理解的场景"
            },
            "自蒸馏": {
                "原理": "使用模型自身历史预测",
                "优点": "无需额外教师模型",
                "缺点": "提升有限",
                "适用": "模型增强和正则化"
            }
        }
        
        for method, details in comparison.items():
            print(f"\n{method}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
    
    @staticmethod
    def demo_hyperparameters():
        """演示关键超参数"""
        print("\n" + "=" * 60)
        print("关键超参数影响")
        print("=" * 60)
        
        print("\n温度参数T:")
        print("  T=2-5: 常用范围，平衡软硬标签")
        print("  较高T: 更关注类别间相似性")
        print("  较低T: 更关注自信预测")
        
        print("\nα参数（软硬标签权重）:")
        print("  α=0.5: 软硬标签同等重要")
        print("  α>0.5: 更依赖教师知识")
        print("  α<0.5: 更依赖硬标签")
        
        print("\nβ参数（特征蒸馏权重）:")
        print("  β=0.1-1.0: 控制特征蒸馏强度")
        print("  过高可能导致主任务损失下降")


def demo_knowledge_distillation():
    """知识蒸馏演示主函数"""
    print("=" * 60)
    print("知识蒸馏实战演示")
    print("=" * 60)
    
    DistillationDemo.demo_distillation_concept()
    DistillationDemo.demo_distillation_types()
    DistillationDemo.demo_hyperparameters()
    
    print("\n" + "=" * 60)
    print("知识蒸馏实战演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_knowledge_distillation()
