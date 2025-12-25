"""
全量微调实战 - Week 3 微调实践

本模块涵盖大模型全量微调的完整实现，包括：
1. 全量微调基础概念与原理
2. 多任务微调配置与实现
3. 训练超参数优化策略
4. 混合精度训练与显存优化
5. 模型保存、加载与推理
6. 分布式训练入门

Author: learn_llm
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup, AdamW
)
from torch.optim import AdamW as TorchAdamW
from torch.cuda.amp import autocast, GradScaler
from datasets import Dataset as HFDataset
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """全量微调配置类"""
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    fp16: bool = True
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./finetuned_model"


class FullFinetuningBasics:
    """
    全量微调基础类 - 讲解全量微调的核心概念与原理
    
    全量微调（Full Fine-tuning）指在微调过程中更新预训练模型的所有参数。
    与参数高效微调相比，全量微调具有更高的表达能力，但计算成本也显著增加。
    
    适用场景：
    - 目标任务与预训练任务差异较大
    - 追求最高可能的模型性能
    - 计算资源充足（多GPU或TPU）
    - 训练数据量较大
    """
    
    @staticmethod
    def explain_full_vs_partial_finetuning():
        """对比全量微调与部分微调"""
        comparison = {
            "参数更新范围": {
                "全量微调": "更新所有模型参数（包括 Embedding 层）",
                "部分微调": "仅更新特定层或新增模块参数"
            },
            "显存占用": {
                "全量微调": "需要存储所有参数的梯度，显存占用大",
                "部分微调": "只需存储部分参数梯度，显存占用小"
            },
            "表达能力": {
                "全量微调": "完全适应目标任务，表达能力最强",
                "部分微调": "受限于适配器容量，表达能力受限"
            },
            "训练时间": {
                "全量微调": "需要更新所有参数，训练时间长",
                "部分微调": "参数更新量小，训练速度快"
            },
            "适用场景": {
                "全量微调": "数据充足、追求最高精度、资源充足",
                "部分微调": "资源受限、快速迭代、多任务复用"
            }
        }
        
        for category, comparison_dict in comparison.items():
            logger.info(f"\n{category}:")
            for method, description in comparison_dict.items():
                logger.info(f"  {method}: {description}")
        
        return comparison
    
    @staticmethod
    def analyze_compute_requirements(
        model_name: str = "bert-base-chinese"
    ) -> Dict[str, Any]:
        """
        分析全量微调的计算需求
        
        Returns:
            包含参数数量、显存需求等信息的字典
        """
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(model_name)
        
        # 估算参数数量
        vocab_size = config.vocab_size
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size
        
        # 参数估算公式
        embedding_params = vocab_size * hidden_size
        attention_params = num_layers * (
            4 * hidden_size * hidden_size +  # Query, Key, Value, Output
            2 * hidden_size * hidden_size    # Attention dense
        )
        ffn_params = num_layers * (
            2 * hidden_size * intermediate_size * 2  # Two FFN layers
        )
        layer_norm_params = num_layers * 4 * hidden_size
        
        total_params = (
            embedding_params + attention_params + 
            ffn_params + layer_norm_params
        )
        
        # 显存估算（FP32）
        params_memory_gb = total_params * 4 / (1024**3)
        gradients_memory_gb = params_memory_gb
        activations_memory_gb = params_memory_gb * 2  # 估算激活值显存
        
        # 使用混合精度时显存减半
        fp16_params_memory_gb = params_memory_gb / 2
        
        analysis = {
            "model_name": model_name,
            "total_parameters": total_params,
            "parameters_million": total_params / 1e6,
            "fp32_total_memory_gb": params_memory_gb + gradients_memory_gb + activations_memory_gb,
            "fp16_total_memory_gb": fp16_params_memory_gb + gradients_memory_gb + activations_memory_gb,
            "config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "intermediate_size": intermediate_size
            }
        }
        
        logger.info(f"模型: {model_name}")
        logger.info(f"参数量: {total_params/1e6:.2f}M")
        logger.info(f"FP32显存需求: {analysis['fp32_total_memory_gb']:.2f} GB")
        logger.info(f"FP16显存需求: {analysis['fp16_total_memory_gb']:.2f} GB")
        
        return analysis


class TextClassificationFinetuner:
    """
    文本分类全量微调类
    
    演示针对文本分类任务的完整微调流程
    """
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def prepare_data(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ):
        """
        准备分类任务数据
        
        Args:
            train_texts: 训练文本列表
            train_labels: 训练标签列表
            val_texts: 验证文本列表
            val_labels: 验证标签列表
        """
        logger.info("准备分类数据集...")
        
        class ClassificationDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx] if self.labels else 0
                
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        self.train_dataset = ClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.config.max_length
        )
        
        if val_texts:
            self.val_dataset = ClassificationDataset(
                val_texts, val_labels, self.tokenizer, self.config.max_length
            )
        else:
            self.val_dataset = None
    
    def load_model(self, num_labels: int):
        """
        加载预训练模型用于分类
        
        Args:
            num_labels: 分类标签数量
        """
        logger.info(f"加载预训练模型: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels
        )
        
        logger.info(f"模型加载完成，参数量: {self.model.num_parameters()/1e6:.2f}M")
        
    def create_training_args(self) -> TrainingArguments:
        """创建训练参数"""
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            report_to="none",
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            seed=42,
        )
        
        logger.info("训练参数创建完成")
        return args
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            eval_pred: 包含预测值和真实值的元组
            
        Returns:
            包含各指标的字典
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        accuracy = (predictions == labels).mean()
        
        return {"accuracy": float(accuracy)}
    
    def create_trainer(self):
        """创建Trainer实例"""
        training_args = self.create_training_args()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer
        )
        
        logger.info("Trainer创建完成")
        return self.trainer
    
    def train(self, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        执行训练
        
        Args:
            resume_from_checkpoint: 是否从检查点恢复
            
        Returns:
            训练结果字典
        """
        from transformers.trainer_utils import get_last_checkpoint
        
        checkpoint = None
        if resume_from_checkpoint:
            last_checkpoint = get_last_checkpoint(self.config.output_dir)
            if last_checkpoint:
                checkpoint = last_checkpoint
                logger.info(f"从检查点恢复: {checkpoint}")
        
        logger.info("开始训练...")
        start_time = time.time()
        
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        
        elapsed = time.time() - start_time
        logger.info(f"训练完成，耗时: {elapsed/60:.2f}分钟")
        
        return {
            "training_loss": train_result.training_loss,
            "training_time": elapsed,
            "global_step": train_result.global_step
        }
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.val_dataset is None:
            raise ValueError("未提供验证数据集")
        
        metrics = self.trainer.evaluate()
        logger.info(f"评估指标: {metrics}")
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        """保存模型"""
        output_dir = output_dir or self.config.output_dir
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"模型已保存到: {output_dir}")
    
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        预测新样本
        
        Args:
            texts: 待预测文本列表
            
        Returns:
            包含预测结果和概率的字典
        """
        self.model.eval()
        
        inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
        
        return {
            "predictions": predictions.numpy().tolist(),
            "probabilities": probabilities.numpy().tolist()
        }


class CausalLMFinetuner:
    """
    因果语言模型全量微调类
    
    演示针对GPT等因果语言模型的微调流程
    """
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def load_model(self, model_name: str = "gpt2"):
        """加载因果语言模型"""
        logger.info(f"加载因果语言模型: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"模型加载完成，参数量: {self.model.num_parameters()/1e6:.2f}M")
    
    def prepare_causal_lm_data(
        self,
        texts: List[str],
        block_size: int = 128
    ) -> HFDataset:
        """
        准备因果语言模型训练数据
        
        Args:
            texts: 文本列表
            block_size: 文本块大小
            
        Returns:
            处理后的数据集
        """
        logger.info("准备因果语言模型数据...")
        
        def tokenize_function(examples):
            output = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=block_size,
                padding='max_length',
                return_special_tokens_mask=True
            )
            output["labels"] = output["input_ids"].copy()
            return output
        
        dataset = HFDataset.from_dict({"text": texts})
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing dataset"
        )
        
        logger.info(f"数据准备完成，样本数: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def create_lm_training_args(self) -> TrainingArguments:
        """创建语言模型训练参数"""
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="epoch",
            fp16=self.config.fp16 and torch.cuda.is_available(),
            report_to="none",
            seed=42,
        )
        return args
    
    def train(self, train_dataset: HFDataset, eval_dataset: Optional[HFDataset] = None):
        """执行语言模型训练"""
        from transformers import DataCollatorForLanguageModeling
        
        training_args = self.create_lm_training_args()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("开始语言模型训练...")
        self.trainer.train()
        
        return self.trainer
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样阈值
            do_sample: 是否采样
            
        Returns:
            生成的文本
        """
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


class CustomTrainingLoop:
    """
    自定义训练循环类 - 展示手动实现训练过程
    
    适用于需要精细控制训练过程的场景
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = TorchAdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        total_steps = len(train_loader) * 10
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.scaler = GradScaler()
        
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', input_ids)
            labels = labels.to(self.device)
            
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            loss = loss / self.gradient_accumulation_steps()
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps() == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += loss.item() * self.gradient_accumulation_steps()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def gradient_accumulation_steps(self) -> int:
        """获取梯度累积步数"""
        return getattr(self.train_loader, 'gradient_accumulation_steps', 1)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels', input_ids)
            labels = labels.to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            if hasattr(outputs, 'logits'):
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(
        self,
        num_epochs: int,
        eval_interval: int = 1
    ) -> Dict[str, List]:
        """
        执行完整训练
        
        Args:
            num_epochs: 训练轮数
            eval_interval: 评估间隔
            
        Returns:
            训练历史记录
        """
        logger.info(f"开始训练，共 {num_epochs} 个epochs")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - start_time
            
            self.train_history.append(train_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Time: {epoch_time:.1f}s")
            
            if (epoch + 1) % eval_interval == 0 and self.val_loader:
                val_metrics = self.evaluate()
                self.val_history.append(val_metrics.get('loss', 0))
                logger.info(f"Validation - Loss: {val_metrics.get('loss'):.4f}, "
                           f"Accuracy: {val_metrics.get('accuracy'):.4f}")
        
        return {
            "train_loss": self.train_history,
            "val_loss": self.val_history
        }
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        logger.info(f"检查点已从 {filepath} 加载")


class DistributedTrainingHelper:
    """
    分布式训练辅助类
    
    支持多GPU训练，加速大模型微调
    """
    
    @staticmethod
    def setup_distributed():
        """设置分布式训练环境"""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler
        
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            
            logger.info(f"分布式训练初始化完成，local_rank: {local_rank}")
            
            return local_rank, DDP, DistributedSampler
        
        return None, None, None
    
    @staticmethod
    def cleanup_distributed():
        """清理分布式训练环境"""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


class FinetuningBestPractices:
    """
    微调最佳实践类
    
    总结全量微调的经验与建议
    """
    
    @staticmethod
    def get_hyperparameter_recommendations() -> Dict[str, Any]:
        """
        获取超参数推荐值
        
        基于 Hugging Face 官方最佳实践和大量实验总结
        """
        recommendations = {
            "学习率": {
                "bert": {"start": 2e-5, "end": 5e-5, "note": "BERT类模型推荐较小学习率"},
                "gpt": {"start": 2e-5, "end": 6e-5, "note": "GPT类模型推荐学习率范围"},
                "t5": {"start": 3e-5, "end": 5e-5, "note": "T5/编码器-解码器模型"}
            },
            "batch_size": {
                "per_device": 8,
                "effective": 32,
                "note": "通过梯度累积实现更大有效batch"
            },
            "训练轮数": {
                "classification": "2-4 epochs",
                "generation": "3-5 epochs",
                "note": "根据验证集性能早停"
            },
            "warmup": {
                "ratio": 0.1,
                "note": "训练总步数的10%用于学习率预热"
            },
            "权重衰减": {
                "value": 0.01,
                "note": "应用于所有偏置和LayerNorm参数除外"
            },
            "混合精度": {
                "fp16": True,
                "note": "显著减少显存占用，加速训练"
            }
        }
        
        for category, details in recommendations.items():
            logger.info(f"\n{category}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    logger.info(f"  {key}: {value}")
        
        return recommendations
    
    @staticmethod
    def get_common_issues_and_solutions() -> Dict[str, str]:
        """
        常见问题与解决方案
        
        Returns:
            问题描述与解决方案的字典
        """
        issues = {
            "Loss不下降": "检查学习率是否过小或过大，尝试使用学习率搜索",
            "模型不收敛": "检查数据标签是否正确，增加训练数据量",
            "显存不足": "减小batch_size，使用梯度累积，启用混合精度",
            "过拟合": "增加正则化（dropout、weight_decay），增加数据增强",
            "训练不稳定": "减小学习率，使用梯度裁剪，检查数据质量",
            "预测结果全为同一类": "检查数据分布是否不平衡，使用类别平衡采样"
        }
        
        for issue, solution in issues.items():
            logger.info(f"\n问题: {issue}")
            logger.info(f"解决方案: {solution}")
        
        return issues


def demo_full_finetuning():
    """全量微调演示主函数"""
    print("=" * 60)
    print("全量微调实战演示")
    print("=" * 60)
    
    print("\n1. 全量微调与部分微调对比")
    FullFinetuningBasics.explain_full_vs_partial_finetuning()
    
    print("\n2. 计算需求分析")
    FullFinetuningBasics.analyze_compute_requirements("bert-base-chinese")
    
    print("\n3. 最佳实践总结")
    FinetuningBestPractices.get_hyperparameter_recommendations()
    FinetuningBestPractices.get_common_issues_and_solutions()
    
    print("\n" + "=" * 60)
    print("全量微调实战演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_full_finetuning()
