"""
LoRA/PEFT高效微调实战 - Week 3 微调实践

本模块涵盖参数高效微调（PEFT）技术的完整实现，包括：
1. LoRA（Low-Rank Adaptation）低秩适配原理与实现
2. QLoRA量化低秩适配，消费级GPU微调大模型
3. PEFT库封装与多种高效微调方法对比
4. Prefix Tuning与Prompt Tuning
5. Adapter Layer实现
6. 多任务PEFT实战

Author: learn_llm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, PeftConfig,
    LoftQConfig, prepare_model_for_int8_training,
    PrefixTuningConfig, PromptEncoderConfig,
    AdaLoraConfig, IA3Config
)
from datasets import Dataset as HFDataset
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import numpy as np
import json
import os
import time
from pathlib import Path
from datetime import datetime
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA配置类"""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    bias: str = "none"
    inference_mode: bool = False


class LoraImplementation:
    """
    LoRA实现类 - 深入理解低秩适配原理
    
    LoRA（Low-Rank Adaptation）核心思想：
    预训练模型的权重矩阵W₀通常是高秩的，但在特定任务上，
    权重的变化ΔW可以近似为低秩矩阵。
    
    数学表达：
    W(x) = W₀x + ΔWx = W₀x + BAx
    
    其中B∈ℝ^(d×r)，A∈ℝ^(r×k)，r远小于d和k。
    
    优点：
    - 训练参数量大幅减少（从d×k降到(d+k)×r）
    - 可与其他PEFT方法组合
    - 无推理延迟
    - 可切换不同任务的适配器
    """
    
    @staticmethod
    def explain_lora_mathematics():
        """详细讲解LoRA的数学原理"""
        explanation = {
            "核心思想": [
                "预训练模型的权重矩阵W₀是高维的（d×k）",
                "在下游任务上，权重变化ΔW也是高维的",
                "但ΔW的有效秩很低，可以分解为低秩矩阵",
                "LoRA通过低秩分解BA近似ΔW"
            ],
            "数学公式": {
                "前向传播": "h = W₀x + ΔWx = W₀x + BAx",
                "参数化": "A ~ N(0, σ²), B = 0 (初始化)",
                "秩": "r << min(d, k)，通常r=8, 16, 32"
            },
            "参数量分析": {
                "全量微调": "d × k 参数",
                "LoRA": "(d + k) × r 参数",
                "压缩比": f"例如: d=4096, k=4096, r=8",
                "全量参数": "16,777,216",
                "LoRA参数": "65,536",
                "压缩率": "约256倍"
            },
            "应用方式": [
                "将LoRA适配器注入到Attention层",
                "通常选择Query和Value矩阵",
                "也可应用于MLP层",
                "训练时冻结原始权重"
            ]
        }
        
        for section, content in explanation.items():
            logger.info(f"\n{section}:")
            if isinstance(content, list):
                for item in content:
                    logger.info(f"  - {item}")
            elif isinstance(content, dict):
                for key, value in content.items():
                    logger.info(f"  {key}: {value}")
        
        return explanation
    
    @staticmethod
    def manual_lora_layer(
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05
    ):
        """
        手动实现LoRA层
        
        Args:
            in_features: 输入维度
            out_features: 输出维度
            r: 秩（低秩维度）
            alpha: 缩放因子
            dropout: Dropout概率
            
        Returns:
            自定义LoRA层
        """
        class LoRALayer(nn.Module):
            def __init__(self, in_features, out_features, r, alpha, dropout):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.r = r
                self.alpha = alpha
                
                if r > 0:
                    self.lora_A = nn.Linear(in_features, r, bias=False)
                    self.lora_B = nn.Linear(r, out_features, bias=False)
                    self.dropout = nn.Dropout(p=dropout)
                else:
                    self.lora_A = None
                    self.lora_B = None
                    self.dropout = None
                
                self.reset_parameters()
            
            def reset_parameters(self):
                if self.lora_A is not None:
                    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                if self.lora_B is not None:
                    nn.init.zeros_(self.lora_B.weight)
            
            def forward(self, x, original_weight):
                if self.r == 0:
                    return F.linear(x, original_weight, bias=None)
                
                original_output = F.linear(x, original_weight, bias=None)
                
                lora_output = self.lora_B(self.lora_A(self.dropout(x)))
                scaled_output = original_output + (lora_output * (self.alpha / self.r))
                
                return scaled_output
            
            def merge_weights(self):
                """合并LoRA权重到原始权重"""
                if self.lora_A is not None:
                    return original_weight + (self.lora_B.weight @ self.lora_A.weight) * (self.alpha / self.r)
                return original_weight
        
        return LoRALayer(in_features, out_features, r, alpha, dropout)
    
    @staticmethod
    def apply_lora_to_linear(
        layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        lora_dropout: float = 0.05,
        merge_weights: bool = True
    ) -> nn.Module:
        """
        为现有Linear层添加LoRA适配器
        
        Args:
            layer: 原始Linear层
            r: 秩
            alpha: 缩放因子
            lora_dropout: Dropout概率
            merge_weights: 是否合并权重
            
        Returns:
            添加LoRA后的层
        """
        class LoRALinear(nn.Module):
            def __init__(self, original_layer, r, alpha, dropout):
                super().__init__()
                self.original_layer = original_layer
                self.in_features = original_layer.in_features
                self.out_features = original_layer.out_features
                self.r = r
                self.alpha = alpha
                
                self.lora_A = nn.Linear(self.in_features, r, bias=False)
                self.lora_B = nn.Linear(r, self.out_features, bias=False)
                self.dropout = nn.Dropout(p=dropout)
                
                self.scaling = self.alpha / self.r
                self.merge_weights = merge_weights
                
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)
                
                self._freeze_original()
            
            def _freeze_original(self):
                for param in self.original_layer.parameters():
                    param.requires_grad = False
            
            def forward(self, x):
                original_output = self.original_layer(x)
                
                if self.r > 0:
                    lora_output = self.lora_B(self.lora_A(self.dropout(x)))
                    return original_output + lora_output * self.scaling
                return original_output
            
            def get_trainable_params(self):
                """获取可训练参数"""
                params = []
                for name, param in self.named_parameters():
                    if 'lora' in name:
                        params.append(param)
                return params
        
        return LoRALinear(layer, r, alpha, lora_dropout)


class PEFTIntegration:
    """
    PEFT库集成类 - 使用Hugging Face PEFT工具
    
    PEFT（Parameter-Efficient Fine-Tuning）库提供了统一的
    高效微调接口，支持多种方法：
    - LoRA
    - Prefix Tuning
    - Prompt Tuning
    - AdaLoRA
    - IA3
    """
    
    @staticmethod
    def get_lora_config(
        task_type: TaskType = TaskType.SEQ_CLS,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        bias: str = "none",
        inference_mode: bool = False
    ) -> LoraConfig:
        """
        创建LoRA配置
        
        Args:
            task_type: 任务类型（SEQ_CLS, CAUSAL_LM, etc.）
            r: 秩
            lora_alpha: 缩放因子
            lora_dropout: Dropout概率
            target_modules: 目标模块列表
            bias: 偏置处理方式
            inference_mode: 推理模式
            
        Returns:
            LoraConfig实例
        """
        if target_modules is None:
            target_modules = ["query", "value"]
        
        config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            inference_mode=inference_mode
        )
        
        logger.info(f"LoRA配置创建完成: r={r}, alpha={lora_alpha}, target_modules={target_modules}")
        
        return config
    
    @staticmethod
    def get_prefix_tuning_config(
        num_virtual_tokens: int = 20,
        encoder_hidden_size: int = 128
    ) -> PrefixTuningConfig:
        """
        创建Prefix Tuning配置
        
        Prefix Tuning在每层Transformer前添加可训练的virtual tokens，
        这些tokens作为上下文引导模型生成。
        """
        config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=encoder_hidden_size
        )
        
        logger.info(f"Prefix Tuning配置创建完成: virtual_tokens={num_virtual_tokens}")
        return config
    
    @staticmethod
    def get_prompt_tuning_config(
        num_virtual_tokens: int = 20,
        prompt_encoder_hidden_size: int = 128
    ) -> PromptEncoderConfig:
        """
        创建Prompt Tuning配置
        
        Prompt Tuning仅在embedding层添加可训练的virtual tokens，
        参数量最小，适合超大规模模型。
        """
        config = PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=prompt_encoder_hidden_size
        )
        
        logger.info(f"Prompt Tuning配置创建完成: virtual_tokens={num_virtual_tokens}")
        return config
    
    @staticmethod
    def get_adalora_config(
        target_r: int = 8,
        init_r: int = 12,
        tfinal: int = 0,
        deltaT: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85
    ) -> AdaLoraConfig:
        """
        创建AdaLoRA配置
        
        AdaLoRA（Adaptive LoRA）自动调整不同层的秩分配，
        将更多参数分配到更重要 layers。
        """
        config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_r=target_r,
            init_r=init_r,
            tfinal=tfinal,
            deltaT=deltaT,
            beta1=beta1,
            beta2=beta2
        )
        
        logger.info(f"AdaLoRA配置创建完成: target_r={target_r}, init_r={init_r}")
        return config
    
    @staticmethod
    def get_ia3_config(
        target_modules: List[str] = None,
        feedforward_modules: List[str] = None
    ) -> IA3Config:
        """
        创建IA3配置
        
        IA3（Infusion of Invariant Amplified Vectors）通过缩放
        激活向量来适应任务，比LoRA更轻量。
        """
        config = IA3Config(
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules or ["key", "value", "output"],
            feedforward_modules=feedforward_modules or []
        )
        
        logger.info("IA3配置创建完成")
        return config
    
    @staticmethod
    def apply_peft_model(
        model: nn.Module,
        peft_config: Any,
        adapter_name: str = "default"
    ) -> PeftModel:
        """
        应用PEFT配置到模型
        
        Args:
            model: 原始模型
            peft_config: PEFT配置
            adapter_name: 适配器名称
            
        Returns:
            添加PEFT适配器后的模型
        """
        model = get_peft_model(model, peft_config, adapter_name)
        
        model.print_trainable_parameters()
        
        logger.info(f"PEFT模型创建完成，可训练参数: {model.num_parameters(only_trainable=True)}")
        
        return model
    
    @staticmethod
    def compare_peft_methods() -> Dict[str, Dict]:
        """
        比较不同PEFT方法的特点
        
        Returns:
            各方法的参数量、适用场景对比
        """
        comparison = {
            "LoRA": {
                "参数量": "约0.1%-1%原始参数",
                "原理": "低秩矩阵分解",
                "优点": "效果好，可组合，无推理延迟",
                "缺点": "需要选择目标层",
                "适用": "大多数场景，推荐首选"
            },
            "Prefix Tuning": {
                "参数量": "约0.1%-5%",
                "原理": "虚拟token引导",
                "优点": "不修改原模型",
                "缺点": "对超参数敏感",
                "适用": "生成任务"
            },
            "Prompt Tuning": {
                "参数量": "仅embedding层",
                "原理": "软提示向量",
                "优点": "参数量最少",
                "缺点": "效果可能受限",
                "适用": "超大规模模型"
            },
            "AdaLoRA": {
                "参数量": "与LoRA相当",
                "原理": "自适应秩分配",
                "优点": "自动优化参数分配",
                "缺点": "实现复杂",
                "适用": "需要精细控制的场景"
            },
            "IA3": {
                "参数量": "极小（仅缩放向量）",
                "原理": "激活向量缩放",
                "优点": "参数量最少，效果好",
                "缺点": "适用范围有限",
                "适用": "资源极度受限场景"
            }
        }
        
        for method, details in comparison.items():
            logger.info(f"\n{method}:")
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
        
        return comparison


class QLoRAImplementation:
    """
    QLoRA实现类 - 消费级GPU微调大模型
    
    QLoRA（Quantized LoRA）结合了4位量化和LoRA技术，
    可以在消费级GPU（如RTX 3090）上微调65B参数模型。
    
    核心技术：
    - 4位NormalFloat量化（NF4）
    - 双重量化（量化量化器）
    - 分页优化器状态
    - LoRA适配器
    """
    
    @staticmethod
    def explain_qlora_principles():
        """讲解QLoRA的原理"""
        principles = {
            "核心技术": [
                "4位NormalFloat（NF4）量化",
                "双重量化减少显存",
                "分页AdamW优化器",
                "LoRA适配器注入"
            ],
            "显存分析": {
                "65B模型FP16": "约130GB",
                "65B模型4-bit": "约32GB",
                "添加LoRA后": "约34GB",
                "消费级GPU": "24GB显存可运行"
            },
            "精度权衡": {
                "FP16": "最高精度，显存占用大",
                "Int8": "精度损失小，显存减半",
                "NF4": "4位量化，精度可控损失"
            }
        }
        
        for section, content in principles.items():
            logger.info(f"\n{section}:")
            if isinstance(content, list):
                for item in content:
                    logger.info(f"  - {item}")
            elif isinstance(content, dict):
                for key, value in content.items():
                    logger.info(f"  {key}: {value}")
        
        return principles
    
    @staticmethod
    def create_qlora_config(
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        bias: str = "none",
        task_type: TaskType = TaskType.CAUSAL_LM
    ) -> Tuple[LoraConfig, LoftQConfig]:
        """
        创建QLoRA配置
        
        Args:
            r: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: Dropout概率
            target_modules: 目标模块
            bias: 偏置处理
            task_type: 任务类型
            
        Returns:
            (LoraConfig, LoftQConfig) 元组
        """
        lora_config = LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["query", "key", "value", "dense"],
            bias=bias,
            inference_mode=False
        )
        
        loftq_config = LoftQConfig(
            loftq_bits=4,
            loftq_iter=1
        )
        
        logger.info(f"QLoRA配置创建完成: r={r}, alpha={lora_alpha}")
        
        return lora_config, loftq_config
    
    @staticmethod
    def prepare_model_for_qlora(
        model: AutoModelForCausalLM,
        use_gradient_checkpointing: bool = True,
        use_8bit: bool = False
    ) -> AutoModelForCausalLM:
        """
        准备QLoRA训练模型
        
        Args:
            model: 原始模型
            use_gradient_checkpointing: 使用梯度检查点
            use_8bit: 使用8位量化
            
        Returns:
            准备好的模型
        """
        model = prepare_model_for_int8_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        logger.info("QLoRA模型准备完成")
        
        return model


class EfficientFinetuner:
    """
    高效微调器类 - 完整微调流程封装
    """
    
    def __init__(self, model_name: str, peft_method: str = "lora"):
        """
        初始化高效微调器
        
        Args:
            model_name: 预训练模型名称
            peft_method: PEFT方法（lora, prefix, prompt, adalora, ia3）
        """
        self.model_name = model_name
        self.peft_method = peft_method
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_tokenizer(self):
        """加载分词器"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"分词器加载完成: {self.model_name}")
    
    def load_base_model(
        self,
        task_type: TaskType = TaskType.SEQ_CLS,
        num_labels: int = None
    ):
        """加载基础模型"""
        model_configs = {
            TaskType.SEQ_CLS: AutoModelForSequenceClassification,
            TaskType.CAUSAL_LM: AutoModelForCausalLM,
        }
        
        model_class = model_configs.get(task_type, AutoModel)
        
        if task_type == TaskType.SEQ_CLS and num_labels:
            self.model = model_class.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
        else:
            self.model = model_class.from_pretrained(self.model_name)
        
        logger.info(f"基础模型加载完成: {self.model_name}")
    
    def apply_peft(
        self,
        config: Any = None,
        **kwargs
    ) -> PeftModel:
        """
        应用PEFT配置
        
        Args:
            config: PEFT配置，如果为None则使用默认配置
            
        Returns:
            PEFT模型
        """
        if config is None:
            if self.peft_method == "lora":
                config = PEFTIntegration.get_lora_config(
                    target_modules=kwargs.get("target_modules", ["query", "value"])
                )
            elif self.peft_method == "prefix":
                config = PEFTIntegration.get_prefix_tuning_config()
            elif self.peft_method == "prompt":
                config = PEFTIntegration.get_prompt_tuning_config()
            elif self.peft_method == "adalora":
                config = PEFTIntegration.get_adalora_config()
            elif self.peft_method == "ia3":
                config = PEFTIntegration.get_ia3_config()
        
        self.peft_model = PEFTIntegration.apply_peft_model(
            self.model, config
        )
        
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset = None,
        output_dir: str = "./peft_output",
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_epochs: int = 3,
        **kwargs
    ) -> Trainer:
        """创建Trainer"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            gradient_accumulation_steps=kwargs.get("gradient_accumulation", 1),
            warmup_ratio=0.1,
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=42,
        )
        
        self.trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            **kwargs
        )
        
        return self.trainer
    
    def train(self, resume: bool = False):
        """执行训练"""
        logger.info(f"开始{self.peft_method.upper()}微调...")
        result = self.trainer.train(resume_from_checkpoint=resume)
        return result
    
    def save_adapter(self, output_dir: str):
        """保存PEFT适配器"""
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"适配器已保存到: {output_dir}")
    
    def load_adapter(self, adapter_path: str, adapter_name: str = "default"):
        """加载PEFT适配器"""
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            adapter_name=adapter_name
        )
        logger.info(f"适配器从{adapter_path}加载完成")
    
    def merge_and_save(self, output_dir: str):
        """合并LoRA权重并保存完整模型"""
        if self.peft_method == "lora":
            self.peft_model = self.peft_model.merge_and_unload()
        
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"合并后的模型已保存到: {output_dir}")


class MultiTaskPEFT:
    """
    多任务PEFT类 - 演示如何在同一基础模型上管理多个任务
    
    优势：
    - 共享基础模型参数
    - 每个任务独立适配器
    - 快速切换不同任务
    - 节省显存和存储
    """
    
    def __init__(self, base_model_name: str):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.adapters = {}
    
    def add_task_adapter(
        self,
        task_name: str,
        task_data: Dataset,
        peft_config: LoraConfig,
        output_dir: str
    ):
        """
        为新任务添加适配器
        
        Args:
            task_name: 任务名称
            task_data: 任务训练数据
            peft_config: PEFT配置
            output_dir: 输出目录
        """
        peft_model = get_peft_model(self.base_model, peft_config, adapter_name=task_name)
        
        trainer = Trainer(
            model=peft_model,
            train_dataset=task_data,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                learning_rate=1e-4,
                num_train_epochs=3,
                fp16=torch.cuda.is_available()
            )
        )
        
        trainer.train()
        
        peft_model.save_pretrained(f"{output_dir}/adapter")
        
        self.adapters[task_name] = {
            "model": peft_model,
            "output_dir": output_dir
        }
        
        logger.info(f"任务 {task_name} 适配器训练完成")
    
    def switch_task(self, task_name: str):
        """
        切换到指定任务
        
        Args:
            task_name: 目标任务名称
        """
        if task_name not in self.adapters:
            raise ValueError(f"任务 {task_name} 不存在")
        
        self.current_task = task_name
        
        adapter_path = f"{self.adapters[task_name]['output_dir']}/adapter"
        self.active_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            adapter_name=task_name
        )
        
        logger.info(f"已切换到任务: {task_name}")
        
        return self.active_model
    
    def inference(self, text: str) -> Dict[str, Any]:
        """
        对当前任务进行推理
        
        Args:
            text: 输入文本
            
        Returns:
            推理结果
        """
        if not hasattr(self, 'active_model'):
            raise ValueError("请先调用 switch_task 切换任务")
        
        self.active_model.eval()
        
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.active_model(**inputs)
        
        return {"logits": outputs.logits}


class PEFTVisualization:
    """
    PEFT可视化类 - 帮助理解不同PEFT方法的效果
    """
    
    @staticmethod
    def visualize_parameter_distribution():
        """可视化不同PEFT方法的参数量分布"""
        methods = {
            "Full Fine-tuning": {"total": 100, "trainable": 100, "frozen": 0},
            "LoRA (r=8)": {"total": 100, "trainable": 0.1, "frozen": 99.9},
            "LoRA (r=64)": {"total": 100, "trainable": 0.8, "frozen": 99.2},
            "Prefix (20 tokens)": {"total": 100, "trainable": 0.5, "frozen": 99.5},
            "Prompt (20 tokens)": {"total": 100, "trainable": 0.01, "frozen": 99.99},
            "IA3": {"total": 100, "trainable": 0.05, "frozen": 99.95},
            "AdaLoRA": {"total": 100, "trainable": 0.3, "frozen": 99.7}
        }
        
        logger.info("\n参数量分布对比（以完整模型为100%）：")
        logger.info("-" * 60)
        logger.info(f"{'方法':<25} {'可训练参数%':<15} {'冻结参数%':<15}")
        logger.info("-" * 60)
        
        for method, distribution in methods.items():
            logger.info(f"{method:<25} {distribution['trainable']:<15.2f} {distribution['frozen']:<15.2f}")
        
        return methods


def demo_lora_peft_finetuning():
    """LoRA/PEFT微调演示主函数"""
    print("=" * 60)
    print("LoRA/PEFT 高效微调实战演示")
    print("=" * 60)
    
    print("\n1. LoRA数学原理讲解")
    LoraImplementation.explain_lora_mathematics()
    
    print("\n2. PEFT方法对比")
    PEFTIntegration.compare_peft_methods()
    
    print("\n3. QLoRA原理讲解")
    QLoRAImplementation.explain_qlora_principles()
    
    print("\n4. 参数分布可视化")
    PEFTVisualization.visualize_parameter_distribution()
    
    print("\n" + "=" * 60)
    print("LoRA/PEFT 高效微调实战演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_lora_peft_finetuning()
