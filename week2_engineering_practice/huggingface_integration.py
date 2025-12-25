"""
Hugging Face 集成教程 - Week2 工程实践

本模块涵盖Hugging Face Transformers库的核心使用，包括：
1. AutoModel/AutoTokenizer自动加载机制
2. 模型配置和自定义
3. Pipeline推理接口
4. Trainer API高效微调
5. 训练参数配置
6. 模型保存和推理最佳实践

Author: learn_llm
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, AutoPipeline,
    BitsAndBytesConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import OptimizerNames
from datasets import Dataset as HFDataset
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
import logging
import os
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HuggingFaceBasics:
    """
    Hugging Face基础类 - 演示模型和分词器的加载与使用
    
    核心概念:
    - AutoModel: 自动根据配置加载预训练模型
    - AutoTokenizer: 自动加载对应模型的分词器
    - AutoConfig: 自动加载模型配置
    
    优势:
    - 统一接口：不同模型使用相同的API
    - 丰富生态：支持数千个预训练模型
    - 易于切换：轻松切换不同模型架构
    """
    
    @staticmethod
    def load_pretrained_model(
        model_name: str = "bert-base-chinese",
        device: str = "auto"
    ) -> Tuple[Any, Any]:
        """
        加载预训练模型和分词器
        
        Args:
            model_name: 模型名称或路径
            device: 设备选择，"auto"自动选择GPU/CPU
            
        Returns:
            模型和分词器
        """
        logger.info(f"加载模型: {model_name}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"分词器加载完成，词表大小: {tokenizer.vocab_size}")
        
        # 加载模型
        model = AutoModel.from_pretrained(model_name)
        logger.info(f"模型加载完成，参数量: {model.num_parameters() / 1e6:.2f}M")
        
        # 移动到设备
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model, tokenizer
    
    @staticmethod
    def load_model_for_task(
        model_name: str,
        task: str = "text-classification",
        num_labels: int = None
    ) -> Any:
        """
        为特定任务加载模型
        
        支持的任务:
        - text-classification
        - token-classification (NER)
        - question-answering
        - text-generation
        - translation
        - summarization
        
        Args:
            model_name: 模型名称
            task: 任务类型
            num_labels: 分类标签数（用于分类任务）
        """
        task_models = {
            'text-classification': AutoModelForSequenceClassification,
            'text-generation': AutoModelForCausalLM,
            'token-classification': None,
            'question-answering': None,
        }
        
        if task not in task_models:
            raise ValueError(f"不支持的任务: {task}")
        
        model_class = task_models[task]
        
        if task == 'text-classification':
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            model = model_class.from_pretrained(model_name, config=config)
        else:
            model = model_class.from_pretrained(model_name)
        
        logger.info(f"{task} 模型加载完成")
        return model
    
    @staticmethod
    def model_config_demo(model_name: str = "bert-base-chinese"):
        """模型配置演示"""
        config = AutoConfig.from_pretrained(model_name)
        
        logger.info("模型配置信息:")
        logger.info(f"  模型类型: {config.model_type}")
        logger.info(f"  隐藏层大小: {config.hidden_size}")
        logger.info(f"  注意力头数: {config.num_attention_heads}")
        logger.info(f"  层数: {config.num_hidden_layers}")
        logger.info(f"  词表大小: {config.vocab_size}")
        logger.info(f"  最大位置编码: {config.max_position_embeddings}")
        
        return config


class TokenizerUsage:
    """
    分词器使用类 - 演示Tokenizer的各种功能
    
    核心功能:
    - encode: 文本到ID
    - decode: ID到文本
    - batch_encode_plus: 批量编码
    - padding/truncation策略
    """
    
    @staticmethod
    def basic_tokenization(tokenizer: Any, texts: List[str]):
        """
        基本分词操作
        
        Args:
            tokenizer: 分词器实例
            texts: 文本列表
        """
        # 单文本编码
        single_result = tokenizer("你好，世界！")
        logger.info(f"单文本编码结果: {single_result.keys()}")
        
        # 批量编码（推荐）
        batch_result = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        logger.info(f"批量编码结果形状: {batch_result['input_ids'].shape}")
        
        # 返回attention_mask
        logger.info(f"Attention mask: {batch_result['attention_mask']}")
        
        return batch_result
    
    @staticmethod
    def decode_tokens(tokenizer: Any, input_ids: torch.Tensor):
        """
        解码token IDs为文本
        
        Args:
            tokenizer: 分词器实例
            input_ids: token ID序列
        """
        # 解码单个序列
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logger.info(f"解码结果: {text}")
        
        # 解码时保留特殊token
        text_with_special = tokenizer.decode(input_ids[0])
        logger.info(f"含特殊token: {text_with_special}")
        
        return text
    
    @staticmethod
    def special_tokens_demo(tokenizer: Any):
        """
        特殊token演示
        
        特殊token包括:
        - [PAD]: 填充token
        - [UNK]: 未知词token
        - [CLS]: 分类token
        - [SEP]: 分隔token
        - [MASK]: 掩码token
        """
        logger.info(f"Padding token: {tokenizer.pad_token}")
        logger.info(f"Unknown token: {tokenizer.unk_token}")
        logger.info(f"Cls token: {tokenizer.cls_token}")
        logger.info(f"Sep token: {tokenizer.sep_token}")
        logger.info(f"Mask token: {tokenizer.mask_token}")
        
        # 获取特殊token ID
        special_ids = {
            'pad': tokenizer.pad_token_id,
            'unk': tokenizer.unk_token_id,
            'cls': tokenizer.cls_token_id,
            'sep': tokenizer.sep_token_id,
            'mask': tokenizer.mask_token_id
        }
        logger.info(f"特殊token IDs: {special_ids}")
        
        return special_ids
    
    @staticmethod
    def encode_plus_detailed(tokenizer: Any, text: str):
        """
        详细编码选项
        
        Returns:
            包含各种编码信息的字典
        """
        # 完整编码选项
        result = tokenizer.encode_plus(
            text,
            add_special_tokens=True,      # 添加[CLS]和[SEP]
            max_length=100,               # 最大长度
            padding='max_length',         # 填充策略
            truncation=True,              # 截断策略
            return_tensors='pt',          # 返回格式
            return_token_type_ids=True,   # 返回token类型
            return_attention_mask=True,   # 返回attention mask
            return_overflowing_tokens=False,  # 返回溢出tokens
            return_special_tokens_mask=True   # 返回特殊token掩码
        )
        
        logger.info(f"Input IDs: {result['input_ids']}")
        logger.info(f"Token Type IDs: {result['token_type_ids']}")
        logger.info(f"Attention Mask: {result['attention_mask']}")
        logger.info(f"Special Tokens Mask: {result['special_tokens_mask']}")
        
        return result


class PipelineUsage:
    """
    Pipeline使用类 - 端到端推理接口
    
    Pipeline特点:
    - 零代码推理
    - 预定义任务接口
    - 支持GPU加速
    - 批量推理优化
    """
    
    @staticmethod
    def create_pipeline(
        task: str = "sentiment-analysis",
        model: Optional[str] = None,
        device: int = -1
    ) -> Any:
        """
        创建Pipeline
        
        Args:
            task: 任务类型
            model: 模型名称或路径
            device: 设备ID，-1表示CPU
            
        Returns:
            Pipeline实例
        """
        classifier = pipeline(
            task,
            model=model,
            device=device
        )
        
        logger.info(f"Pipeline创建成功: {task}")
        return classifier
    
    @staticmethod
    def run_pipeline_inference(pipeline: Any, texts: List[str]):
        """
        执行Pipeline推理
        
        Args:
            pipeline: Pipeline实例
            texts: 输入文本列表
        """
        # 单文本推理
        single_result = pipeline(texts[0])
        logger.info(f"单文本结果: {single_result}")
        
        # 批量推理
        batch_results = pipeline(texts, batch_size=2)
        logger.info(f"批量结果数量: {len(batch_results)}")
        
        return batch_results
    
    @staticmethod
    def available_pipelines() -> Dict[str, str]:
        """
        可用的Pipeline任务类型
        
        Returns:
            任务类型字典
        """
        tasks = {
            "text-classification": "文本分类（如情感分析）",
            "token-classification": "token分类（如命名实体识别）",
            "question-answering": "问答系统",
            "fill-mask": "掩码填充",
            "summarization": "文本摘要",
            "translation": "机器翻译",
            "text-generation": "文本生成",
            "text2text-generation": "文本到文本生成",
            "conversational": "对话系统"
        }
        
        for task, desc in tasks.items():
            logger.info(f"  {task}: {desc}")
        
        return tasks


class TrainerAPI:
    """
    Trainer API类 - Hugging Face的高效训练工具
    
    Trainer优势:
    - 简化训练循环
    - 内置日志和评估
    - 支持混合精度训练
    - 支持分布式训练
    - 丰富的回调机制
    """
    
    @staticmethod
    def create_training_arguments(
        output_dir: str = "./output",
        per_device_train_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 3,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        save_total_limit: int = 3,
        fp16: bool = True,
        report_to: str = "none"
    ) -> TrainingArguments:
        """
        创建训练参数
        
        重要参数:
        - output_dir: 模型保存目录
        - per_device_train_batch_size: 每设备训练批大小
        - gradient_accumulation_steps: 梯度累积步数
        - learning_rate: 学习率
        - num_train_epochs: 训练轮数
        - warmup_steps: 预热步数
        - fp16: 混合精度训练
        - logging_steps: 日志记录步数
        - eval_steps: 评估步数
        - save_steps: 模型保存步数
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=fp16 and torch.cuda.is_available(),
            report_to=report_to,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            no_cuda=False,
            seed=42,
            data_seed=42,
            run_name="llm-training",
        )
        
        logger.info("训练参数创建完成")
        return training_args
    
    @staticmethod
    def create_trainer(
        model: Any,
        training_args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any = None,
        data_collator: Any = None,
        tokenizer: Any = None,
        compute_metrics: Any = None,
        callbacks: List[Any] = None
    ) -> Trainer:
        """
        创建Trainer实例
        
        Args:
            model: 待训练模型
            training_args: 训练参数
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            data_collator: 数据整理器
            tokenizer: 分词器
            compute_metrics: 评估指标计算函数
            callbacks: 回调函数列表
        """
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        logger.info("Trainer创建完成")
        return trainer
    
    @staticmethod
    def train_model(trainer: Trainer, resume_from_checkpoint: bool = True) -> Any:
        """
        训练模型
        
        Args:
            trainer: Trainer实例
            resume_from_checkpoint: 是否从检查点恢复
            
        Returns:
            训练后的模型
        """
        # 检查是否存在检查点
        checkpoint = None
        if resume_from_checkpoint:
            last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
            if last_checkpoint:
                checkpoint = last_checkpoint
                logger.info(f"从检查点恢复: {checkpoint}")
        
        # 开始训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        logger.info(f"训练完成，最终损失: {train_result.training_loss}")
        
        # 保存模型
        trainer.save_model()
        logger.info(f"模型已保存到: {trainer.args.output_dir}")
        
        return train_result
    
    @staticmethod
    def evaluate_model(trainer: Trainer, eval_dataset: Any = None) -> Dict:
        """
        评估模型
        
        Args:
            trainer: Trainer实例
            eval_dataset: 评估数据集
            
        Returns:
            评估指标字典
        """
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"评估指标: {metrics}")
        return metrics


class FineTuningExamples:
    """
    微调示例类 - 提供完整的微调流程
    """
    
    @staticmethod
    def finetune_bert_for_classification(
        model_name: str = "bert-base-chinese",
        train_texts: List[str] = None,
        train_labels: List[int] = None,
        output_dir: str = "./classification_model"
    ):
        """
        分类任务微调BERT
        
        Args:
            model_name: 预训练模型名称
            train_texts: 训练文本列表
            train_labels: 训练标签列表
            output_dir: 输出目录
        """
        logger.info("开始分类任务微调")
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = len(set(train_labels)) if train_labels else 2
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
        # 准备数据集
        class TextClassificationDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
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
        
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
        
        # 创建训练参数
        training_args = TrainerAPI.create_training_arguments(
            output_dir=output_dir,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=3,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            eval_steps=500
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        
        # 训练
        trainer.train()
        
        return trainer
    
    @staticmethod
    def finetune_for_causal_lm(
        model_name: str = "gpt2",
        train_texts: List[str] = None,
        output_dir: str = "./causal_lm_model",
        block_size: int = 128
    ):
        """
        因果语言模型微调（如GPT）
        
        Args:
            model_name: 预训练模型名称
            train_texts: 训练文本列表
            output_dir: 输出目录
            block_size: 文本块大小
        """
        logger.info("开始因果语言模型微调")
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # 准备数据集
        def tokenize_function(examples):
            output = tokenizer(
                examples["text"],
                truncation=True,
                max_length=block_size,
                padding='max_length'
            )
            output["labels"] = output["input_ids"].copy()
            return output
        
        # 转换为HuggingFace Dataset格式
        hf_dataset = HFDataset.from_dict({"text": train_texts})
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # 数据整理器（用于MLM）
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # 因果语言模型不使用MLM
        )
        
        # 训练参数
        training_args = TrainerAPI.create_training_arguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            learning_rate=3e-5,
            num_train_epochs=3,
            warmup_steps=200,
            logging_steps=100,
            save_steps=1000
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # 训练
        trainer.train()
        
        return trainer


class ModelSavingAndLoading:
    """
    模型保存和加载类
    """
    
    @staticmethod
    def save_full_model(model: Any, output_dir: str, tokenizer: Any = None):
        """
        保存完整模型（包含分词器）
        
        保存内容:
        - 模型权重
        - 配置文件
        - 分词器文件
        - 词汇表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model.save_pretrained(output_dir)
        logger.info(f"模型已保存到: {output_dir}")
        
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
            logger.info(f"分词器已保存到: {output_dir}")
    
    @staticmethod
    def save_model_weights_only(model: Any, output_path: str):
        """
        只保存模型权重（用于轻量级部署）
        """
        torch.save(model.state_dict(), output_path)
        logger.info(f"模型权重已保存到: {output_path}")
    
    @staticmethod
    def load_model_from_pretrained(output_dir: str) -> Tuple[Any, Any]:
        """
        从保存的目录加载模型和分词器
        """
        model = AutoModel.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        logger.info(f"模型从 {output_dir} 加载成功")
        return model, tokenizer
    
    @staticmethod
    def load_model_weights(model: Any, weights_path: str) -> Any:
        """
        只加载模型权重
        """
        model.load_state_dict(torch.load(weights_path))
        logger.info(f"权重从 {weights_path} 加载成功")
        return model


class QuantizationConfig:
    """
    量化配置类 - 用于模型压缩和加速
    
    量化类型:
    - 8-bit量化 (INT8)
    - 4-bit量化 (INT4)
    
    优点:
    - 减少显存占用
    - 加速推理
    - 保持模型质量
    """
    
    @staticmethod
    def get_8bit_config() -> BitsAndBytesConfig:
        """
        获取8位量化配置
        """
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=False
        )
    
    @staticmethod
    def get_4bit_config() -> BitsAndBytesConfig:
        """
        获取4位量化配置
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
    
    @staticmethod
    def load_quantized_model(
        model_name: str,
        quantization: str = "4bit"
    ) -> Any:
        """
        加载量化模型
        
        Args:
            model_name: 模型名称
            quantization: 量化类型，"4bit"或"8bit"
        """
        if quantization == "4bit":
            quantization_config = QuantizationConfig.get_4bit_config()
        else:
            quantization_config = QuantizationConfig.get_8bit_config()
        
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        logger.info(f"量化模型加载成功: {quantization}")
        return model


class InferenceOptimization:
    """
    推理优化类 - 提供高效的推理方法
    """
    
    @staticmethod
    @torch.no_grad()
    def optimize_inference(
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        优化推理过程
        
        优化策略:
        1. 使用torch.no_grad()
        2. 梯度检查点
        3. 混合精度推理
        """
        model.eval()
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        with torch.cuda.amp.autocast():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return output
    
    @staticmethod
    def generate_text(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        文本生成
        
        Args:
            model: 语言模型
            tokenizer: 分词器
            prompt: 提示词
            max_length: 最大生成长度
            num_return_sequences: 返回序列数
            do_sample: 是否采样
            temperature: 温度参数
            top_p: Top-p采样阈值
        """
        model.eval()
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length + input_ids.shape[1],
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_texts = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output
        ]
        
        return generated_texts


class HuggingFaceIntegrationDemo:
    """
    Hugging Face集成演示类
    """
    
    @staticmethod
    def demo_pipeline():
        """Pipeline演示"""
        logger.info("演示Pipeline使用")
        
        try:
            sentiment_pipeline = PipelineUsage.create_pipeline(
                task="sentiment-analysis",
                model="uer/roberta-base-finetuned-chinanews-chinese"
            )
            
            test_texts = [
                "这部电影真是太棒了！",
                "服务态度很差，体验不好",
                "产品还可以，中规中矩"
            ]
            
            results = PipelineUsage.run_pipeline_inference(sentiment_pipeline, test_texts)
            
            return results
        except Exception as e:
            logger.error(f"Pipeline演示失败: {e}")
            return None
    
    @staticmethod
    def demo_model_loading():
        """模型加载演示"""
        logger.info("演示模型加载")
        
        model_name = "bert-base-chinese"
        
        config = HuggingFaceBasics.model_config_demo(model_name)
        model, tokenizer = HuggingFaceBasics.load_pretrained_model(model_name)
        
        # 测试推理
        text = "今天天气真好"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logger.info(f"模型推理成功，输出形状: {outputs.last_hidden_state.shape}")
        
        return model, tokenizer
    
    @staticmethod
    def demo_inference_optimization():
        """推理优化演示"""
        logger.info("演示推理优化")
        
        model_name = "bert-base-chinese"
        model, tokenizer = HuggingFaceBasics.load_pretrained_model(model_name)
        
        text = "这是一个测试句子"
        inputs = tokenizer(text, return_tensors="pt")
        
        # 优化推理
        start_time = time.time()
        output = InferenceOptimization.optimize_inference(
            model,
            inputs['input_ids'],
            inputs.get('attention_mask')
        )
        elapsed = time.time() - start_time
        
        logger.info(f"推理耗时: {elapsed*1000:.2f}ms")
        
        return output


def demo_huggingface_integration():
    """Hugging Face集成演示主函数"""
    print("=" * 60)
    print("Hugging Face 集成教程演示")
    print("=" * 60)
    
    print("\n1. Pipeline演示")
    HuggingFaceIntegrationDemo.demo_pipeline()
    
    print("\n2. 模型加载演示")
    HuggingFaceIntegrationDemo.demo_model_loading()
    
    print("\n3. 推理优化演示")
    HuggingFaceIntegrationDemo.demo_inference_optimization()
    
    print("\n" + "=" * 60)
    print("Hugging Face 集成教程演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_huggingface_integration()
