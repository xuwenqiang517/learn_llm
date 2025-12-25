"""
大语言模型评估指标实现 - Week 3 微调实践

本模块涵盖大语言模型评估的完整实现，包括：
1. 困惑度（Perplexity）计算
2. BLEU分数计算
3. ROUGE系列指标
4. 精确率、召回率、F1分数
5. 语义相似度评估
6. 综合评估框架

Author: learn_llm
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from math import exp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置类"""
    max_length: int = 512
    stride: int = 512
    device: str = "cuda"
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    clean_up_tokenization_spaces: bool = True


class PerplexityEvaluator:
    """
    困惑度评估器类
    
    困惑度（Perplexity）是衡量语言模型预测能力的核心指标。
    它表示模型对测试数据的平均不确定性，值越低表示模型越好。
    
    数学定义：
    PPL = exp(-1/N * Σ log P(w_i | w_1, ..., w_{i-1}))
    
    特点：
    - 对语言生成任务特别有用
    - 越低的困惑度表示模型越能准确预测下一个词
    - 可用于比较不同语言模型的性能
    - 对短文本可能不够稳定
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: EvaluationConfig = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    @torch.no_grad()
    def calculate_perplexity(
        self,
        texts: List[str],
        batch_size: int = 1,
        return_individual: bool = False
    ) -> Union[float, Tuple[float, List[float]]]:
        """
        计算文本列表的困惑度
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            return_individual: 是否返回每个文本的困惑度
            
        Returns:
            平均困惑度，或（平均困惑度，每个文本困惑度列表）
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        individual_ppls = []
        
        for text in texts:
            encodings = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors
            )
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss.item()
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
            ppl = exp(loss)
            individual_ppls.append(ppl)
        
        avg_ppl = exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        
        if return_individual:
            return avg_ppl, individual_ppls
        
        return avg_ppl
    
    @torch.no_grad()
    def calculate_ppl_with_stride(
        self,
        text: str,
        stride: int = None
    ) -> float:
        """
        使用滑动窗口计算长文本的困惑度
        
        适用于超过模型最大长度的长文本
        
        Args:
            text: 输入文本
            stride: 滑动窗口步长
            
        Returns:
            困惑度值
        """
        self.model.eval()
        
        stride = stride or self.config.stride
        encodings = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors=self.config.return_tensors
        )
        
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        seq_length = input_ids.size(1)
        
        losses = []
        weights = []
        
        for start in range(0, seq_length, stride):
            end = min(start + self.config.max_length, seq_length)
            
            chunk_input_ids = input_ids[:, start:end]
            chunk_attention = attention_mask[:, start:end]
            
            outputs = self.model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention,
                labels=chunk_input_ids
            )
            
            losses.append(outputs.loss.item())
            weights.append(chunk_input_ids.size(1))
        
        total_loss = sum(l * w for l, w in zip(losses, weights))
        total_weight = sum(weights)
        
        ppl = exp(total_loss / total_weight) if total_weight > 0 else float('inf')
        
        logger.info(f"长文本困惑度计算完成: PPL = {ppl:.4f}")
        
        return ppl


class BLEUEvaluator:
    """
    BLEU分数评估器类
    
    BLEU（Bilingual Evaluation Understudy）是一种基于n-gram重叠的
    评估指标，广泛用于机器翻译和文本生成任务。
    
    计算公式：
    BLEU = BP * exp(Σ w_n * log p_n)
    
    其中：
    - BP是惩罚因子（Brevity Penalty）
    - p_n是n-gram精确率
    - w_n是权重（通常均匀分配）
    
    特点：
    - 取值范围0到1
    - 简单高效，广泛使用
    - 对词序变化不敏感
    - 可能忽略语义准确性
    """
    
    def __init__(self, smooth: bool = True, smooth_method: str = "epsilon"):
        """
        初始化BLEU评估器
        
        Args:
            smooth: 是否使用平滑
            smooth_method: 平滑方法（epsilon, floor, etc.）
        """
        self.smooth = smooth
        self.smooth_func = SmoothingFunction().method1 if smooth else None
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词（字符级）
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的列表
        """
        return list(text)
    
    def calculate_bleu(
        self,
        references: List[str],
        candidates: List[str],
        max_n: int = 4,
        weights: List[float] = None,
        language: str = "chinese"
    ) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            max_n: 最大n-gram阶数
            weights: 各阶权重
            language: 语言类型（chinese, english）
            
        Returns:
            包含各阶BLEU分数的字典
        """
        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量必须相同")
        
        if weights is None:
            weights = [1.0 / max_n] * max_n
        
        tokenizer = self.tokenize_chinese if language == "chinese" else str.split
        
        total_scores = {f"bleu_{i}": [] for i in range(1, max_n + 1)}
        total_scores["bleu"] = []
        
        for ref, cand in zip(references, candidates):
            ref_tokens = [tokenizer(ref)]
            cand_tokens = tokenizer(cand)
            
            try:
                scores = []
                for n in range(1, max_n + 1):
                    score = sentence_bleu(
                        ref_tokens,
                        cand_tokens,
                        weights=weights[:n],
                        smoothing_function=self.smooth_func
                    )
                    scores.append(score)
                    total_scores[f"bleu_{n}"].append(score)
                
                final_bleu = sentence_bleu(
                    ref_tokens,
                    cand_tokens,
                    weights=weights,
                    smoothing_function=self.smooth_func
                )
                total_scores["bleu"].append(final_bleu)
                
            except Exception as e:
                logger.warning(f"BLEU计算警告: {e}")
                for n in range(1, max_n + 1):
                    total_scores[f"bleu_{n}"].append(0.0)
                total_scores["bleu"].append(0.0)
        
        result = {}
        for key, values in total_scores.items():
            result[key] = np.mean(values) if values else 0.0
        
        logger.info(f"BLEU评估完成: BLEU-4 = {result.get('bleu_4', 0):.4f}")
        
        return result
    
    def calculate_corpus_bleu(
        self,
        references: List[List[str]],
        candidates: List[str],
        max_n: int = 4
    ) -> float:
        """
        计算语料库级BLEU分数
        
        Args:
            references: 参考文本列表（每个可以有多个参考）
            candidates: 候选文本列表
            max_n: 最大n-gram阶数
            
        Returns:
            语料库BLEU分数
        """
        tokenizer = self.tokenize_chinese
        
        ref_tokenized = [[tokenizer(r) for r in ref_group] for ref_group in references]
        cand_tokenized = [tokenizer(c) for c in candidates]
        
        try:
            corpus_bleu = 0.0
            for n in range(1, max_n + 1):
                score = sentence_bleu(
                    ref_tokenized,
                    cand_tokenized,
                    weights=[1.0 / max_n] * n,
                    smoothing_function=self.smooth_func
                )
                corpus_bleu += score
            
            corpus_bleu /= max_n
            
            return corpus_bleu
            
        except Exception as e:
            logger.error(f"语料库BLEU计算错误: {e}")
            return 0.0


class ROUGEEvaluator:
    """
    ROUGE分数评估器类
    
    ROUGE（Recall-Oriented Understudy for Gisting Evaluation）是一组
    基于召回率的评估指标，主要用于文本摘要任务。
    
    主要指标：
    - ROUGE-N: 基于n-gram的召回率
    - ROUGE-L: 基于最长公共子序列
    - ROUGE-W: 加权最长公共子序列
    - ROUGE-S: 基于跳词的n-gram
    
    特点：
    - 与BLEU互补，更注重召回率
    - 对摘要任务特别有用
    - ROUGE-L不需要参数
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 1.0):
        """
        初始化ROUGE评估器
        
        Args:
            alpha: 精确率权重（用于ROUGE-LCS）
            beta: F值的beta参数
        """
        self.alpha = alpha
        self.beta = beta
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        return list(text)
    
    def _lcs(self, s1: List[str], s2: List[str]) -> int:
        """
        计算最长公共子序列长度
        
        Args:
            s1: 序列1
            s2: 序列2
            
        Returns:
            LCS长度
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _rouge_n(self, reference: List[str], candidate: List[str], n: int) -> Dict[str, float]:
        """
        计算ROUGE-N分数
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            n: n-gram大小
            
        Returns:
            包含精确率、召回率、F1的字典
        """
        ref_ngrams = Counter([' '.join(reference[i:i+n]) for i in range(len(reference)-n+1)])
        cand_ngrams = Counter([' '.join(candidate[i:i+n]) for i in range(len(candidate)-n+1)])
        
        overlap = sum((ref_ngrams & cand_ngrams).values())
        
        precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _rouge_l(
        self,
        reference: List[str],
        candidate: List[str]
    ) -> Dict[str, float]:
        """
        计算ROUGE-L分数
        
        Args:
            reference: 参考文本
            candidate: 候选文本
            
        Returns:
            包含精确率、召回率、F1的字典
        """
        lcs_length = self._lcs(reference, candidate)
        
        precision = lcs_length / len(candidate) if candidate else 0
        recall = lcs_length / len(reference) if reference else 0
        
        if self.alpha * precision + (1 - self.alpha) * recall > 0:
            f1 = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        else:
            f1 = 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def calculate_rouge(
        self,
        references: List[str],
        candidates: List[str],
        ngram_range: Tuple[int, int] = (1, 4),
        language: str = "chinese"
    ) -> Dict[str, Dict[str, float]]:
        """
        计算ROUGE分数
        
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            ngram_range: n-gram范围
            language: 语言类型
            
        Returns:
            包含各指标的字典
        """
        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量必须相同")
        
        tokenizer = self.tokenize_chinese if language == "chinese" else str.split
        
        results = {
            "rouge_1": {"precision": [], "recall": [], "f1": []},
            "rouge_2": {"precision": [], "recall": [], "f1": []},
            "rouge_l": {"precision": [], "recall": [], "f1": []}
        }
        
        for ref, cand in zip(references, candidates):
            ref_tokens = tokenizer(ref)
            cand_tokens = tokenizer(cand)
            
            for n in range(ngram_range[0], min(ngram_range[1], 4) + 1):
                rouge_n_result = self._rouge_n(ref_tokens, cand_tokens, n)
                results[f"rouge_{n}"]["precision"].append(rouge_n_result["precision"])
                results[f"rouge_{n}"]["recall"].append(rouge_n_result["recall"])
                results[f"rouge_{n}"]["f1"].append(rouge_n_result["f1"])
            
            rouge_l_result = self._rouge_l(ref_tokens, cand_tokens)
            results["rouge_l"]["precision"].append(rouge_l_result["precision"])
            results["rouge_l"]["recall"].append(rouge_l_result["recall"])
            results["rouge_l"]["f1"].append(rouge_l_result["f1"])
        
        final_results = {}
        for metric, values in results.items():
            final_results[metric] = {
                "precision": np.mean(values["precision"]) if values["precision"] else 0,
                "recall": np.mean(values["recall"]) if values["recall"] else 0,
                "f1": np.mean(values["f1"]) if values["f1"] else 0
            }
        
        logger.info(f"ROUGE评估完成: ROUGE-L F1 = {final_results['rouge_l']['f1']:.4f}")
        
        return final_results


class ClassificationEvaluator:
    """
    分类任务评估器类
    
    支持分类任务的常用评估指标：
    - 精确率（Precision）
    - 召回率（Recall）
    - F1分数
    - 准确率（Accuracy）
    - 混淆矩阵
    """
    
    def __init__(self, average: str = "weighted"):
        """
        初始化分类评估器
        
        Args:
            average: 聚合方式（micro, macro, weighted, None）
        """
        self.average = average
    
    def calculate_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[int] = None
    ) -> Dict[str, float]:
        """
        计算分类评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            
        Returns:
            包含各指标的字典
        """
        metrics = {}
        
        accuracy = accuracy_score(y_true, y_pred)
        metrics["accuracy"] = accuracy
        
        if labels is None:
            labels = list(set(y_true) | set(y_pred))
        
        precision = precision_score(
            y_true, y_pred, average=self.average, labels=labels, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, average=self.average, labels=labels, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, average=self.average, labels=labels, zero_division=0
        )
        
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        
        logger.info(f"分类评估完成: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[int] = None,
        target_names: List[str] = None
    ) -> str:
        """
        获取分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            target_names: 标签名称
            
        Returns:
            格式化的分类报告
        """
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names or [f"class_{i}" for i in labels],
            output_dict=True
        )
        
        return report
    
    def calculate_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[int] = None
    ) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            
        Returns:
            混淆矩阵
        """
        from sklearn.metrics import confusion_matrix
        
        labels = labels or sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return cm


class SemanticSimilarityEvaluator:
    """
    语义相似度评估器类
    
    使用预训练的Sentence Transformer计算语义相似度，
    适用于评估生成文本与参考文本的语义一致性。
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化语义相似度评估器
        
        Args:
            model_name: Sentence Transformer模型名称
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            logger.info(f"语义相似度模型加载完成: {model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers未安装，使用替代方案")
            self.model = None
    
    def calculate_similarity(
        self,
        references: List[str],
        candidates: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        计算参考文本与候选文本的语义相似度
        
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            batch_size: 批处理大小
            
        Returns:
            包含各相似度指标的字典
        """
        if self.model is None:
            logger.error("模型未加载")
            return {"error": "模型未加载"}
        
        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量必须相同")
        
        ref_embeddings = self.model.encode(
            references, batch_size=batch_size, convert_to_tensor=True
        )
        cand_embeddings = self.model.encode(
            candidates, batch_size=batch_size, convert_to_tensor=True
        )
        
        cosine_similarities = F.cosine_similarity(
            ref_embeddings, cand_embeddings, dim=1
        )
        
        similarities = cosine_similarities.cpu().numpy()
        
        result = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "individual_similarities": similarities.tolist()
        }
        
        logger.info(f"语义相似度评估完成: 平均相似度 = {result['mean_similarity']:.4f}")
        
        return result
    
    def calculate_embedding_based_score(
        self,
        references: List[str],
        candidates: List[str],
        score_type: str = "cosine"
    ) -> Dict[str, float]:
        """
        计算基于Embedding的评估分数
        
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            score_type: 分数类型（cosine, euclidean, manhattan）
            
        Returns:
            评估分数
        """
        if self.model is None:
            return {"error": "模型未加载"}
        
        ref_embeddings = self.model.encode(references, convert_to_numpy=True)
        cand_embeddings = self.model.encode(candidates, convert_to_numpy=True)
        
        if score_type == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarities = cosine_similarity(ref_embeddings, cand_embeddings)
            scores = np.diag(similarities)
            
        elif score_type == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            
            distances = euclidean_distances(ref_embeddings, cand_embeddings)
            scores = 1 / (1 + np.diag(distances))
        
        else:
            from sklearn.metrics.pairwise import manhattan_distances
            
            distances = manhattan_distances(ref_embeddings, cand_embeddings)
            scores = 1 / (1 + np.diag(distances))
        
        result = {
            f"mean_{score_type}_score": float(np.mean(scores)),
            f"std_{score_type}_score": float(np.std(scores)),
            "individual_scores": scores.tolist()
        }
        
        return result


class ComprehensiveEvaluator:
    """
    综合评估器类 - 整合多种评估方法
    
    提供统一的评估接口，支持同时计算多个评估指标。
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """
        初始化综合评估器
        
        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.bleu_evaluator = BLEUEvaluator()
        self.rouge_evaluator = ROUGEEvaluator()
        self.classification_evaluator = ClassificationEvaluator()
    
    def evaluate_generation(
        self,
        references: List[str],
        candidates: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        评估生成任务
        
        Args:
            references: 参考文本列表
            candidates: 候选文本列表
            metrics: 需要计算的指标列表
            
        Returns:
            评估结果字典
        """
        if metrics is None:
            metrics = ["bleu", "rouge"]
        
        results = {}
        
        if "bleu" in metrics:
            bleu_results = self.bleu_evaluator.calculate_bleu(
                references, candidates
            )
            results.update(bleu_results)
        
        if "rouge" in metrics:
            rouge_results = self.rouge_evaluator.calculate_rouge(
                references, candidates
            )
            results["rouge"] = rouge_results
        
        return results
    
    def evaluate_classification(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: List[int] = None
    ) -> Dict[str, float]:
        """
        评估分类任务
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            
        Returns:
            评估结果
        """
        return self.classification_evaluator.calculate_metrics(
            y_true, y_pred, labels
        )
    
    def evaluate_task(
        self,
        task_type: str,
        references: List[str],
        candidates: List[str],
        y_true: List[int] = None,
        y_pred: List[int] = None,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        统一评估接口
        
        Args:
            task_type: 任务类型（generation, classification）
            references: 参考文本
            candidates: 候选文本
            y_true: 分类真实标签
            y_pred: 分类预测标签
            metrics: 指标列表
            
        Returns:
            评估结果
        """
        results = {"task_type": task_type}
        
        if task_type == "generation":
            results["generation_metrics"] = self.evaluate_generation(
                references, candidates, metrics
            )
        
        elif task_type == "classification":
            results["classification_metrics"] = self.evaluate_classification(
                y_true, y_pred
            )
        
        else:
            logger.warning(f"未知任务类型: {task_type}")
        
        return results


class EvaluationMetricsDemo:
    """评估指标演示类"""
    
    @staticmethod
    def demo_perplexity():
        """困惑度评估演示"""
        print("\n" + "=" * 50)
        print("困惑度（Perplexity）评估演示")
        print("=" * 50)
        
        print("\n困惑度定义:")
        print("PPL = exp(-1/N * Σ log P(w_i | w_1, ..., w_{i-1}))")
        print("  - 值越低表示模型预测能力越强")
        print("  - 适用于语言模型性能评估")
        print("  - 计算简单但可能对短文本不够稳定")
    
    @staticmethod
    def demo_bleu():
        """BLEU评估演示"""
        print("\n" + "=" * 50)
        print("BLEU分数评估演示")
        print("=" * 50)
        
        print("\nBLEU计算公式:")
        print("BLEU = BP * exp(Σ w_n * log p_n)")
        print("  - BP: Brevity Penalty（长度惩罚）")
        print("  - p_n: n-gram精确率")
        print("  - 范围: 0-1，越高越好")
        
        references = ["今天天气真好", "我喜欢学习人工智能"]
        candidates = ["今天天气不错", "我爱学习AI"]
        
        evaluator = BLEUEvaluator()
        results = evaluator.calculate_bleu(references, candidates)
        
        print(f"\n示例结果:")
        print(f"  BLEU-1: {results.get('bleu_1', 0):.4f}")
        print(f"  BLEU-4: {results.get('bleu_4', 0):.4f}")
    
    @staticmethod
    def demo_rouge():
        """ROUGE评估演示"""
        print("\n" + "=" * 50)
        print("ROUGE分数评估演示")
        print("=" * 50)
        
        print("\n主要指标:")
        print("  ROUGE-N: 基于n-gram的召回率")
        print("  ROUGE-L: 基于最长公共子序列")
        print("  ROUGE-W: 加权最长公共子序列")
        print("  ROUGE-S: 基于跳词的n-gram")
        
        references = ["今天天气很好，适合出去玩"]
        candidates = ["今天天气不错，可以去玩"]
        
        evaluator = ROUGEEvaluator()
        results = evaluator.calculate_rouge(references, candidates)
        
        print(f"\n示例结果:")
        print(f"  ROUGE-1 F1: {results.get('rouge_1', {}).get('f1', 0):.4f}")
        print(f"  ROUGE-L F1: {results.get('rouge_l', {}).get('f1', 0):.4f}")
    
    @staticmethod
    def demo_classification():
        """分类评估演示"""
        print("\n" + "=" * 50)
        print("分类评估演示")
        print("=" * 50)
        
        print("\n主要指标:")
        print("  Precision: 精确率")
        print("  Recall: 召回率")
        print("  F1: 精确率和召回率的调和平均")
        print("  Accuracy: 准确率")
        
        y_true = [0, 1, 1, 0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 1, 0]
        
        evaluator = ClassificationEvaluator()
        results = evaluator.calculate_metrics(y_true, y_pred)
        
        print(f"\n示例结果:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
    
    @staticmethod
    def run_all_demos():
        """运行所有演示"""
        print("=" * 60)
        print("大语言模型评估指标演示")
        print("=" * 60)
        
        EvaluationMetricsDemo.demo_perplexity()
        EvaluationMetricsDemo.demo_bleu()
        EvaluationMetricsDemo.demo_rouge()
        EvaluationMetricsDemo.demo_classification()
        
        print("\n" + "=" * 60)
        print("评估指标演示完成!")
        print("=" * 60)


def demo_evaluation_metrics():
    """评估指标演示主函数"""
    EvaluationMetricsDemo.run_all_demos()


if __name__ == "__main__":
    demo_evaluation_metrics()
