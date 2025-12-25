"""
高效数据管道教程 - Week2 工程实践

本模块涵盖大模型训练中的高效数据处理技术，包括：
1. 自定义Dataset实现和数据加载
2. 动态padding和attention mask处理
3. 内存优化的数据预处理
4. 分布式数据并行
5. 流式数据处理和采样策略
6. Hugging Face datasets库集成

Author: learn_llm
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Iterator, Union, Tuple
import random
import os
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import mmap
import pickle
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DataCollatorConfig:
    """数据整理器配置"""
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    return_attention_mask: bool = True
    return_token_type_ids: bool = False


class TextDataset(Dataset):
    """
    文本数据集类 - 演示自定义Dataset的正确实现
    
    Dataset是PyTorch数据加载的核心抽象类。
    必须实现__len__和__getitem__两个方法。
    
    设计考虑:
    - 惰性加载：避免一次性加载所有数据到内存
    - 缓存机制：对预处理结果进行缓存以提高效率
    - 线程安全：确保多进程加载时的安全性
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: Any,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        overwrite_cache: bool = False
    ):
        """
        初始化文本数据集
        
        Args:
            file_path: 文本文件路径，每行一个样本
            tokenizer: 分词器
            max_length: 最大序列长度
            cache_dir: 缓存目录
            overwrite_cache: 是否覆盖现有缓存
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.overwrite_cache = overwrite_cache
        
        self.examples = self._load_or_cache_examples()
        
    def _get_cache_key(self) -> str:
        """生成缓存键"""
        cache_str = f"{self.file_path}_{self.max_length}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _load_or_cache_examples(self) -> List[Dict[str, Any]]:
        """加载或从缓存读取样本"""
        if self.cache_dir and not self.overwrite_cache:
            cache_key = self._get_cache_key()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                logger.info(f"从缓存加载数据: {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        examples = self._load_examples()
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_key = self._get_cache_key()
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            logger.info(f"缓存数据到: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(examples, f)
        
        return examples
    
    def _load_examples(self) -> List[Dict[str, Any]]:
        """加载并预处理样本"""
        examples = []
        logger.info(f"从 {self.file_path} 加载数据")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                if line_idx % 10000 == 0:
                    logger.info(f"已处理 {line_idx} 行")
                
                tokenized = self.tokenizer(
                    line,
                    max_length=self.max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None
                )
                
                examples.append({
                    'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized.get('attention_mask', [1] * len(tokenized['input_ids'])),
                    'text': line
                })
        
        logger.info(f"共加载 {len(examples)} 个样本")
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        return self.examples[idx]


class StreamingDataset(IterableDataset):
    """
    流式数据集类 - 处理超大数据集
    
    适用场景:
    - 数据量超过内存容量
    - 数据存储在远程服务器
    - 数据需要实时生成
    
    特点:
    - 使用迭代器模式
    - 支持分布式采样
    - 惰性加载数据
    """
    
    def __init__(
        self,
        file_patterns: List[str],
        tokenizer: Any,
        max_length: int = 512,
        shuffle_buffer_size: int = 1000
    ):
        """
        Args:
            file_patterns: 文件路径模式列表
            tokenizer: 分词器
            max_length: 最大序列长度
            shuffle_buffer_size: shuffle缓冲大小
        """
        self.file_patterns = file_patterns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        
        self.files = self._get_files()
        
    def _get_files(self) -> List[str]:
        """获取所有匹配的文件"""
        files = []
        import glob
        for pattern in self.file_patterns:
            files.extend(glob.glob(pattern))
        return sorted(files)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        迭代器实现
        
        使用buffer进行shuffle，需要处理分布式场景下的数据分片
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        files_to_process = self.files[worker_id::num_workers]
        buffer = []
        
        for file_path in files_to_process:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    tokenized = self.tokenizer(
                        line,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                    
                    item = {
                        'input_ids': tokenized['input_ids'],
                        'attention_mask': tokenized.get('attention_mask', [1] * len(tokenized['input_ids']))
                    }
                    
                    if len(buffer) < self.shuffle_buffer_size:
                        buffer.append(item)
                    else:
                        idx = random.randint(0, self.shuffle_buffer_size)
                        yield buffer[idx]
                        buffer[idx] = item
        
        while buffer:
            yield buffer.pop()


class DataCollator:
    """
    数据整理器 - 处理批次内的padding和对齐
    
    重要性:
    在自然语言处理中，不同样本通常长度不同。
    DataLoader需要将它们对齐到批次内最长序列。
    
    策略:
    - 动态padding：对齐到当前批次最长序列
    - 固定padding：对齐到预设最大长度
    - 只pad到max_length
    """
    
    def __init__(self, config: Optional[DataCollatorConfig] = None):
        self.config = config or DataCollatorConfig()
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            examples: 样本列表，每个样本是字典
            
        Returns:
            整理后的批次字典
        """
        if not examples:
            return {}
        
        max_len = self._get_max_length(examples)
        
        if self.config.pad_to_multiple_of:
            max_len = ((max_len + self.config.pad_to_multiple_of - 1) 
                      // self.config.pad_to_multiple_of * self.config.pad_to_multiple_of)
        
        batch = defaultdict(list)
        
        for example in examples:
            for key, value in example.items():
                if key == 'text':
                    continue
                padded = self._pad_sequence(value, max_len)
                batch[key].append(padded)
        
        result = {
            k: torch.tensor(v, dtype=torch.long) 
            for k, v in batch.items()
        }
        
        if self.config.return_attention_mask and 'attention_mask' in examples[0]:
            result['attention_mask'] = torch.tensor(
                [ex.get('attention_mask', [1] * len(ex['input_ids'])) for ex in examples],
                dtype=torch.long
            )
        
        return result
    
    def _get_max_length(self, examples: List[Dict[str, Any]]) -> int:
        """计算批次内最大长度"""
        if self.config.max_length:
            return min(
                max(len(ex.get('input_ids', [])) for ex in examples),
                self.config.max_length
            )
        return max(len(ex.get('input_ids', [])) for ex in examples)
    
    def _pad_sequence(self, sequence: List, max_len: int, pad_value: int = 0) -> List:
        """填充序列到指定长度"""
        if len(sequence) >= max_len:
            return sequence[:max_len]
        return sequence + [pad_value] * (max_len - len(sequence))


class DynamicDataCollator:
    """
    动态数据整理器 - 支持多种NLP任务的灵活padding
    
    支持的任务:
    - 文本分类
    - 掩码语言模型(MLM)
    - 因果语言模型(CLM)
    - 问答任务
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        
        self.special_tokens = set(tokenizer.all_special_ids)
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self._prepare_batch(examples)
        return self._pad_batch(batch)
    
    def _prepare_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, List]:
        """准备批次数据"""
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for example in examples:
            input_ids = example['input_ids']
            attention_mask = example.get('attention_mask', [1] * len(input_ids))
            
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(input_ids.copy())
        
        return batch
    
    def _pad_batch(self, batch: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """填充批次"""
        max_len = min(
            max(len(ids) for ids in batch['input_ids']),
            self.max_length
        )
        
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        
        padded_batch = {}
        
        for key in batch:
            if key == 'labels':
                padded = [self._pad_to_length(ids, max_len, -100) for ids in batch[key]]
            else:
                padded = [self._pad_to_length(ids, max_len, self.tokenizer.pad_token_id) 
                         for ids in batch[key]]
            padded_batch[key] = torch.tensor(padded, dtype=torch.long)
        
        return padded_batch
    
    def _pad_to_length(self, sequence: List, target_len: int, pad_value: int) -> List:
        """将序列填充到目标长度"""
        if len(sequence) >= target_len:
            return sequence[:target_len]
        return sequence + [pad_value] * (target_len - len(sequence))
    
    def prepare_mlm_labels(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为掩码语言模型准备标签
        
        策略:
        - 随机选择15%的token进行掩码
        - 80%替换为[MASK]，10%替换为随机token，10%保持不变
        """
        input_ids = batch['input_ids'].clone()
        labels = batch['input_ids'].clone()
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        special_tokens_mask = torch.tensor([
            [1 if token in self.special_tokens else 0 for token in ids]
            for ids in input_ids
        ])
        probability_matrix.masked_fill_(special_tokens_mask.bool(), 0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        for i in range(input_ids.shape[0]):
            indices = masked_indices[i].nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                for idx in indices:
                    rand = random.random()
                    if rand < 0.8:
                        input_ids[i][idx] = self.tokenizer.mask_token_id
                    elif rand < 0.9:
                        input_ids[i][idx] = random.randint(0, self.tokenizer.vocab_size - 1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': batch['attention_mask'],
            'labels': labels
        }


class MemoryEfficientPreprocessor:
    """
    内存高效预处理器 - 处理大规模数据集
    
    优化策略:
    - 分块处理：避免一次性加载整个数据集
    - 并行预处理：利用多核CPU
    - 内存映射：处理超大文件
    - 渐进式编码：流式处理tokenization
    """
    
    def __init__(
        self,
        tokenizer: Any,
        chunk_size: int = 10000,
        num_processes: int = None
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.num_processes = num_processes or os.cpu_count()
    
    def preprocess_large_file(
        self,
        input_path: str,
        output_path: str,
        max_length: int = 512
    ):
        """
        流式预处理大文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            max_length: 最大序列长度
        """
        logger.info(f"开始预处理文件: {input_path}")
        start_time = time.time()
        
        results = []
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            buffer = []
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue
                
                buffer.append(line)
                
                if len(buffer) >= self.chunk_size:
                    chunk_results = self._process_chunk(buffer, max_length)
                    results.extend(chunk_results)
                    buffer = []
                    
                    if line_idx % 100000 == 0:
                        logger.info(f"已处理 {line_idx} 行")
        
        if buffer:
            results.extend(self._process_chunk(buffer, max_length))
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for item in results:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        elapsed = time.time() - start_time
        logger.info(f"预处理完成，耗时: {elapsed:.2f}秒，输出 {len(results)} 个样本")
        
        return results
    
    def _process_chunk(self, chunk: List[str], max_length: int) -> List[Dict]:
        """处理一个数据块"""
        if self.num_processes > 1:
            return self._process_chunk_parallel(chunk, max_length)
        return self._process_chunk_sequential(chunk, max_length)
    
    def _process_chunk_sequential(self, chunk: List[str], max_length: int) -> List[Dict]:
        """顺序处理数据块"""
        results = []
        for text in chunk:
            tokenized = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            results.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized.get('attention_mask', [1] * len(tokenized['input_ids']))
            })
        return results
    
    def _process_chunk_parallel(self, chunk: List[str], max_length: int) -> List[Dict]:
        """并行处理数据块"""
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [
                executor.submit(self._tokenize_single, text, max_length)
                for text in chunk
            ]
            results = [f.result() for f in futures]
        return results
    
    def _tokenize_single(self, text: str, max_length: int) -> Dict:
        """单条文本tokenization"""
        tokenized = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized.get('attention_mask', [1] * len(tokenized['input_ids']))
        }


class DistributedDataLoader:
    """
    分布式数据加载器 - 支持多GPU训练
    
    功能:
    - 自动数据分片
    - 分布式采样
    - 同步epoch计数
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        self.sampler = None
        self.dataloader = None
    
    def create_distributed_sampler(self, rank: int, world_size: int):
        """创建分布式采样器"""
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )
    
    def get_dataloader(self, rank: int = 0, world_size: int = 1) -> DataLoader:
        """获取数据加载器"""
        if world_size > 1 and self.sampler is None:
            self.create_distributed_sampler(rank, world_size)
        
        sampler = self.sampler if world_size > 1 else None
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0
        )
        
        return self.dataloader
    
    def set_epoch(self, epoch: int):
        """设置当前epoch（用于分布式训练）"""
        if self.sampler:
            self.sampler.set_epoch(epoch)


class SamplerStrategies:
    """
    采样策略类 - 实现多种数据采样方法
    
    策略:
    - 随机采样
    - 顺序采样
    - 类别平衡采样
    - 加权随机采样
    - 过采样/欠采样
    """
    
    @staticmethod
    def create_weighted_sampler(labels: List[int]) -> torch.utils.data.WeightedRandomSampler:
        """
        创建加权随机采样器
        
        用于处理类别不平衡问题
        
        Args:
            labels: 样本标签列表
            
        Returns:
            加权随机采样器
        """
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        
        class_weights = [1.0 / class_counts[label] for label in labels]
        weights = torch.tensor(class_weights, dtype=torch.double)
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(labels),
            replacement=True
        )
        
        return sampler
    
    @staticmethod
    def create_balanced_batch_sampler(dataset: Dataset, batch_size: int, n_classes: int):
        """
        创建类别平衡批次采样器
        
        每个批次包含来自每个类别的相等数量的样本
        """
        class_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            label = item.get('labels', item.get('label', 0))
            class_to_indices[label].append(idx)
        
        class_iters = {
            cls: iter(indices) 
            for cls, indices in class_to_indices.items()
        }
        
        class_cycle = {cls: 0 for cls in class_to_indices}
        
        class BalancedBatchSampler:
            def __init__(self, class_to_indices, batch_size, n_classes):
                self.class_to_indices = class_to_indices
                self.batch_size = batch_size
                self.n_classes = n_classes
                self.samples_per_class = batch_size // n_classes
                
            def __iter__(self):
                for _ in range(len(self)):
                    batch = []
                    for cls in self.class_to_indices:
                        try:
                            for _ in range(self.samples_per_class):
                                batch.append(next(self.class_to_indices[cls]))
                        except StopIteration:
                            self.class_to_indices[cls] = iter(
                                self.class_to_indices[cls]
                            )
                            batch.append(next(self.class_to_indices[cls]))
                    random.shuffle(batch)
                    yield batch
            
            def __len__(self):
                return sum(len(indices) for indices in self.class_to_indices.values()) // self.batch_size
        
        return BalancedBatchSampler(class_to_indices, batch_size, n_classes)


class DataPipelineDemo:
    """
    数据管道演示类
    """
    
    @staticmethod
    def demo_basic_dataset():
        """演示基本Dataset使用"""
        logger.info("演示基本Dataset创建")
        
        class SimpleDataset(Dataset):
            def __init__(self, size: int = 100):
                self.data = list(range(size))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {
                    'input': torch.tensor(self.data[idx]),
                    'label': torch.tensor(self.data[idx] % 10)
                }
        
        dataset = SimpleDataset(100)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )
        
        for batch in dataloader:
            logger.info(f"批次形状: input={batch['input'].shape}, label={batch['label'].shape}")
            break
        
        return dataset, dataloader
    
    @staticmethod
    def demo_collate_function():
        """演示自定义collate函数"""
        logger.info("演示自定义collate函数")
        
        def custom_collate(batch):
            inputs = [item['input'] for item in batch]
            labels = [item['label'] for item in batch]
            
            max_len = max(len(inp) for inp in inputs)
            
            padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
            for i, inp in enumerate(inputs):
                padded_inputs[i, :len(inp)] = inp
            
            return {
                'input_ids': padded_inputs,
                'labels': torch.tensor(labels)
            }
        
        class VariableLengthDataset(Dataset):
            def __init__(self, size: int = 50):
                self.lengths = [random.randint(5, 20) for _ in range(size)]
            
            def __len__(self):
                return len(self.lengths)
            
            def __getitem__(self, idx):
                return {
                    'input': torch.randint(0, 100, (self.lengths[idx],)),
                    'label': torch.tensor(self.lengths[idx] % 5)
                }
        
        dataset = VariableLengthDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=custom_collate
        )
        
        for batch in dataloader:
            logger.info(f"批次形状: {batch['input_ids'].shape}")
            break
        
        return dataset, dataloader
    
    @staticmethod
    def demo_memory_efficiency():
        """演示内存优化技巧"""
        logger.info("演示内存优化技巧")
        
        class MemoryEfficientDataset(Dataset):
            def __init__(self, file_path: str, cache_size: int = 1000):
                self.file_path = file_path
                self.cache_size = cache_size
                self.cache = {}
                self.cache_order = []
                self._load_metadata()
            
            def _load_metadata(self):
                with open(self.file_path, 'r') as f:
                    self.total_lines = sum(1 for _ in f)
            
            def __len__(self):
                return self.total_lines
            
            def __getitem__(self, idx):
                if idx in self.cache:
                    return self.cache[idx]
                
                with open(self.file_path, 'r') as f:
                    for _ in range(idx):
                        f.readline()
                    line = f.readline().strip()
                
                result = {'text': line, 'processed': hash(line)}
                
                self.cache[idx] = result
                self.cache_order.append(idx)
                
                if len(self.cache_order) > self.cache_size:
                    old_idx = self.cache_order.pop(0)
                    del self.cache[old_idx]
                
                return result
        
        return MemoryEfficientDataset


class HuggingFaceDatasetsIntegration:
    """
    Hugging Face datasets库集成
    
    提供与datasets库的便捷集成
    """
    
    @staticmethod
    def load_from_disk(dataset_path: str) -> Any:
        """从磁盘加载数据集"""
        from datasets import load_from_disk
        return load_from_disk(dataset_path)
    
    @staticmethod
    def load_csv_dataset(
        csv_path: str,
        text_column: str = 'text',
        **kwargs
    ) -> Any:
        """加载CSV数据集"""
        from datasets import load_dataset
        return load_dataset('csv', data_files=csv_path, **kwargs)
    
    @staticmethod
    def tokenize_dataset(
        dataset: Any,
        tokenizer: Any,
        batched: bool = True,
        remove_columns: List[str] = None,
        **tokenizer_kwargs
    ) -> Any:
        """对数据集进行tokenization"""
        return dataset.map(
            lambda x: tokenizer(x[text_column], **tokenizer_kwargs),
            batched=batched,
            remove_columns=remove_columns or [text_column]
        )


def demo_data_pipeline():
    """数据管道演示主函数"""
    print("=" * 60)
    print("高效数据管道教程演示")
    print("=" * 60)
    
    print("\n1. 基本Dataset演示")
    DataPipelineDemo.demo_basic_dataset()
    
    print("\n2. 自定义Collate函数演示")
    DataPipelineDemo.demo_collate_function()
    
    print("\n3. 内存优化演示")
    DataPipelineDemo.demo_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("数据管道教程演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_data_pipeline()
