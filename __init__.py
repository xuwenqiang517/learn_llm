"""
learn_llm - 大模型工程师技能学习项目

这是一个系统化的大模型学习项目,涵盖从理论基础到工程实践的完整技能体系。

目录结构:
- week1_theory_foundations/: 理论基础(数学原理、注意力机制)
- week2_engineering_practice/: 工程实践(PyTorch、数据管道、HuggingFace)
- week3_finetuning_practice/: 微调实践(全量微调、LoRA/PEFT)
- week4_deployment_optimization/: 部署优化(量化、蒸馏、API、分布式)

Author: learn_llm
"""

__version__ = "1.0.0"
__author__ = "learn_llm"

from .config import (
    ProjectConfig,
    ModelConfig,
    TrainingConfig,
    QuantizationConfig,
    DistributedConfig,
    APIConfig,
    config_manager
)

from .utils import (
    Timer,
    ProgressBar,
    set_seed,
    timer_decorator,
    retry,
    batch_process,
    chunk_list,
    unique_list,
    download_file,
    clear_memory,
    get_memory_usage,
    estimate_model_size
)
