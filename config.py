"""
项目配置文件 - learn_llm

包含所有模块共享的配置常量、环境变量加载和全局设置。

Author: learn_llm
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """项目基础配置"""
    project_name: str = "learn_llm"
    project_version: str = "1.0.0"
    working_dir: Path = Path(__file__).parent.parent
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")


@dataclass
class ModelConfig:
    """模型相关配置"""
    default_model: str = "Qwen/Qwen-7B-Chat"
    model_cache_dir: str = "~/.cache/huggingface/hub"
    max_sequence_length: int = 2048
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1


@dataclass
class TrainingConfig:
    """训练通用配置"""
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_weight_decay: float = 0.01
    max_steps: int = 10000
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 10


@dataclass
class QuantizationConfig:
    """量化配置"""
    bits: int = 4
    group_size: int = 128
    calibration_samples: int = 100
    quantize_method: str = "awq"


@dataclass
class DistributedConfig:
    """分布式训练配置"""
    world_size: int = 1
    local_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    backend: str = "nccl"
    seed: int = 42


@dataclass
class APIConfig:
    """API服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size: int = 10 * 1024 * 1024
    max_batch_size: int = 32
    rate_limit: int = 100
    rate_limit_window: int = 60
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_size_mb: int = 100
    backup_count: int = 5


class ConfigLoader:
    """配置加载器"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_env()
        return cls._instance
    
    def _load_env(self):
        """加载环境变量"""
        env_mappings = {
            "HF_TOKEN": ("huggingface", "token"),
            "CUDA_VISIBLE_DEVICES": ("gpu", "devices"),
            "MASTER_ADDR": ("distributed", "master_addr"),
            "MASTER_PORT": ("distributed", "master_port"),
            "WORLD_SIZE": ("distributed", "world_size"),
            "RANK": ("distributed", "rank"),
            "LOCAL_RANK": ("distributed", "local_rank"),
        }
        
        for env_key, (section, key) in env_mappings.items():
            value = os.environ.get(env_key)
            if value:
                if section not in self._config:
                    self._config[section] = {}
                self._config[section][key] = value
                logger.debug(f"Loaded {env_key} from environment")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self._config.get(section, {}).get(key, default)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.project = ProjectConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.quantization = QuantizationConfig()
        self.distributed = DistributedConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self._config_loader = ConfigLoader()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                if isinstance(values, dict):
                    for key, value in values.items():
                        if hasattr(getattr(self, section), key):
                            setattr(getattr(self, section), key, value)
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        import json
        
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if hasattr(attr, '__dict__'):
                    config_dict[attr_name] = {
                        k: v for k, v in attr.__dict__.items()
                        if not k.startswith('_')
                    }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Config saved to {output_path}")
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print("项目配置摘要")
        print("=" * 60)
        
        print(f"\n项目名称: {self.project.project_name}")
        print(f"默认模型: {self.model.default_model}")
        print(f"最大序列长度: {self.model.max_sequence_length}")
        print(f"学习率: {self.training.learning_rate}")
        print(f"批次大小: {self.training.batch_size}")
        print(f"量化位数: {self.quantization.bits}")
        print(f"API端口: {self.api.port}")
        print(f"分布式后端: {self.distributed.backend}")
        
        print("\n" + "=" * 60)


config_manager = ConfigManager()


def setup_logging(logging_config: LoggingConfig = None) -> logging.Logger:
    """设置日志配置"""
    config = logging_config or config_manager.logging
    
    logging.basicConfig(
        level=getattr(logging, config.level),
        format=config.format
    )
    
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(logging.Formatter(config.format))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def get_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_directories():
    """创建必要的目录"""
    dirs = [
        config_manager.project.log_dir,
        config_manager.project.cache_dir,
        config_manager.project.output_dir,
        config_manager.project.data_dir
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {dir_path}")


def get_default_device() -> str:
    """获取默认设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


import torch


if __name__ == "__main__":
    setup_directories()
    config_manager.print_summary()
