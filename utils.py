"""
工具函数库 - learn_llm

提供项目中通用的工具函数,包括:
1. 字符串处理
2. 文件操作
3. 时间处理
4. 日志工具
5. 性能计时器
6. 进度显示
7. 随机数生成
8. 类型转换

Author: learn_llm
"""

import os
import sys
import time
import json
import random
import hashlib
import logging
import inspect
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable, TypeVar, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps, lru_cache
import tempfile
import shutil
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

T = TypeVar('T')


class Timer:
    """高精度计时器"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0
        self.running = False
    
    def start(self) -> "Timer":
        self.start_time = time.perf_counter()
        self.running = True
        return self
    
    def stop(self) -> "Timer":
        if self.running:
            self.end_time = time.perf_counter()
            self.elapsed = self.end_time - self.start_time
            self.running = False
        return self
    
    def reset(self) -> "Timer":
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed = 0.0
        self.running = False
        return self
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def get_elapsed_seconds(self) -> float:
        if self.running:
            return time.perf_counter() - self.start_time
        return self.elapsed
    
    def get_elapsed_ms(self) -> float:
        return self.get_elapsed_seconds() * 1000
    
    def get_elapsed_formatted(self) -> str:
        elapsed = self.get_elapsed_seconds()
        if elapsed < 0.001:
            return f"{elapsed * 1000000:.2f}μs"
        elif elapsed < 1:
            return f"{elapsed * 1000:.2f}ms"
        elif elapsed < 60:
            return f"{elapsed:.2f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.2f}s"


class ProgressBar:
    """进度条显示"""
    
    def __init__(
        self,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50,
        fill: str = "█",
        unit: str = "it"
    ):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.unit = unit
        self.iteration = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1, info: str = ""):
        self.iteration += n
        percent = self.iteration / self.total
        filled_length = int(self.length * percent)
        bar = self.fill * filled_length + "-" * (self.length - filled_length)
        
        elapsed = time.time() - self.start_time
        eta = (elapsed / self.iteration) * (self.total - self.iteration) if self.iteration > 0 else 0
        
        print(
            f"\r{self.prefix} |{bar}| {percent:.1%} "
            f"[{self.iteration}/{self.total} {self.unit}] "
            f"ETA: {eta:.1f}s {info}",
            end="",
            flush=True
        )
        
        if self.iteration >= self.total:
            print()
    
    def close(self):
        if self.iteration < self.total:
            print()


def set_seed(seed: int, deterministic: bool = False):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    import numpy as np
    np.random.seed(seed)
    
    logger.info(f"Random seed set to {seed}")


@contextmanager
def temp_directory():
    """临时目录上下文管理器"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小(字节)"""
    path = Path(file_path)
    if path.exists():
        return path.stat().st_size
    return 0


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def count_lines(file_path: Union[str, Path]) -> int:
    """统计文件行数"""
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def read_text_file(file_path: Union[str, Path]) -> str:
    """读取文本文件"""
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(file_path: Union[str, Path], content: str):
    """写入文本文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def read_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """读取JSON文件"""
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(file_path: Union[str, Path], data: Any, indent: int = 2):
    """写入JSON文件"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def ensure_directory(path: Union[str, Path]):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def list_files(
    directory: Union[str, Path],
    extension: Optional[str] = None,
    recursive: bool = False
) -> List[Path]:
    """列出目录中的文件"""
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        return []
    
    if extension:
        pattern = "**/*" + extension if recursive else "*" + extension
    else:
        pattern = "**/*" if recursive else "*"
    
    return list(path.glob(pattern))


def get_subdirectories(directory: Union[str, Path]) -> List[Path]:
    """获取子目录列表"""
    path = Path(directory)
    if path.exists() and path.is_dir():
        return [d for d in path.iterdir() if d.is_dir()]
    return []


@lru_cache(maxsize=128)
def cached_hash(text: str) -> str:
    """计算文本哈希值(带缓存)"""
    return hashlib.md5(text.encode()).hexdigest()


def batch_process(
    items: List[T],
    batch_size: int,
    process_func: Callable[[T], Any],
    parallel: bool = False,
    max_workers: int = 4
) -> List[Any]:
    """批量处理数据"""
    results = []
    
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, item) for item in items]
            for future in futures:
                results.append(future.result())
    else:
        for item in items:
            results.append(process_func(item))
    
    return results


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """展平嵌套列表"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """将列表分块"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def unique_list(items: List[T]) -> List[T]:
    """去重保持顺序"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """清理空白字符"""
    return " ".join(text.split())


def format_timestamp(timestamp: Optional[float] = None, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间戳"""
    dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
    return dt.strftime(format)


def parse_timestamp(timestamp_str: str, format: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """解析时间戳字符串"""
    return datetime.strptime(timestamp_str, format)


def get_time_diff(start_time: datetime, end_time: Optional[datetime] = None) -> timedelta:
    """计算时间差"""
    end = end_time or datetime.now()
    return end - start_time


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Exception] = (Exception,)
):
    """重试装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def timer_decorator(func: Callable) -> Callable:
    """函数计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(func.__name__)
        timer.start()
        result = func(*args, **kwargs)
        timer.stop()
        logger.info(f"{func.__name__} executed in {timer.get_elapsed_formatted()}")
        return result
    return wrapper


def log_arguments(func: Callable) -> Callable:
    """记录函数参数的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        params = sig.bind(*args, **kwargs).arguments
        logger.debug(f"Calling {func.__name__} with params: {params}")
        return func(*args, **kwargs)
    return wrapper


class ThreadSafeCounter:
    """线程安全计数器"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        with self._lock:
            self._value -= 1
            return self._value
    
    def get_value(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self, value: int = 0):
        with self._lock:
            self._value = value


def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage() -> Dict[str, float]:
    """获取内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    result = {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }
    
    if torch.cuda.is_available():
        result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return result


def download_file(url: str, dest_path: Union[str, Path], progress: bool = True):
    """下载文件"""
    import requests
    from tqdm import tqdm
    
    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, 'wb') as f:
        if progress:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def extract_archive(archive_path: Union[str, Path], dest_dir: Union[str, Path]):
    """解压压缩包"""
    import tarfile
    import zipfile
    
    path = Path(archive_path)
    dest = Path(dest_dir)
    
    if path.suffix == '.tar.gz' or path.suffix == '.tgz':
        with tarfile.open(path, 'r:gz') as tar:
            tar.extractall(dest)
    elif path.suffix == '.tar':
        with tarfile.open(path, 'r') as tar:
            tar.extractall(dest)
    elif path.suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {path.suffix}")


class Singleton(type):
    """单例模式元类"""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def deep_copy(obj: Any) -> Any:
    """深拷贝对象"""
    import copy
    return copy.deepcopy(obj)


def estimate_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """估算模型参数量和大小"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = size_bytes / 1024 / 1024
    return num_params, size_mb


def calculate_flops(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda"
) -> int:
    """估算模型FLOPs"""
    from torch.profiler import profile, record_function
    
    model = model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    
    with profile(activities=[torch.profiler.ProfilerActivity.CUDA], with_flops=True) as prof:
        with record_function("model_forward"):
            model(dummy_input)
    
    return sum(e.flops for e in prof.key_averages())


import torch
import torch.nn as nn


if __name__ == "__main__":
    print("=" * 60)
    print("工具函数演示")
    print("=" * 60)
    
    with Timer("测试计时器") as timer:
        time.sleep(0.5)
    print(f"计时结果: {timer.get_elapsed_formatted()}")
    
    set_seed(42)
    print(f"随机种子已设置")
    
    print(f"\n文件大小格式化: {format_file_size(1024 * 1024 * 5)}")
    
    sample_list = [1, 2, 2, 3, 3, 3, 4]
    print(f"列表去重: {unique_list(sample_list)}")
    
    print(f"\n列表分块: {chunk_list([1,2,3,4,5,6,7], 3)}")
    
    memory = get_memory_usage()
    print(f"内存使用: {memory}")
