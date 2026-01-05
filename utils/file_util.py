"""
文件工具类

提供文件操作相关功能，最小依赖原则。
"""

from pathlib import Path
from datetime import datetime
from typing import Optional


class FileUtil:
    """文件操作工具类"""
    
    @staticmethod
    def ensure_dir(dir_path: Path) -> None:
        """
        确保目录存在
        
        Args:
            dir_path: 目录路径
        """
        dir_path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def ensure_dirs(*dir_paths: Path) -> None:
        """
        确保多个目录存在
        
        Args:
            *dir_paths: 多个目录路径
        """
        for dir_path in dir_paths:
            FileUtil.ensure_dir(dir_path)
    
    @staticmethod
    def is_cache_valid(file_path: Path, expiry_days: int = 1) -> bool:
        """
        检查缓存文件是否有效
        
        Args:
            file_path: 缓存文件路径
            expiry_days: 有效期（天）
            
        Returns:
            缓存是否有效
        """
        if not file_path.exists():
            return False
        
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age = (datetime.now() - mtime).days
            return age < expiry_days
        except Exception:
            return False
    
    @staticmethod
    def read_text(file_path: Path, encoding: str = 'utf-8') -> Optional[str]:
        """
        读取文本文件
        
        Args:
            file_path: 文件路径
            encoding: 编码格式，默认utf-8
            
        Returns:
            文件内容，失败返回None
        """
        try:
            if file_path.exists():
                return file_path.read_text(encoding=encoding)
        except Exception:
            pass
        return None
    
    @staticmethod
    def write_text(content: str, file_path: Path, encoding: str = 'utf-8') -> bool:
        """
        写入文本文件
        
        Args:
            content: 文件内容
            file_path: 文件路径
            encoding: 编码格式，默认utf-8
            
        Returns:
            是否写入成功
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding=encoding)
            return True
        except Exception:
            return False

