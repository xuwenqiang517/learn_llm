"""
JSON工具类

提供JSON文件的读取和保存功能，最小依赖原则。
"""

import json
from pathlib import Path
from typing import Optional, Any


class JsonUtil:
    """JSON操作工具类"""
    
    @staticmethod
    def load(file_path: Path) -> Optional[Any]:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            解析后的数据，失败返回None
        """
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return None
    
    @staticmethod
    def save(data: Any, file_path: Path, indent: int = 2) -> bool:
        """
        保存数据为JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
            indent: JSON缩进，默认2
            
        Returns:
            是否保存成功
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            return True
        except Exception:
            return False
    
    @staticmethod
    def dumps(data: Any, indent: Optional[int] = 2) -> str:
        """
        将数据转换为JSON字符串
        
        Args:
            data: 要转换的数据
            indent: JSON缩进，None表示压缩格式
            
        Returns:
            JSON字符串
        """
        try:
            return json.dumps(data, ensure_ascii=False, indent=indent)
        except Exception:
            return "{}"
    
    @staticmethod
    def loads(json_str: str) -> Optional[Any]:
        """
        解析JSON字符串
        
        Args:
            json_str: JSON字符串
            
        Returns:
            解析后的数据，失败返回None
        """
        try:
            return json.loads(json_str)
        except Exception:
            return None

