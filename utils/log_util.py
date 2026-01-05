"""
日志工具类

提供统一的日志配置，最小依赖原则。
"""

import logging
from typing import Optional


class LogUtil:
    """日志工具类"""
    
    _configured = False
    
    @staticmethod
    def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """
        获取日志记录器
        
        Args:
            name: 日志记录器名称（通常是模块名）
            level: 日志级别，默认INFO
            
        Returns:
            日志记录器实例
        """
        if not LogUtil._configured:
            logging.basicConfig(
                level=level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            LogUtil._configured = True
        
        return logging.getLogger(name)
    
    @staticmethod
    def configure(level: int = logging.INFO, 
                  format_str: Optional[str] = None) -> None:
        """
        配置日志系统
        
        Args:
            level: 日志级别，默认INFO
            format_str: 日志格式字符串，None使用默认格式
        """
        if format_str is None:
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(level=level, format=format_str)
        LogUtil._configured = True

