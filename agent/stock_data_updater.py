"""
股票数据更新工具

独立的MCP/Tool，负责更新股票数据缓存。
返回字符串消息：数据更新完成、数据更新失败、数据都已缓存
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Tuple

import akshare as ak
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_util import JsonUtil
from utils.file_util import FileUtil
from utils.log_util import LogUtil

logger = LogUtil.get_logger(__name__)

# ==================== 目录结构定义 ====================
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
DAILY_DATA_DIR = DATA_DIR / "daily"

# 确保目录存在
FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, DAILY_DATA_DIR)

# 基础数据文件路径
STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.json"
INDUSTRY_MAP_FILE = BASE_DATA_DIR / "industry_map.json"
CONCEPT_MAP_FILE = BASE_DATA_DIR / "concept_map.json"

# 缓存有效期（天）
CACHE_EXPIRY_DAYS = 1


def _is_cache_valid(cache_file: Path, force_update: bool = False) -> bool:
    """
    检查缓存是否有效
    
    Args:
        cache_file: 缓存文件路径
        force_update: 是否强制更新
        
    Returns:
        缓存是否有效
    """
    if force_update:
        return False
    
    if not FileUtil.is_cache_valid(cache_file, CACHE_EXPIRY_DAYS):
        return False
    
    cached = JsonUtil.load(cache_file)
    return cached is not None


# ==================== 基础数据更新 ====================

def _update_stock_list(force_update: bool = False) -> bool:
    """
    更新股票列表
    
    Args:
        force_update: 是否强制更新
        
    Returns:
        是否更新成功
    """
    if _is_cache_valid(STOCK_LIST_FILE, force_update):
        return True
    
    try:
        stock_list = ak.stock_info_a_code_name()
        if not stock_list.empty:
            JsonUtil.save(stock_list.to_dict('records'), STOCK_LIST_FILE)
            logger.info(f"股票列表已更新（{len(stock_list)}只）")
            return True
    except Exception as e:
        logger.error(f"更新股票列表失败: {e}")
    
    return False


def _update_industry_map(force_update: bool = False) -> bool:
    """
    更新行业板块映射
    
    Args:
        force_update: 是否强制更新
        
    Returns:
        是否更新成功
    """
    if _is_cache_valid(INDUSTRY_MAP_FILE, force_update):
        return True
    
    try:
        stock_board = ak.stock_board_industry_name_em()
        if stock_board.empty:
            return False
        
        industry_map = {}
        industries = stock_board['板块名称'].tolist()
        
        for industry in tqdm(industries, desc="更新行业映射", unit="行业"):
            try:
                cons = ak.stock_board_industry_cons_em(symbol=industry)
                if not cons.empty and '代码' in cons.columns:
                    for _, row in cons.iterrows():
                        code = str(row['代码']).zfill(6)
                        if code not in industry_map:
                            industry_map[code] = industry
            except Exception:
                continue
        
        if industry_map:
            JsonUtil.save(industry_map, INDUSTRY_MAP_FILE)
            logger.info(f"行业映射已更新（{len(industry_map)}只股票）")
            return True
    except Exception as e:
        logger.error(f"更新行业映射失败: {e}")
    
    return False


def _update_concept_map(force_update: bool = False) -> bool:
    """
    更新概念板块映射
    
    Args:
        force_update: 是否强制更新
        
    Returns:
        是否更新成功
    """
    if _is_cache_valid(CONCEPT_MAP_FILE, force_update):
        return True
    
    try:
        concept_df = ak.stock_board_concept_name_em()
        if concept_df is None or concept_df.empty or '板块名称' not in concept_df.columns:
            return False
        
        concept_map = {}
        concepts = concept_df['板块名称'].tolist()
        
        for concept in tqdm(concepts, desc="更新概念映射", unit="概念"):
            try:
                cons = ak.stock_board_concept_cons_em(symbol=concept)
                if cons is not None and not cons.empty and '代码' in cons.columns:
                    for _, row in cons.iterrows():
                        code = str(row['代码']).zfill(6)
                        if code not in concept_map:
                            concept_map[code] = []
                        if concept not in concept_map[code]:
                            concept_map[code].append(concept)
            except Exception:
                continue
        
        if concept_map:
            JsonUtil.save(concept_map, CONCEPT_MAP_FILE)
            logger.info(f"概念映射已更新（{len(concept_map)}只股票）")
            return True
    except Exception as e:
        logger.error(f"更新概念映射失败: {e}")
    
    return False


# ==================== 交易日数据获取 ====================

def _should_include_today() -> bool:
    """
    判断是否应该包含今天的数据
    
    Returns:
        如果当前时间在16:00之后，返回True；否则返回False
    """
    return datetime.now().hour >= 16


def _get_trading_days(days: int = 15) -> list:
    """
    获取交易日列表
    
    Args:
        days: 向前查找的天数
        
    Returns:
        交易日列表（格式：YYYYMMDD）
    """
    now = datetime.now()
    if not _should_include_today():
        now = now - timedelta(days=1)
    end_date = now
    start_date = end_date - timedelta(days=days)
    
    try:
        df = ak.tool_trade_date_hist_sina()
        if df is not None and 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            trading_days = df.loc[mask, 'trade_date'].dt.strftime('%Y%m%d').tolist()
            if trading_days:
                return trading_days
    except Exception:
        pass
    
    # 降级方案：简单计算（排除周末）
    trading_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            trading_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return trading_days


def _download_stock_daily(symbol: str, date: str) -> Optional[dict]:
    """
    下载并缓存单只股票单日数据
    
    Args:
        symbol: 股票代码
        date: 日期（YYYYMMDD）
        
    Returns:
        股票数据字典，失败返回None
    """
    cache_file = DAILY_DATA_DIR / date / f"{symbol}.json"
    
    # 如果缓存已存在，直接返回
    cached = JsonUtil.load(cache_file)
    if cached is not None:
        return cached
    
    # 创建目录
    FileUtil.ensure_dir(cache_file.parent)
    
    try:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=date, end_date=date, adjust="qfq")
        
        if not df.empty and len(df) > 0:
            row = df.iloc[0]
            data = {
                'symbol': symbol,
                'date': date,
                'open': float(row.get('开盘', 0)),
                'close': float(row.get('收盘', 0)),
                'high': float(row.get('最高', 0)),
                'low': float(row.get('最低', 0)),
                'volume': int(row.get('成交量', 0)),
                'amount': float(row.get('成交额', 0)),
                'amplitude': float(row.get('振幅', 0)),
                'change_pct': float(row.get('涨跌幅', 0)),
                'change': float(row.get('涨跌额', 0)),
                'turnover': float(row.get('换手率', 0)),
                'pre_close': float(row.get('前收盘', row.get('收盘', 0) * 0.99))
            }
            
            JsonUtil.save(data, cache_file)
            return data
    except Exception:
        pass
    
    return None


def _get_stock_codes_for_market(market: str) -> list:
    """
    获取指定市场的股票代码列表
    
    Args:
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        
    Returns:
        股票代码列表
    """
    stock_list_data = JsonUtil.load(STOCK_LIST_FILE)
    if not stock_list_data:
        return []
    
    stock_list = pd.DataFrame(stock_list_data)
    if stock_list.empty:
        return []
    
    if market == "sh":
        stock_list = stock_list[stock_list['code'].str.startswith('6')]
    elif market == "sz":
        stock_list = stock_list[stock_list['code'].str.startswith(('0', '3'))]
    
    return stock_list['code'].tolist()


def _collect_cache_tasks(stock_codes: list, trading_days: list, force_update: bool) -> list:
    """
    收集需要下载的缓存任务
    
    Args:
        stock_codes: 股票代码列表
        trading_days: 交易日列表
        force_update: 是否强制更新
        
    Returns:
        缓存任务列表 [(code, date), ...]
    """
    cache_tasks = []
    for code in stock_codes:
        for date in trading_days:
            cache_file = DAILY_DATA_DIR / date / f"{code}.json"
            if force_update or not cache_file.exists():
                cache_tasks.append((code, date))
    return cache_tasks


def _download_with_thread_pool(cache_tasks: list) -> int:
    """
    使用线程池并发下载数据
    
    Args:
        cache_tasks: 缓存任务列表 [(code, date), ...]
        
    Returns:
        成功更新的数据条数
    """
    updated_count = 0
    max_workers = min(32, os.cpu_count() or 8)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_stock_daily, code, date): (code, date)
                for code, date in cache_tasks
            }
            
            with tqdm(total=len(futures), desc="更新日线数据", unit="条") as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    code, date = futures[future]
                    try:
                        result = future.result()
                        if result:
                            updated_count += 1
                    except Exception:
                        pass
                    pbar.set_postfix_str(f"{code} {date}")
        
        return updated_count
    except Exception as e:
        logger.error(f"并发下载失败: {e}")
        return updated_count


def _update_daily_data(days: int = 15, market: str = "all", 
                      force_update: bool = False) -> Tuple[bool, int]:
    """
    更新日线数据缓存
    
    Args:
        days: 需要缓存的天数
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        force_update: 是否强制更新所有数据
        
    Returns:
        (是否成功, 更新的数据条数)
    """
    trading_days = _get_trading_days(days)
    stock_codes = _get_stock_codes_for_market(market)
    
    if not stock_codes:
        return False, 0
    
    cache_tasks = _collect_cache_tasks(stock_codes, trading_days, force_update)
    
    if not cache_tasks:
        return True, 0
    
    updated_count = _download_with_thread_pool(cache_tasks)
    
    return True, updated_count


# ==================== 主入口函数 ====================

def _check_cache_completeness(days: int, market: str) -> bool:
    """
    检查缓存是否完整（基础数据+日线数据）
    
    Args:
        days: 需要缓存的天数
        market: 市场类型
        
    Returns:
        缓存是否完整
    """
    # 检查基础数据
    if not _is_cache_valid(STOCK_LIST_FILE):
        return False
    if not _is_cache_valid(INDUSTRY_MAP_FILE):
        return False
    if not _is_cache_valid(CONCEPT_MAP_FILE):
        return False
    
    # 检查日线数据
    trading_days = _get_trading_days(days)
    stock_codes = _get_stock_codes_for_market(market)
    
    if not stock_codes:
        return False
    
    for code in stock_codes:
        for date in trading_days:
            cache_file = DAILY_DATA_DIR / date / f"{code}.json"
            if not cache_file.exists():
                return False
    
    return True


def update_stock_data(days: int = 15, market: str = "all", 
                     force_update: bool = False) -> bool:
    """
    更新股票数据（主入口函数）
    
    返回布尔值：
    - True：基础数据和日线数据更新成功，或已有完整缓存可用
    - False：更新过程中出现错误，数据状态不完整
    
    Args:
        days: 需要缓存的天数，默认15天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        force_update: 是否强制更新所有数据
        
    Returns:
        是否更新成功
    """
    try:
        # 0. 检查缓存完整性（非强制更新时）
        if not force_update and _check_cache_completeness(days, market):
            logger.info("缓存完整，无需更新")
            return True
        
        # 1. 更新基础数据
        logger.info("开始更新股票数据...")
        
        stock_list_ok = _update_stock_list(force_update)
        if not stock_list_ok:
            logger.error("股票列表更新失败")
            return False
        
        industry_ok = _update_industry_map(force_update)
        concept_ok = _update_concept_map(force_update)
        if not industry_ok:
            logger.error("行业板块映射更新失败")
            return False
        if not concept_ok:
            logger.error("概念板块映射更新失败")
            return False
        
        # 2. 更新日线数据
        daily_ok, updated_count = _update_daily_data(days, market, force_update)
        
        if not daily_ok:
            logger.error("日线数据更新失败")
            return False
        
        # 3. 判断结果（有更新或已有完整缓存都算成功）
        if updated_count == 0:
            logger.info("日线数据已全部缓存，无需更新")
        else:
            logger.info(f"数据更新完成，共更新 {updated_count} 条日线数据")
        return True
            
    except Exception as e:
        logger.error(f"数据更新异常: {e}")
        return False


# ==================== MCP/Tool 包装 ====================

def get_mcp_tool() -> dict:
    """
    获取MCP工具定义
    
    Returns:
        MCP工具定义字典
    """
    return {
        "name": "update_stock_data",
        "description": "更新股票数据缓存（基础数据和日线数据）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "需要缓存的天数，默认15天",
                    "default": 15
                },
                "market": {
                    "type": "string",
                    "description": "市场类型：'all'全市场，'sh'上海，'sz'深圳",
                    "enum": ["all", "sh", "sz"],
                    "default": "all"
                },
                "force_update": {
                    "type": "boolean",
                    "description": "是否强制更新所有数据，默认False",
                    "default": False
                }
            },
            "required": []
        }
    }


def handle_mcp_call(arguments: dict) -> dict:
    """
    处理MCP工具调用
    
    Args:
        arguments: 工具参数
        
    Returns:
        工具执行结果
    """
    try:
        days = arguments.get("days", 15)
        market = arguments.get("market", "all")
        force_update = arguments.get("force_update", False)
        
        ok = update_stock_data(days=days, market=market, force_update=force_update)
        msg = "数据更新完成" if ok else "数据更新失败"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": msg
                }
            ]
        }
    except Exception as e:
        logger.error(f"MCP工具调用失败: {e}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": "数据更新失败"
                }
            ],
            "isError": True
        }


# ==================== 主函数 ====================

def main():
    """主函数 - CLI入口"""
    print("=" * 70)
    print("股票数据更新工具")
    print("=" * 70)
    
    result = update_stock_data(days=15, market="all", force_update=False)
    
    print(f"\n更新结果: {result}")
    print("=" * 70)


if __name__ == "__main__":
    main()

