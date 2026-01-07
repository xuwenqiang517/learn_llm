"""
股票连续上涨计算工具

独立的MCP/Tool，负责计算连续上涨的股票。
基于本地缓存数据进行分析，返回JSON格式结果和Markdown表格。
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

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
OUTPUT_DIR = TEMP_DIR / "output"
TOOLS_OUTPUT_DIR = OUTPUT_DIR / "tools"

# 确保目录存在
FileUtil.ensure_dirs(TOOLS_OUTPUT_DIR)

# 基础数据文件路径
STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.json"
INDUSTRY_MAP_FILE = BASE_DATA_DIR / "industry_map.json"
CONCEPT_MAP_FILE = BASE_DATA_DIR / "concept_map.json"

# 概念黑名单
BLACKLIST_CONCEPTS = {
    '昨日连板', '昨日涨停', '昨日涨停_含一字', '昨日连板_含一字',
    '今日涨停', '今日连板', '近期强势股', '近期活跃股',
    '融资融券', '沪股通', '深股通', '港股通', '龙虎榜', '机构重仓'
}


# ==================== 辅助函数 ====================

def _get_trading_days(days: int = 15) -> List[str]:
    """
    获取交易日列表（简化版，基于日期计算）
    
    Args:
        days: 向前查找的天数
        
    Returns:
        交易日列表（格式：YYYYMMDD）
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    trading_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 周一到周五
            trading_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return trading_days


def _get_stock_daily(symbol: str, date: str) -> Optional[Dict]:
    """
    从缓存获取单只股票单日数据
    
    Args:
        symbol: 股票代码
        date: 日期（YYYYMMDD）
        
    Returns:
        股票数据字典，不存在返回None
    """
    cache_file = DAILY_DATA_DIR / date / f"{symbol}.json"
    return JsonUtil.load(cache_file)


def _load_stock_data_from_cache(symbol: str, dates: List[str]) -> Optional[pd.DataFrame]:
    """
    从本地缓存加载股票历史数据
    
    Args:
        symbol: 股票代码
        dates: 日期列表
        
    Returns:
        股票数据DataFrame，失败返回None
    """
    records = []
    for date in dates:
        data = _get_stock_daily(symbol, date)
        if data is not None:
            records.append(data)
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def _filter_concepts(concepts: List[str]) -> List[str]:
    """
    过滤掉黑名单概念
    
    Args:
        concepts: 概念列表
        
    Returns:
        过滤后的概念列表
    """
    if not concepts:
        return []
    return [c for c in concepts if c not in BLACKLIST_CONCEPTS]


def _analyze_stock(code: str, name: str, dates: List[str], 
                  industry_map: Dict[str, str], concept_map: Dict[str, List[str]],
                  min_increase: float = 5.0, check_st: bool = True) -> Optional[Dict]:
    """
    分析单只股票
    
    Args:
        code: 股票代码
        name: 股票名称
        dates: 需要分析的日期列表
        industry_map: 行业板块映射
        concept_map: 概念板块映射
        min_increase: 最小累计涨幅阈值（%）
        check_st: 是否检查ST股票
        
    Returns:
        符合条件返回股票信息字典，否则返回None
    """
    # 过滤ST股票
    if check_st and ('ST' in name.upper() or '*ST' in name.upper()):
        return None
    
    # 加载数据
    df = _load_stock_data_from_cache(code, dates)
    if df is None or df.empty or len(df) < 3:
        return None
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 从最近一天往前计算连续上涨天数
    actual_rising_days = 0
    total_return = 0.0
    
    for i in range(len(df) - 1, -1, -1):
        change_pct = df.iloc[i].get('change_pct', 0)
        if change_pct > 0:
            actual_rising_days += 1
            total_return += change_pct
        else:
            break
    
    # 检查条件
    if actual_rising_days < 3 or total_return < min_increase:
        return None
    
    # 提取连涨期间的数据
    last_n_days_df = df.tail(actual_rising_days).reset_index(drop=True)
    
    daily_returns = []
    daily_volumes = []
    for i in range(actual_rising_days):
        change_pct = last_n_days_df.iloc[i].get('change_pct', 0)
        volume = last_n_days_df.iloc[i].get('volume', 0)
        daily_returns.append(round(change_pct, 2))
        daily_volumes.append(volume)
    
    return {
        'code': code,
        'name': name,
        'dates': last_n_days_df['date'].tolist(),
        'daily_increases': daily_returns,
        'daily_volumes': daily_volumes,
        'total_increase': round(total_return, 2),
        'last_close': round(last_n_days_df.iloc[-1].get('close', 0), 2),
        'last_volume': int(last_n_days_df.iloc[-1].get('volume', 0)),
        'industry': industry_map.get(code, '未知'),
        'concepts': concept_map.get(code, []),
        'actual_rising_days': actual_rising_days
    }


def _format_stock_result(df: pd.DataFrame, rising_days: int = 3, 
                         compress: bool = False) -> str:
    """
    格式化股票查询结果为JSON
    
    Args:
        df: 股票DataFrame
        rising_days: 连续上涨天数（用于日期显示）
        compress: 是否压缩数据（减少token消耗）
        
    Returns:
        JSON格式的字符串
    """
    if df.empty:
        return JsonUtil.dumps({"message": "未找到符合条件的股票"})
    
    stocks = []
    for _, row in df.iterrows():
        actual_days = row.get('actual_rising_days', rising_days)
        daily_increases = row.get('daily_increases', [])
        daily_volumes = row.get('daily_volumes', [])
        
        # 每只股票单独计算其对应的日期表头
        row_dates = row.get('dates', [])
        if isinstance(row_dates, list) and len(row_dates) >= 3:
            row_date_headers = [
                datetime.strptime(d, "%Y%m%d").strftime("%m-%d") for d in row_dates[-3:]
            ]
        else:
            row_date_headers = ['日期1', '日期2', '日期3']
        
        last_3_days_increase = sum(daily_increases[-3:]) if len(daily_increases) >= 3 else sum(daily_increases)
        
        if compress:
            # 压缩格式（减少token）
            rising_inc = row.get('total_increase', 0)
            daily_data = []
            for i in range(3):
                if i < len(daily_increases):
                    inc = f"{daily_increases[-(3-i)]:+.1f}%"
                    daily_data.append(f"{row_date_headers[i]}:{inc}")
                else:
                    daily_data.append(f"{row_date_headers[i]}:-")
            
            stocks.append({
                "c": row['code'],
                "n": row['name'],
                "i": row['industry'][:4] if row['industry'] else "未知",
                "con": [c[:8] for c in _filter_concepts(row.get('concepts', []))][:3],
                "p": round(row['last_close'], 2),
                "inc3": f"{last_3_days_increase:+.1f}%",
                "inc_r": f"{rising_inc:+.1f}%",
                "r": actual_days,
                "d": ", ".join(daily_data)
            })
        else:
            # 完整格式
            daily_data = []
            for i in range(3):
                if i < len(daily_increases):
                    vol = daily_volumes[-(3-i)] if i < len(daily_volumes) else 0
                    vol_str = f"{vol/1000000:.1f}M" if vol >= 1000000 else f"{vol/1000:.1f}K" if vol >= 1000 else str(vol)
                    
                    daily_data.append({
                        "date": row_date_headers[i],
                        "increase": f"{daily_increases[-(3-i)]:+.2f}%",
                        "volume": vol_str
                    })
                else:
                    daily_data.append({
                        "date": row_date_headers[i],
                        "increase": "-",
                        "volume": "-"
                    })
            
            stocks.append({
                "code": row['code'],
                "name": row['name'],
                "industry": row['industry'],
                "concepts": _filter_concepts(row.get('concepts', [])),
                "last_close": round(row['last_close'], 2),
                "last_volume": f"{row['last_volume']:,}" if isinstance(row['last_volume'], (int, float)) else row['last_volume'],
                "total_increase": f"{last_3_days_increase:+.2f}%",
                "rising_total_increase": f"{row['total_increase']:+.2f}%",
                "rising_days": actual_days,
                "daily_data": daily_data
            })
    
    result = {
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(df),
        "rising_days": max(row.get('actual_rising_days', rising_days) for _, row in df.iterrows()),
        "stocks": stocks
    }
    
    if compress:
        result["_note"] = "压缩模式: c=code, n=name, i=industry, p=price, inc3=3日涨幅, inc_r=连涨涨幅, r=rising_days, d=daily"
    
    return JsonUtil.dumps(result, indent=None if compress else 2)


# ==================== 表格生成 ====================

def generate_table_from_results(result: Dict, save_path: Optional[Path] = None) -> str:
    """
    从搜索结果生成Markdown表格
    
    Args:
        result: analyze_rising_stocks 返回的结果字典
        save_path: 可选的文件保存路径
        
    Returns:
        Markdown格式的表格字符串
    """
    stocks = result.get('stocks', [])
    if not stocks:
        return "未找到符合条件的股票"
    
    total_count = result.get('total_count', len(stocks))
    query_time = result.get('query_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    sample_stock = stocks[0] if stocks else {}
    daily_data = sample_stock.get('daily_data', [])
    date_headers = [d.get('date', f'D{i+1}') for i, d in enumerate(daily_data[:3])]
    if len(date_headers) < 3:
        date_headers = [f'Day{i+1}' for i in range(3)]
    
    lines = []
    lines.append(f"# 股票数据汇总 ({total_count}只)")
    lines.append(f"\n**查询时间**: {query_time}\n")
    
    header_cols = ["代码", "名称", "连涨", "连涨累计涨幅", "最近3天累计涨幅"]
    header_cols.extend([f"{h}涨幅" for h in date_headers])
    header_cols.extend(["行业", "核心概念"])
    
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["------"] * len(header_cols)) + "|")
    
    stocks_sorted = sorted(
        stocks, 
        key=lambda x: (
            x.get('rising_days', 0), 
            float(x.get('rising_total_increase', '0%').replace('%', '').replace('+', ''))
        ), 
        reverse=True
    )
    
    for stock in stocks_sorted:
        code = stock.get('code', '')
        name = stock.get('name', '')
        rising_days = stock.get('rising_days', '-')
        rising_total_increase = stock.get('rising_total_increase', '-')
        industry = stock.get('industry', '未知')
        total_increase = stock.get('total_increase', '0%')
        daily_data = stock.get('daily_data', [])
        concepts = stock.get('concepts', [])
        filtered_concepts = _filter_concepts(concepts)
        main_concepts = ', '.join(filtered_concepts[:3]) if filtered_concepts else '-'
        
        cols = [code, name, f"{rising_days}天", rising_total_increase, total_increase]
        
        for i in range(3):
            if i < len(daily_data):
                day = daily_data[i]
                inc = day.get('increase', '-')
            else:
                inc = '-'
            cols.append(inc)
        
        cols.append(industry)
        cols.append(main_concepts)
        formatted_cols = [str(c) for c in cols]
        line = "| " + " | ".join(formatted_cols) + " |"
        lines.append(line)
    
    lines.append("")
    lines.append("---")
    lines.append("\n**核心概念统计（按出现频次）**\n")
    
    concept_counter = {}
    for stock in stocks:
        concepts = _filter_concepts(stock.get('concepts', []))
        for c in concepts:
            concept_counter[c] = concept_counter.get(c, 0) + 1
    
    top_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_concepts:
        for concept, count in top_concepts:
            pct = 100 * count / total_count
            lines.append(f"- **{concept}**: {count}只 ({pct:.1f}%)")
    
    full_content = "\n".join(lines)
    
    if save_path:
        FileUtil.write_text(full_content, save_path)
        logger.info(f"表格已保存到: {save_path}")
    
    return full_content


# ==================== 主入口函数 ====================

def _check_cache_valid(days: int, market: str, min_increase: float,
                       include_kc: bool, include_cy: bool) -> Optional[Path]:
    """
    检查缓存是否有效
    
    Args:
        days: 连续上涨天数
        market: 市场类型
        min_increase: 最小累计涨幅阈值
        include_kc: 是否包含科创板
        include_cy: 是否包含创业板
        
    Returns:
        缓存文件路径，如果缓存无效则返回None
    """
    current_date = datetime.now().strftime("%Y%m%d")
    table_file = TOOLS_OUTPUT_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.md"
    
    if table_file.exists():
        return table_file
    return None


def calculate_rising_stocks(days: int = 3, market: str = "all", 
                           min_increase: float = 10.0, 
                           include_kc: bool = False, 
                           include_cy: bool = False,
                           compress: bool = False,
                           save_table: bool = False,
                           table_path: Optional[Path] = None) -> Tuple[str, str]:
    """
    计算连续N天上涨的股票（主入口函数）
    
    内置数据更新和缓存逻辑：
    - 自动检查缓存有效性
    - 自动更新数据（如果需要）
    - 自动保存结果
    
    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        min_increase: 最小累计涨幅阈值（%），默认10.0%
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        compress: 是否压缩JSON格式，默认False
        save_table: 是否保存表格文件，默认False
        table_path: 表格保存路径，如果为None则自动生成
        
    Returns:
        (JSON格式的字符串结果, Markdown表格字符串)
    """
    try:
        current_date = datetime.now().strftime("%Y%m%d")
        
        if table_path is None and save_table:
            table_path = TOOLS_OUTPUT_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.md"
        
        cache_file = _check_cache_valid(days, market, min_increase, include_kc, include_cy)
        
        if cache_file:
            table_content = FileUtil.read_text(cache_file)
            if table_content:
                result_json = analyze_rising_stocks(
                    days=days, market=market, min_increase=min_increase,
                    include_kc=include_kc, include_cy=include_cy,
                    compress=compress
                )
                result_dict = JsonUtil.loads(result_json) or {}
                result_dict["from_cache"] = True
                result_dict["table_output_path"] = str(cache_file)
                return JsonUtil.dumps(result_dict), table_content
        
        trading_days = _get_trading_days(days + 12)
        
        logger.info("开始分析股票数据...")
        
        # 加载基础数据
        stock_list_data = JsonUtil.load(STOCK_LIST_FILE)
        if not stock_list_data:
            return JsonUtil.dumps({"message": "股票列表数据不存在，请先更新数据"}), "数据不存在"
        
        stock_list = pd.DataFrame(stock_list_data)
        if stock_list.empty:
            return JsonUtil.dumps({"message": "股票列表为空"}), "股票列表为空"
        
        # 市场筛选
        if market == "sh":
            stock_list = stock_list[stock_list['code'].str.startswith('6')]
        elif market == "sz":
            stock_list = stock_list[stock_list['code'].str.startswith(('0', '3'))]
        
        # 板块筛选
        if not include_kc:
            stock_list = stock_list[~stock_list['code'].str.startswith('68')]
        if not include_cy:
            stock_list = stock_list[~stock_list['code'].str.startswith('3')]
        
        # 加载板块映射
        industry_map = JsonUtil.load(INDUSTRY_MAP_FILE) or {}
        concept_map = JsonUtil.load(CONCEPT_MAP_FILE) or {}
        
        stock_codes = stock_list['code'].tolist()
        logger.info(f"分析 {len(stock_codes)} 只股票...")
        
        # 分析股票
        rising_stocks = []
        stock_tuples = [(row['code'], row['name']) for _, row in stock_list.iterrows()]
        
        # 使用进度条显示分析进度
        for code, name in tqdm(stock_tuples, desc="分析股票", unit="只"):
            result = _analyze_stock(
                code, name, trading_days, industry_map, concept_map, 
                min_increase, check_st=True
            )
            if result:
                rising_stocks.append(result)
        
        result_df = pd.DataFrame(rising_stocks)
        if not result_df.empty:
            result_df = result_df.sort_values(
                ['actual_rising_days', 'total_increase'], 
                ascending=[False, False]
            )
            logger.info(f"找到 {len(result_df)} 只符合条件的股票")
        
        # 格式化JSON结果
        result_json = _format_stock_result(result_df, rising_days=days, compress=compress)
        result_dict = JsonUtil.loads(result_json) or {}
        
        # 生成表格
        if result_dict.get("stocks"):
            table_content = generate_table_from_results(result_dict, save_path=table_path if save_table else None)
        else:
            table_content = "未找到符合条件的股票"
        
        result_dict["from_cache"] = False
        return result_json, table_content
        
    except Exception as e:
        logger.error(f"计算股票数据失败: {e}")
        error_result = JsonUtil.dumps({"message": f"计算失败: {str(e)}"})
        return error_result, "计算失败"


# ==================== 兼容性函数（保持向后兼容）====================

def analyze_rising_stocks(days: int = 3, market: str = "all", 
                         min_increase: float = 10.0, 
                         include_kc: bool = False, 
                         include_cy: bool = False,
                         compress: bool = False) -> str:
    """
    分析连续N天上涨的股票（兼容性函数，仅返回JSON）
    
    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        min_increase: 最小累计涨幅阈值（%），默认10.0%
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        compress: 是否压缩JSON格式，默认False
        
    Returns:
        JSON格式的字符串结果
    """
    result_json, _ = calculate_rising_stocks(
        days=days, market=market, min_increase=min_increase,
        include_kc=include_kc, include_cy=include_cy, compress=compress
    )
    return result_json


# ==================== MCP/Tool 包装 ====================

def get_mcp_tool() -> Dict:
    """
    获取MCP工具定义
    
    Returns:
        MCP工具定义字典
    """
    return {
        "name": "analyze_rising_stocks",
        "description": "分析连续N天上涨的A股股票（基于本地缓存数据）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "连续上涨天数，默认3天",
                    "default": 3
                },
                "market": {
                    "type": "string",
                    "description": "市场类型：'all'全市场，'sh'上海，'sz'深圳",
                    "enum": ["all", "sh", "sz"],
                    "default": "all"
                },
                "min_increase": {
                    "type": "number",
                    "description": "最小累计涨幅阈值（%），默认10.0",
                    "default": 10.0
                },
                "include_kc": {
                    "type": "boolean",
                    "description": "是否包含科创板，默认False",
                    "default": False
                },
                "include_cy": {
                    "type": "boolean",
                    "description": "是否包含创业板，默认False",
                    "default": False
                },
                "compress": {
                    "type": "boolean",
                    "description": "是否压缩JSON格式（减少token），默认False",
                    "default": False
                }
            },
            "required": []
        }
    }


def handle_mcp_call(arguments: Dict) -> Dict:
    """
    处理MCP工具调用
    
    Args:
        arguments: 工具参数
        
    Returns:
        工具执行结果
    """
    try:
        days = arguments.get("days", 3)
        market = arguments.get("market", "all")
        min_increase = arguments.get("min_increase", 10.0)
        include_kc = arguments.get("include_kc", False)
        include_cy = arguments.get("include_cy", False)
        compress = arguments.get("compress", False)
        
        result_json, _ = calculate_rising_stocks(
            days=days,
            market=market,
            min_increase=min_increase,
            include_kc=include_kc,
            include_cy=include_cy,
            compress=compress
        )
        result = result_json
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": result
                }
            ]
        }
    except Exception as e:
        logger.error(f"MCP工具调用失败: {e}")
        return {
            "content": [
                {
                    "type": "text",
                    "text": JsonUtil.dumps({"error": str(e)})
                }
            ],
            "isError": True
        }


# ==================== 主函数 ====================

def main():
    """主函数 - CLI入口"""
    print("=" * 70)
    print("股票分析工具 - 连续上涨股票分析")
    print("=" * 70)
    
    result_json, table_content = calculate_rising_stocks(
        days=3, 
        market="all", 
        min_increase=10.0, 
        include_kc=False, 
        include_cy=False,
        compress=False,
        save_table=True
    )
    result = result_json
    
    result_dict = JsonUtil.loads(result)
    if result_dict:
        stocks = result_dict.get("stocks", [])
        print(f"\n找到 {len(stocks)} 只符合条件的股票")
        print(f"查询时间: {result_dict.get('query_time', 'N/A')}")
        print("\n前5只股票:")
        for i, stock in enumerate(stocks[:5], 1):
            print(f"{i}. {stock.get('code', '')} {stock.get('name', '')} "
                  f"连涨{stock.get('rising_days', 0)}天 "
                  f"累计涨幅{stock.get('rising_total_increase', '0%')}")
    
    print("\n" + "=" * 70)
    print("完整结果（JSON）:")
    print("=" * 70)
    print(result)


if __name__ == "__main__":
    main()

