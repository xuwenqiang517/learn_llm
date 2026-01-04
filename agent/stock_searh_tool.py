"""
MCP 股票搜索服务模块

提供A股股票查询功能，筛选连续上涨的股票。
采用两步式架构：
1. 预热缓存：stock_data_cache_warmer() 将远程数据批量下载到本地
2. 本地分析：get_rising_stocks() 完全基于本地缓存进行分析

依赖：
    pip install akshare tabulate tqdm

使用示例：
    from agent.stock_searh_mcp import stock_data_cache_warmer, get_rising_stocks, search_rising_stocks

    # 预热缓存（耗时较长，但只需执行一次）
    stock_data_cache_warmer(days=15)

    # 基于缓存分析（秒级完成）
    result = search_rising_stocks(days=3, market="all")
    print(result)
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_DIR = Path(__file__).parent.parent / ".temp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STOCK_LIST_FILE = CACHE_DIR / "stock_list.json"
INDUSTRY_MAP_FILE = CACHE_DIR / "industry_map.json"
CONCEPT_MAP_FILE = CACHE_DIR / "concept_map.json"


def load_json(file_path: Path) -> Optional[any]:
    try:
        return json.load(open(file_path, 'r', encoding='utf-8')) if file_path.exists() else None
    except Exception:
        return None


def save_json(data: any, file_path: Path) -> None:
    """保存 JSON 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_stock_list_with_cache() -> pd.DataFrame:
    """获取股票列表（带缓存）"""
    cached = load_json(STOCK_LIST_FILE)
    if cached is not None:
        df = pd.DataFrame(cached)
        if not df.empty:
            return df
    
    stock_list = ak.stock_info_a_code_name()
    if not stock_list.empty:
        save_json(stock_list.to_dict('records'), STOCK_LIST_FILE)
    return stock_list


def get_industry_map_with_cache() -> dict:
    """获取行业板块映射（带缓存）- 优化版"""
    cached = load_json(INDUSTRY_MAP_FILE)
    if cached is not None:
        return cached
    
    industry_map = {}
    try:
        stock_board = ak.stock_board_industry_name_em()
        if stock_board.empty:
            return {}
        
        industries = stock_board['板块名称'].tolist()
        
        for industry in industries:
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
            save_json(industry_map, INDUSTRY_MAP_FILE)
    except Exception:
        pass
    
    return industry_map


def get_concept_map_with_cache() -> dict:
    """获取概念板块映射（带缓存）"""
    cached = load_json(CONCEPT_MAP_FILE)
    if cached is not None:
        return cached
    
    concept_map = {}
    try:
        concept_df = ak.stock_board_concept_name_em()
        if concept_df is not None and not concept_df.empty and '板块名称' in concept_df.columns:
            concepts = concept_df['板块名称'].tolist()
            
            for concept in concepts:
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
                save_json(concept_map, CONCEPT_MAP_FILE)
    except Exception:
        pass
    
    return concept_map


def get_daily_cache_path(symbol: str, date: str) -> Path:
    """获取单日缓存文件路径"""
    return CACHE_DIR / "daily" / date / f"{symbol}.json"


def get_trading_days(days: int = 15) -> list:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        df = ak.tool_trade_date_hist_sina()
        if df is not None and 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            return df.loc[mask, 'trade_date'].dt.strftime('%Y%m%d').tolist()
    except Exception:
        pass
    
    trading_days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            trading_days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return trading_days


def get_stock_daily_with_cache(symbol: str, date: str) -> Optional[dict]:
    cache_file = get_daily_cache_path(symbol, date)
    return load_json(cache_file)


def download_and_cache_stock_daily(symbol: str, date: str) -> Optional[dict]:
    """下载并缓存单只股票单日数据"""
    cache_file = get_daily_cache_path(symbol, date)
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
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
            
            save_json(data, cache_file)
            return data
    except Exception:
        pass
    
    return None


def stock_data_cache_warmer(days: int = 15, market: str = "all") -> None:
    trading_days = get_trading_days(days)
    
    print("=" * 70)
    print("股票数据缓存预热")
    print("=" * 70)
    
    print(f"\n[1/3] 获取股票列表...")
    stock_list = get_stock_list_with_cache()
    if stock_list.empty:
        print("获取股票列表失败")
        return
    print(f"  获取到 {len(stock_list)} 只股票")
    
    if market == "sh":
        stock_list = stock_list[stock_list['code'].str.startswith('6')]
    elif market == "sz":
        stock_list = stock_list[stock_list['code'].str.startswith(('0', '3'))]
    print(f"  筛选后 {len(stock_list)} 只股票（{market}）")
    
    print(f"\n[2/3] 获取行业板块信息...")
    industry_map = get_industry_map_with_cache()
    print(f"  获取到 {len(industry_map)} 只股票的行业信息")
    
    print(f"\n[3/3] 下载股票日线数据...")
    stock_codes = stock_list['code'].tolist()
    total_tasks = len(stock_codes) * len(trading_days)
    
    cache_tasks = [(code, date) for code in stock_codes for date in trading_days
                   if not get_daily_cache_path(code, date).exists()]
    
    print(f"  需要下载: {len(cache_tasks)}/{total_tasks} 条数据")
    
    if not cache_tasks:
        print("  所有数据已缓存")
        print(f"\n缓存位置: {CACHE_DIR}")
        return
    
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as executor:
        futures = {executor.submit(download_and_cache_stock_daily, code, date): (code, date)
                   for code, date in cache_tasks}
        
        with tqdm(total=len(futures), desc="下载进度", unit="条") as pbar:
            for future in as_completed(futures):
                pbar.update(1)
                pbar.set_postfix_str(f"{futures[future][0]} {futures[future][1]}")
    
    print(f"\n缓存预热完成! 缓存位置: {CACHE_DIR}")


def load_stock_data_from_cache(symbol: str, dates: list) -> Optional[pd.DataFrame]:
    """从本地缓存加载股票历史数据"""
    records = []
    for date in dates:
        data = get_stock_daily_with_cache(symbol, date)
        if data is not None:
            records.append(data)
    
    if not records:
        return None
    
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def analyze_stock_from_cache(code: str, name: str, dates: list, industry_map: dict, concept_map: dict, rising_days: int, min_increase: float = 5.0) -> Optional[dict]:
    """从缓存分析单只股票
    
    Args:
        code: 股票代码
        name: 股票名称
        dates: 需要分析的日期列表
        industry_map: 行业板块映射
        concept_map: 概念板块映射
        rising_days: 连续上涨天数
        min_increase: 最小累计涨幅阈值，默认5%
    
    Returns:
        符合条件返回股票信息，否则返回None
    """
    df = load_stock_data_from_cache(code, dates)
    
    if df is None or df.empty or len(df) < rising_days:
        return None
    
    df = df.tail(rising_days).reset_index(drop=True)
    
    daily_returns = []
    daily_volumes = []
    for i in range(rising_days):
        change_pct = df.iloc[i].get('change_pct', 0)
        volume = df.iloc[i].get('volume', 0)
        daily_returns.append(round(change_pct, 2))
        daily_volumes.append(volume)
    
    if len(daily_returns) >= rising_days and all(r > 0 for r in daily_returns[-rising_days:]):
        total_return = sum(daily_returns[-rising_days:])
        
        if total_return < min_increase:
            return None
        
        return {
            'code': code,
            'name': name,
            'dates': df['date'].tolist()[-rising_days:],
            'daily_increases': [round(r, 2) for r in daily_returns[-rising_days:]],
            'daily_volumes': [int(v) for v in daily_volumes[-rising_days:]],
            'total_increase': round(total_return, 2),
            'last_close': round(df.iloc[-1].get('close', 0), 2),
            'last_volume': int(df.iloc[-1].get('volume', 0)),
            'industry': industry_map.get(code, '未知'),
            'concepts': concept_map.get(code, [])
        }
    
    return None


def get_rising_stocks(days: int = 3, market: str = "all", min_increase: float = 5.0, include_kc: bool = False, include_cy: bool = False) -> pd.DataFrame:
    """
    筛选连续 N 天上涨的股票（完全基于本地缓存分析）
    
    Args:
        days: 连续上涨天数，默认 3 天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        min_increase: 最小累计涨幅阈值，默认5%
        include_kc: 是否包含科创板，默认True
        include_cy: 是否包含创业板，默认True
    
    Returns:
        包含符合条件的股票信息的 DataFrame
    """
    trading_days = get_trading_days(days + 5)
    
    print("正在从本地缓存分析股票数据...")
    
    print("  加载股票列表...")
    stock_list = get_stock_list_with_cache()
    if stock_list.empty:
        return pd.DataFrame()
    
    if market == "sh":
        stock_list = stock_list[stock_list['code'].str.startswith('6')]
    elif market == "sz":
        stock_list = stock_list[stock_list['code'].str.startswith(('0', '3'))]
    
    if not include_kc:
        stock_list = stock_list[~stock_list['code'].str.startswith('68')]
    if not include_cy:
        stock_list = stock_list[~stock_list['code'].str.startswith('3')]
    
    print("  加载行业板块信息...")
    industry_map = get_industry_map_with_cache()
    
    print("  加载概念板块信息...")
    concept_map = get_concept_map_with_cache()
    
    stock_codes = stock_list['code'].tolist()
    print(f"  分析 {len(stock_codes)} 只股票...")
    
    rising_stocks = []
    stock_tuples = [(row['code'], row['name']) for _, row in stock_list.iterrows()]
    
    for idx, (code, name) in enumerate(stock_tuples):
        result = analyze_stock_from_cache(code, name, trading_days, industry_map, concept_map, days, min_increase)
        if result:
            rising_stocks.append(result)
        
        if (idx + 1) % 500 == 0 or idx == len(stock_tuples) - 1:
            print(f"    进度: {idx + 1}/{len(stock_tuples)} ({100 * (idx + 1) / len(stock_tuples):.1f}%)")
    
    result_df = pd.DataFrame(rising_stocks)
    if not result_df.empty:
        result_df = result_df.sort_values('total_increase', ascending=False)
    
    return result_df


def format_stock_result(df: pd.DataFrame, save_path: Optional[Path] = None) -> str:
    """格式化股票查询结果为JSON"""
    if df.empty:
        return json.dumps({"message": "未找到符合条件的股票"}, ensure_ascii=False, indent=2)
    
    if len(df) == 0:
        return json.dumps({"message": "未找到符合条件的股票"}, ensure_ascii=False, indent=2)
    
    sample_row = df.iloc[0]
    dates = sample_row.get('dates', [])
    
    if len(dates) >= 3:
        date_headers = [datetime.strptime(d, "%Y%m%d").strftime("%m-%d") for d in dates[-3:]]
    else:
        date_headers = ['日期1', '日期2', '日期3']
    
    stocks = []
    for _, row in df.iterrows():
        daily_increases = row.get('daily_increases', [])
        daily_volumes = row.get('daily_volumes', [])
        
        daily_data = []
        for i in range(3):
            if i < len(daily_increases):
                vol = daily_volumes[-(3-i)] if i < len(daily_volumes) else 0
                if vol >= 1000000:
                    vol_str = f"{vol/1000000:.1f}M"
                elif vol >= 1000:
                    vol_str = f"{vol/1000:.1f}K"
                else:
                    vol_str = str(vol)
                
                daily_data.append({
                    "date": date_headers[i],
                    "increase": f"{daily_increases[-(3-i)]:+.2f}%",
                    "volume": vol_str
                })
            else:
                daily_data.append({
                    "date": date_headers[i],
                    "increase": "-",
                    "volume": "-"
                })
        
        stocks.append({
            "code": row['code'],
            "name": row['name'],
            "industry": row['industry'],
            "concepts": row.get('concepts', []),
            "last_close": round(row['last_close'], 2),
            "last_volume": f"{row['last_volume']:,}" if isinstance(row['last_volume'], (int, float)) else row['last_volume'],
            "total_increase": f"{row['total_increase']:+.2f}%",
            "daily_data": daily_data
        })
    
    result = {
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_count": len(df),
        "rising_days": 3,
        "stocks": stocks
    }
    
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result_json)
    
    return result_json


def search_rising_stocks(days: int = 3, market: str = "all", current_date: Optional[str] = None, save_result: bool = True, use_cache: bool = True, min_increase: float = 5.0, include_kc: bool = True, include_cy: bool = True) -> dict:
    """
    搜索连续N天上涨的股票（服务主入口）

    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        current_date: 查询日期，格式YYYYMMDD，默认为今天
        save_result: 是否保存结果到文件
        use_cache: 是否使用缓存的查询结果
        min_increase: 最小累计涨幅阈值，默认5%
        include_kc: 是否包含科创板，默认True
        include_cy: 是否包含创业板，默认True

    Returns:
        包含查询结果的字典
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y%m%d")

    cache_file = CACHE_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.json"

    if use_cache and cache_file.exists():
        cached = load_json(cache_file)
        if cached:
            return {
                "success": True,
                "message": f"从缓存读取，找到 {len(cached.get('stocks', []))} 只连续{cached.get('rising_days', days)}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
                "data": cached,
                "saved_path": str(cache_file),
                "from_cache": True
            }

    rising_df = get_rising_stocks(days=days, market=market, min_increase=min_increase, include_kc=include_kc, include_cy=include_cy)

    if rising_df.empty:
        return {
            "success": True,
            "message": f"未找到符合条件的股票或缓存数据不足（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
            "data": []
        }

    save_path = cache_file if save_result else None

    result_json = format_stock_result(rising_df, save_path=save_path)
    result = json.loads(result_json)

    return {
        "success": True,
        "message": f"找到 {len(result['stocks'])} 只连续{result['rising_days']}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
        "data": result,
        "saved_path": str(save_path) if save_path else None,
        "from_cache": False
    }


def main():
    """主函数 - CLI入口"""
    result = search_rising_stocks(days=3, market="all")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
