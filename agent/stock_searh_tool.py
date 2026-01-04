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
import logging
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

logger = logging.getLogger(__name__)

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


def analyze_stock_from_cache(code: str, name: str, dates: list, industry_map: dict, concept_map: dict, min_increase: float = 5.0, check_st: bool = True) -> Optional[dict]:
    """从缓存分析单只股票
    
    Args:
        code: 股票代码
        name: 股票名称
        dates: 需要分析的日期列表
        industry_map: 行业板块映射
        concept_map: 概念板块映射
        min_increase: 最小累计涨幅阈值，默认5%
        check_st: 是否检查ST股票，默认True
    
    Returns:
        符合条件返回股票信息，否则返回None
    """
    if check_st and ('ST' in name.upper() or '*ST' in name.upper() or 'ST' in name):
        return None
    
    df = load_stock_data_from_cache(code, dates)
    
    if df is None or df.empty:
        return None
    
    if len(df) < 3:
        return None
    
    df = df.sort_values('date').reset_index(drop=True)
    
    actual_rising_days = 0
    total_return = 0.0
    
    for i in range(len(df) - 1, -1, -1):
        change_pct = df.iloc[i].get('change_pct', 0)
        if change_pct > 0:
            actual_rising_days += 1
            total_return += change_pct
        else:
            break
    
    if actual_rising_days < 3:
        return None
    
    if total_return < min_increase:
        return None
    
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
    trading_days = get_trading_days(days + 12)
    
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
        result = analyze_stock_from_cache(code, name, trading_days, industry_map, concept_map, min_increase, check_st=True)
        if result:
            rising_stocks.append(result)
        
        if (idx + 1) % 500 == 0 or idx == len(stock_tuples) - 1:
            print(f"    进度: {idx + 1}/{len(stock_tuples)} ({100 * (idx + 1) / len(stock_tuples):.1f}%)")
    
    result_df = pd.DataFrame(rising_stocks)
    if not result_df.empty:
        result_df = result_df.sort_values(['actual_rising_days', 'total_increase'], ascending=[False, False])

    return result_df


def format_stock_result(df: pd.DataFrame, rising_days: int = 3, save_path: Optional[Path] = None, compress: bool = False) -> str:
    """格式化股票查询结果为JSON

    Args:
        df: 股票DataFrame
        rising_days: 连续上涨天数（默认3天，用于日期显示）
        save_path: 可选的文件保存路径
        compress: 是否压缩数据（减少token消耗）

    Returns:
        JSON格式的字符串
    """
    if df.empty:
        return json.dumps({"message": "未找到符合条件的股票"}, ensure_ascii=False, indent=2)

    sample_row = df.iloc[0]
    dates = sample_row.get('dates', [])

    if len(dates) >= 3:
        date_headers = [datetime.strptime(d, "%Y%m%d").strftime("%m-%d") for d in dates[-3:]]
    else:
        date_headers = ['日期1', '日期2', '日期3']

    stocks = []
    for _, row in df.iterrows():
        actual_days = row.get('actual_rising_days', rising_days)
        daily_increases = row.get('daily_increases', [])
        daily_volumes = row.get('daily_volumes', [])

        last_3_days_increase = sum(daily_increases[-3:]) if len(daily_increases) >= 3 else sum(daily_increases)

        if compress:
            daily_data = []
            for i in range(3):
                if i < len(daily_increases):
                    inc = f"{daily_increases[-(3-i)]:+.1f}%"
                    vol = daily_volumes[-(3-i)] if i < len(daily_volumes) else 0
                    vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K" if vol >= 1e3 else str(vol)
                    daily_data.append(f"{date_headers[i]}:{inc}")
                else:
                    daily_data.append(f"{date_headers[i]}:-")

            stocks.append({
                "c": row['code'],
                "n": row['name'],
                "i": row['industry'][:4] if row['industry'] else "未知",
                "con": [c[:8] for c in filter_concepts(row.get('concepts', []))][:3],
                "p": round(row['last_close'], 2),
                "t": f"{last_3_days_increase:+.1f}%",
                "r": actual_days,
                "d": ", ".join(daily_data)
            })
        else:
            daily_data = []
            for i in range(3):
                if i < len(daily_increases):
                    vol = daily_volumes[-(3-i)] if i < len(daily_volumes) else 0
                    vol_str = f"{vol/1000000:.1f}M" if vol >= 1000000 else f"{vol/1000:.1f}K" if vol >= 1000 else str(vol)

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
                "concepts": filter_concepts(row.get('concepts', [])),
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
        result["_note"] = "压缩模式: c=code, n=name, i=industry, p=price, t=total, r=rising_days, d=daily"

    result_json = json.dumps(result, ensure_ascii=False, indent=2 if not compress else None)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(result_json)

    return result_json


BLACKLIST_CONCEPTS = {
    '昨日连板', '昨日涨停', '昨日涨停_含一字', '昨日连板_含一字',
    '今日涨停', '今日连板', '近期强势股', '近期活跃股',
    '融资融券', '沪股通', '深股通', '港股通', '龙虎榜', '机构重仓'
}

def filter_concepts(concepts: list) -> list:
    """过滤掉黑名单概念"""
    if not concepts:
        return []
    return [c for c in concepts if c not in BLACKLIST_CONCEPTS]


def generate_table_from_results(result: dict, save_path: Optional[Path] = None) -> str:
    """从搜索结果生成结构化数据展示

    Args:
        result: search_rising_stocks 返回的结果字典
        save_path: 可选的文件保存路径

    Returns:
        Markdown格式的结构化数据字符串
    """
    stocks = result.get('data', {}).get('stocks', [])
    if not stocks:
        return "未找到符合条件的股票"

    total_count = result.get('data', {}).get('total_count', len(stocks))
    query_time = result.get('data', {}).get('query_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    sample_stock = stocks[0] if stocks else {}
    daily_data = sample_stock.get('daily_data', [])
    date_headers = [d.get('date', f'D{i+1}') for i, d in enumerate(daily_data[:3])]
    if len(date_headers) < 3:
        date_headers = [f'Day{i+1}' for i in range(3)]

    lines = []
    lines.append(f"# 股票数据汇总 ({total_count}只)")
    lines.append(f"\n**查询时间**: {query_time}\n")
    lines.append("| 代码 | 名称 | 连涨 | 连涨累计涨幅 | 最近3天累计涨幅 | " + " | ".join([f"{h}涨幅" for h in date_headers]) + " | 行业 | 核心概念 |")
    sep_count = 5 + len(date_headers) + 2
    lines.append("|" + "|".join(["------"] * sep_count) + "|")

    stocks_sorted = sorted(stocks, key=lambda x: (x.get('rising_days', 0), float(x.get('total_increase', '0%').replace('%', '').replace('+', ''))), reverse=True)

    for stock in stocks_sorted:
        code = stock.get('code', '')
        name = stock.get('name', '')
        rising_days = stock.get('rising_days', '-')
        rising_total_increase = stock.get('rising_total_increase', '-')
        industry = stock.get('industry', '未知')
        total_increase = stock.get('total_increase', '0%')
        daily_data = stock.get('daily_data', [])
        concepts = stock.get('concepts', [])
        filtered_concepts = filter_concepts(concepts)
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
        concepts = filter_concepts(stock.get('concepts', []))
        for c in concepts:
            concept_counter[c] = concept_counter.get(c, 0) + 1

    top_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_concepts:
        for concept, count in top_concepts:
            pct = 100 * count / total_count
            lines.append(f"- **{concept}**: {count}只 ({pct:.1f}%)")

    full_content = "\n".join(lines)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_content)
        logger.info(f"表格已保存到: {save_path}")

    return full_content


def generate_summary_stats(result: dict) -> str:
    """生成统计摘要信息
    
    Args:
        result: search_rising_stocks 返回的结果字典
    
    Returns:
        统计摘要字符串
    """
    stocks = result.get('data', {}).get('stocks', [])
    if not stocks:
        return "暂无数据"
    
    total_count = len(stocks)
    
    concept_counter = {}
    industry_counter = {}
    total_increase_sum = 0
    
    for stock in stocks:
        total_increase = float(stock.get('total_increase', '0%').replace('%', '').replace('+', ''))
        total_increase_sum += total_increase
        
        for concept in stock.get('concepts', []):
            concept_counter[concept] = concept_counter.get(concept, 0) + 1
        
        industry = stock.get('industry', '未知')
        industry_counter[industry] = industry_counter.get(industry, 0) + 1
    
    top_concepts = sorted(concept_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    top_industries = sorted(industry_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    
    avg_increase = total_increase_sum / total_count if total_count > 0 else 0
    
    summary = f"""## 统计摘要

| 指标 | 数值 |
|------|------|
| 股票总数 | {total_count} 只 |
| 平均累计涨幅 | {avg_increase:+.2f}% |
| 涉及概念数 | {len(concept_counter)} 个 |
| 涉及行业数 | {len(industry_counter)} 个 |

### 热门概念 TOP5

| 概念 | 出现次数 | 占比 |
|------|----------|------|
"""
    for concept, count in top_concepts:
        pct = 100 * count / total_count
        summary += f"| {concept} | {count} | {pct:.1f}% |\n"
    
    summary += f"""
### 热门行业 TOP5

| 行业 | 出现次数 | 占比 |
|------|----------|------|
"""
    for industry, count in top_industries:
        pct = 100 * count / total_count
        summary += f"| {industry} | {count} | {pct:.1f}% |\n"
    
    return summary


def search_rising_stocks(days: int = 3, market: str = "all", current_date: Optional[str] = None, save_result: bool = True, use_cache: bool = True, min_increase: float = 10.0, include_kc: bool = False, include_cy: bool = False) -> dict:
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
        包含查询结果的字典（包含data和table两个字段）
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y%m%d")

    cache_file = CACHE_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.json"
    table_file = CACHE_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.md"

    if use_cache and cache_file.exists():
        cached = load_json(cache_file)
        if cached:
            table_content = ""
            if table_file.exists():
                table_content = table_file.read_text(encoding='utf-8')
            else:
                table_content = generate_table_from_results({"data": cached})

            return {
                "success": True,
                "message": f"从缓存读取，找到 {len(cached.get('stocks', []))} 只连续{cached.get('rising_days', days)}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
                "data": cached,
                "table": table_content,
                "saved_path": str(cache_file),
                "table_path": str(table_file),
                "from_cache": True
            }

    rising_df = get_rising_stocks(days=days, market=market, min_increase=min_increase, include_kc=include_kc, include_cy=include_cy)

    if rising_df.empty:
        return {
            "success": True,
            "message": f"未找到符合条件的股票或缓存数据不足（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
            "data": [],
            "table": "未找到符合条件的股票",
            "saved_path": None,
            "table_path": None,
            "from_cache": False
        }

    save_path = cache_file if save_result else None

    result_json = format_stock_result(rising_df, rising_days=days, save_path=save_path)
    result = json.loads(result_json)

    table_content = generate_table_from_results({"data": result}, save_path=table_file)

    return {
        "success": True,
        "message": f"找到 {len(result['stocks'])} 只连续{result['rising_days']}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
        "data": result,
        "table": table_content,
        "saved_path": str(save_path) if save_path else None,
        "table_path": str(table_file),
        "from_cache": False
    }


def main():
    """主函数 - CLI入口"""
    search_rising_stocks(days=3, market="all", min_increase=10.0,use_cache=False)
    


if __name__ == "__main__":
    main()
