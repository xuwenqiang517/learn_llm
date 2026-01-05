"""
大盘概念分析计算工具

【核心目标】
定位目前的大盘主线趋势，通过分析各个概念板块的表现，帮助识别市场热点和投资机会。

【流程】
1.基于本地缓存数据（.temp/data/） 过滤出最近5个交易日的股票数据
2.排除掉异常股票，包括
   - ST股票（名称包含'ST'或'*ST'）
   - 新股（代码以9开头，如920045）
   - 停牌股票（任意一天无数据）
   - 异常涨跌幅（单日涨跌幅超过21%）
   - 黑名单概念（如"昨日触板"等无参考价值的概念）
3.用正常的股票数据进行分析，计算每个概念的指标，按累计平均涨跌幅排序，识别热门概念，取top10
4.分析结果按json格式返回，只包含Top 10概念的基本信息：
   - 概念名称
   - 最近5个交易日累计涨跌幅
   - 对应上涨股票数量
   - 对应下跌股票数量
   支持tool/mcp/main 用于后续agent调用和直接调用
5. 可视化输出 保存到当前项目的.temp/output/tools/目录
   - 生成PNG格式图表，包含3个子图：
     * 子图1：概念累计平均涨跌幅（Top N，横向柱状图，红绿配色）
     * 子图2：概念涨跌统计（上涨/下跌次数对比）
     * 子图3：股票详情表格 按5天累计涨幅排序（每个概念显示最多20只股票）
   - 表格特性：
     * 使用实际日期（MM-DD格式）作为表头
     * 合并股票名称和代码（如：贵州茅台(600519)）
     * 概念名称与日期在同一行，蓝色背景作为表头
     * 添加五日累计涨跌幅列（放在最前面）
     * 交替行颜色提高可读性

"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_util import JsonUtil
from utils.file_util import FileUtil
from utils.log_util import LogUtil

logger = LogUtil.get_logger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    '融资融券', '沪股通', '深股通', '港股通', '龙虎榜', '机构重仓',
    '昨日触板'
}


# ==================== 辅助函数 ====================

def _get_trading_days(days: int = 10) -> List[str]:
    """
    获取交易日列表（从实际数据目录中获取）
    
    Args:
        days: 向前查找的天数
        
    Returns:
        交易日列表（格式：YYYYMMDD），从新到旧排序
    """
    trading_days = []
    
    # 从数据目录中获取实际存在的日期
    if DAILY_DATA_DIR.exists():
        # 获取所有日期目录
        date_dirs = sorted([d.name for d in DAILY_DATA_DIR.iterdir() if d.is_dir()], reverse=True)
        trading_days = date_dirs[:days]
    
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


def _is_st_stock(name: str) -> bool:
    """
    检查是否为ST股票
    
    Args:
        name: 股票名称
        
    Returns:
        True表示是ST股票
    """
    return 'ST' in name.upper() or '*ST' in name.upper()


def _is_abnormal_stock(code: str, name: str, daily_changes: List[Dict], dates: List[str]) -> bool:
    """
    检查是否为异常股票（ST股票、新股、停牌、异常涨跌幅等）
    
    Args:
        code: 股票代码
        name: 股票名称
        daily_changes: 每日涨跌幅数据
        dates: 需要检查的日期列表
        
    Returns:
        True表示是异常股票
    """
    # 检查是否为ST股票
    if _is_st_stock(name):
        return True
    
    # 检查是否为新股（代码以9开头，如920045）
    if code.startswith('9'):
        return True
    
    # 检查是否有任意一天没数据（停牌）
    for date in dates:
        has_data = False
        for dc in daily_changes:
            if dc.get('date', '') == date:
                has_data = True
                break
        if not has_data:
            return True
    
    # 检查涨跌幅是否超过21%
    for dc in daily_changes:
        change_pct = dc.get('change_pct', 0)
        if isinstance(change_pct, (int, float)):
            if abs(change_pct) > 21:
                return True
        elif isinstance(change_pct, str):
            try:
                pct_value = float(change_pct.replace('%', '').replace('+', ''))
                if abs(pct_value) > 21:
                    return True
            except:
                pass
    
    return False


def _calculate_5day_cumulative_change(daily_changes: List[Dict]) -> float:
    """
    计算5日累计涨跌幅
    
    Args:
        daily_changes: 每日涨跌幅数据列表
        
    Returns:
        5日累计涨跌幅
    """
    valid_changes = []
    for dc in daily_changes:
        change_pct = dc.get('change_pct', 0)
        if isinstance(change_pct, (int, float)):
            valid_changes.append(change_pct)
        elif isinstance(change_pct, str):
            try:
                pct_value = float(change_pct.replace('%', '').replace('+', ''))
                valid_changes.append(pct_value)
            except:
                pass
    
    if valid_changes:
        return round(sum(valid_changes), 2)
    return 0.0


def _analyze_concept(concept_name: str, stock_codes: List[str], 
                    stock_list: pd.DataFrame, dates: List[str],
                    concept_map: Dict[str, List[str]]) -> Optional[Dict]:
    """
    分析单个概念（先过滤异常股票，再计算指标）
    
    Args:
        concept_name: 概念名称
        stock_codes: 该概念下的股票代码列表
        stock_list: 股票列表DataFrame
        dates: 需要分析的日期列表（最近5个交易日）
        concept_map: 概念板块映射
        
    Returns:
        概念分析结果字典，失败返回None
    """
    if not stock_codes:
        return None
    
    # 第一步：过滤异常股票（ST、新股、停牌、异常涨跌幅）
    valid_stocks = []
    for code in stock_codes:
        stock_info = stock_list[stock_list['code'] == code]
        if stock_info.empty:
            continue
        
        name = stock_info.iloc[0]['name']
        
        # 获取该股票最近5个交易日的涨跌幅数据
        daily_changes = []
        for date in dates:
            data = _get_stock_daily(code, date)
            if data is not None:
                daily_changes.append({
                    'date': date,
                    'change_pct': data.get('change_pct', 0)
                })
        
        # 检查是否为异常股票
        if not _is_abnormal_stock(code, name, daily_changes, dates):
            valid_stocks.append({
                'code': code,
                'name': name,
                'daily_changes': daily_changes,
                'cumulative_5day': _calculate_5day_cumulative_change(daily_changes)
            })
    
    if not valid_stocks:
        return None
    
    # 第二步：基于过滤后的有效股票计算指标
    valid_stock_count = len(valid_stocks)
    
    # 统计每个交易日的涨跌情况
    daily_stats = []
    
    for date in dates:
        up_count = 0
        down_count = 0
        flat_count = 0
        total_change = 0.0
        valid_data_count = 0
        
        for stock in valid_stocks:
            daily_changes = stock.get('daily_changes', [])
            for dc in daily_changes:
                if dc.get('date', '') == date:
                    change_pct = dc.get('change_pct', 0)
                    total_change += change_pct
                    valid_data_count += 1
                    
                    if change_pct > 0:
                        up_count += 1
                    elif change_pct < 0:
                        down_count += 1
                    else:
                        flat_count += 1
                    break
        
        if valid_data_count > 0:
            avg_change = total_change / valid_data_count
        else:
            avg_change = 0.0
        
        daily_stats.append({
            'date': date,
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'avg_change': round(avg_change, 2),
            'valid_data_count': valid_data_count
        })
    
    # 计算最近5个交易日的累计平均涨跌幅
    if daily_stats:
        total_avg_change = sum(d['avg_change'] for d in daily_stats)
        total_up = sum(d['up_count'] for d in daily_stats)
        total_down = sum(d['down_count'] for d in daily_stats)
    else:
        total_avg_change = 0.0
        total_up = 0
        total_down = 0
    
    # 准备股票详情（按5日累计涨跌幅排序）
    stock_details_sorted = sorted(valid_stocks, key=lambda x: x.get('cumulative_5day', 0), reverse=True)
    stock_details = []
    for stock in stock_details_sorted:
        stock_details.append({
            'code': stock['code'],
            'name': stock['name'],
            'daily_changes': stock['daily_changes'],
            'cumulative_5day': stock['cumulative_5day']
        })
    
    return {
        'concept_name': concept_name,
        'stock_count': valid_stock_count,
        'total_avg_change': round(total_avg_change, 2),
        'total_up_count': total_up,
        'total_down_count': total_down,
        'daily_stats': daily_stats,
        'stock_details': stock_details
    }


def _format_concept_result_for_chart(concept_results: List[Dict], dates: List[str]) -> str:
    """
    格式化概念分析结果为JSON（用于图表生成，包含完整数据）
    
    Args:
        concept_results: 概念分析结果列表
        dates: 日期列表
        
    Returns:
        JSON格式的字符串
    """
    if not concept_results:
        return JsonUtil.dumps({"message": "未找到概念数据"})
    
    # 按累计平均涨跌幅排序，取Top 20用于图表
    concept_results_sorted = sorted(
        concept_results,
        key=lambda x: x['total_avg_change'],
        reverse=True
    )[:20]
    
    # 为图表准备完整数据
    concepts = []
    for concept in concept_results_sorted:
        daily_data = []
        for stat in concept['daily_stats'][-5:]:
            date_str = datetime.strptime(stat['date'], "%Y%m%d").strftime("%Y-%m-%d")
            daily_data.append({
                "date": date_str,
                "avg_change": f"{stat['avg_change']:+.2f}%",
                "up_count": stat['up_count'],
                "down_count": stat['down_count'],
                "flat_count": stat['flat_count'],
                "valid_data_count": stat['valid_data_count']
            })
        
        # 添加股票详情
        stock_details = []
        for stock in concept.get('stock_details', []):
            stock_daily_changes = []
            for dc in stock.get('daily_changes', []):
                stock_daily_changes.append({
                    "date": dc['date'],
                    "change_pct": f"{dc['change_pct']:+.2f}%"
                })
            
            stock_details.append({
                "code": stock['code'],
                "name": stock['name'],
                "daily_changes": stock_daily_changes,
                "cumulative_5day": f"{stock.get('cumulative_5day', 0):+.2f}%"
            })
        
        concepts.append({
            "concept_name": concept['concept_name'],
            "stock_count": concept['stock_count'],
            "total_avg_change": f"{concept['total_avg_change']:+.2f}%",
            "total_up_count": concept['total_up_count'],
            "total_down_count": concept['total_down_count'],
            "daily_stats": daily_data,
            "stock_details": stock_details
        })
    
    result = {
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_concepts": len(concept_results),
        "analysis_days": len(dates),
        "concepts": concepts
    }
    
    return JsonUtil.dumps(result, indent=2)


def _format_concept_result(concept_results: List[Dict], dates: List[str],
                          compress: bool = False) -> str:
    """
    格式化概念分析结果为JSON（只返回Top 10概念的基本信息）
    
    Args:
        concept_results: 概念分析结果列表
        dates: 日期列表
        compress: 是否压缩数据（减少token消耗）
        
    Returns:
        JSON格式的字符串
    """
    if not concept_results:
        return JsonUtil.dumps({"message": "未找到概念数据"})
    
    # 按累计平均涨跌幅排序，取Top 10
    concept_results_sorted = sorted(
        concept_results,
        key=lambda x: x['total_avg_change'],
        reverse=True
    )[:10]
    
    # 只返回Top 10概念的基本信息
    concepts = []
    for concept in concept_results_sorted:
        concepts.append({
            "concept_name": concept['concept_name'],
            "total_avg_change": f"{concept['total_avg_change']:+.2f}%",
            "total_up_count": concept['total_up_count'],
            "total_down_count": concept['total_down_count']
        })
    
    result = {
        "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_concepts": len(concept_results),
        "analysis_days": len(dates),
        "concepts": concepts
    }
    
    return JsonUtil.dumps(result, indent=None if compress else 2)


# ==================== 图表生成 ====================

def generate_chart_from_results(result: Dict, save_path: Optional[Path] = None, top_n: int = 20) -> str:
    """
    从搜索结果生成图表
    
    Args:
        result: analyze_concepts 返回的结果字典
        save_path: 可选的文件保存路径
        top_n: 显示前N个概念，默认20
        
    Returns:
        图片文件路径字符串
    """
    concepts = result.get('concepts', [])
    if not concepts:
        logger.warning("未找到概念数据")
        return "未找到概念数据"
    
    total_concepts = result.get('total_concepts', len(concepts))
    query_time = result.get('query_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    analysis_days = result.get('analysis_days', 5)
    
    # 按累计平均涨跌幅排序，取前N个
    concepts_sorted = sorted(
        concepts,
        key=lambda x: float(x.get('total_avg_change', '0%').replace('%', '').replace('+', '')),
        reverse=True
    )[:top_n]
    
    # 准备数据
    concept_names = [c.get('concept_name', '')[:10] for c in concepts_sorted]
    total_avg_changes = [float(c.get('total_avg_change', '0%').replace('%', '').replace('+', '')) for c in concepts_sorted]
    stock_counts = [c.get('stock_count', 0) for c in concepts_sorted]
    up_counts = [c.get('total_up_count', 0) for c in concepts_sorted]
    down_counts = [c.get('total_down_count', 0) for c in concepts_sorted]
    
    # 创建更大的图表以容纳股票详情
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.3)
    
    fig.suptitle(f'大盘概念趋势分析 (Top {top_n})\n查询时间: {query_time} | 分析周期: 最近{analysis_days}个交易日 | 总概念数: {total_concepts}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 子图1: 累计平均涨跌幅
    ax1 = fig.add_subplot(gs[0])
    colors1 = ['green' if x >= 0 else 'red' for x in total_avg_changes]
    bars1 = ax1.barh(range(len(concept_names)), total_avg_changes, color=colors1, alpha=0.7)
    ax1.set_yticks(range(len(concept_names)))
    ax1.set_yticklabels(concept_names, fontsize=10)
    ax1.set_xlabel('累计平均涨跌幅 (%)', fontsize=11)
    ax1.set_title(f'概念累计平均涨跌幅 (Top {top_n})', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # 在柱子上显示数值
    for i, (bar, val) in enumerate(zip(bars1, total_avg_changes)):
        ax1.text(val, i, f'{val:+.2f}%', 
                va='center', ha='left' if val >= 0 else 'right', 
                fontsize=9, fontweight='bold')
    
    # 子图2: 涨跌统计
    ax2 = fig.add_subplot(gs[1])
    x = range(len(concept_names))
    width = 0.35
    bars2_up = ax2.bar([i - width/2 for i in x], up_counts, width, label='上涨次数', color='red', alpha=0.7)
    bars2_down = ax2.bar([i + width/2 for i in x], down_counts, width, label='下跌次数', color='green', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(concept_names, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('次数', fontsize=11)
    ax2.set_title(f'概念涨跌统计 (Top {top_n})', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar in bars2_up:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2_down:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 子图3: 股票详情表格
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    ax3.set_title(f'股票详情 (Top {top_n})', fontsize=12, fontweight='bold', pad=20)
    
    # 从daily_stats中获取日期列表用于表头
    date_headers = []
    date_map = {}  # 用于匹配股票数据的日期映射
    
    if concepts_sorted and concepts_sorted[0].get('daily_stats'):
        daily_stats = concepts_sorted[0]['daily_stats']
        for stat in daily_stats:
            date_str = stat.get('date', '')
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    mm_dd = date_obj.strftime('%m-%d')
                    date_headers.append(mm_dd)
                    # 创建日期映射：MM-DD -> YYYYMMDD
                    date_map[mm_dd] = date_obj.strftime('%Y%m%d')
                except:
                    date_headers.append(date_str)
    
    # 如果没有获取到日期，使用默认值
    if len(date_headers) < analysis_days:
        date_headers = [f'D{i+1}' for i in range(analysis_days)]
    
    # 为每个概念创建股票详情表格
    table_data = []
    row_colors = []
    
    for idx, concept in enumerate(concepts_sorted):
        concept_name = concept.get('concept_name', '')
        stock_details = concept.get('stock_details', [])
        
        # 添加概念标题行（合并概念名称和表头）
        table_data.append([f'{concept_name}', '五日累计'] + date_headers)
        row_colors.append('#4A90E2')
        
        # 添加股票详情（按5日累计涨跌幅排序，每个概念最多显示20只）
        for stock in stock_details[:20]:
            name = stock.get('name', '')
            code = stock.get('code', '')
            daily_changes = stock.get('daily_changes', [])
            cumulative_5day = stock.get('cumulative_5day', 0)
            
            # 合并名称和代码
            name_code = f'{name}({code})'
            
            # 添加五日累计涨跌幅
            if isinstance(cumulative_5day, str) and '%' in cumulative_5day:
                avg_value = cumulative_5day
            else:
                avg_value = f'{cumulative_5day:+.2f}%'
            
            # 按照表头日期顺序获取涨跌幅数据
            changes = []
            for header in date_headers:
                # 从date_map中获取对应的YYYYMMDD格式日期
                yyyymmdd = date_map.get(header, '')
                if yyyymmdd:
                    # 在daily_changes中查找对应日期的数据
                    found = False
                    for dc in daily_changes:
                        if dc.get('date', '') == yyyymmdd:
                            change_pct = dc.get('change_pct', 0)
                            if isinstance(change_pct, str):
                                changes.append(change_pct)
                            else:
                                changes.append(f'{change_pct:+.2f}%')
                            found = True
                            break
                    if not found:
                        changes.append('--')
                else:
                    changes.append('--')
            
            # 添加数据行：名称代码 + 五日累计 + 各日涨跌幅
            table_data.append([name_code, avg_value] + changes)
            row_colors.append('#FFFFFF' if len(table_data) % 2 == 0 else '#F0F0F0')
        
        # 添加空行分隔
        table_data.append([''] + [''] * (len(date_headers) + 1))
        row_colors.append('#FFFFFF')
    
    # 创建表格
    table = ax3.table(cellText=table_data, cellLoc='left', loc='upper left',
                      colWidths=[0.2] + [0.1] * len(date_headers) + [0.1])
    
    # 设置表格样式
    for (row, col), cell in table.get_celld().items():
        if row < len(table_data):
            cell.set_facecolor(row_colors[row])
            cell.set_fontsize(8)
            cell.set_edgecolor('#CCCCCC')
            cell.set_linewidth(0.5)
            
            # 标题行加粗（蓝色背景的行）
            if row_colors[row] == '#4A90E2':
                cell.set_fontsize(9)
                # 设置文本属性
                text = cell.get_text()
                text.set_fontweight('bold')
                text.set_color('white')
    
    # 调整表格位置
    table.scale(1, 1.5)
    
    # 保存图表
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = TOOLS_OUTPUT_DIR / f"concept_analysis_{timestamp}.png"
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"图表已保存到: {save_path}")
    return str(save_path)


# ==================== 主入口函数 ====================

def analyze_concepts(days: int = 5, market: str = "all",
                    include_kc: bool = False, include_cy: bool = False,
                    compress: bool = False,
                    save_chart: bool = False,
                    chart_path: Optional[Path] = None,
                    top_n: int = 20) -> Tuple[str, str]:
    """
    分析大盘概念趋势（主入口函数）
    
    Args:
        days: 分析天数，默认5天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        compress: 是否压缩JSON格式，默认False
        save_chart: 是否保存图表文件，默认False
        chart_path: 图表保存路径，如果为None则自动生成
        top_n: 显示前N个概念，默认20（仅影响图表显示，不影响JSON输出）
        
    Returns:
        (JSON格式的字符串结果, 图表文件路径字符串)
        
    Note:
        JSON输出只包含Top 10概念的基本信息：
        - concept_name: 概念名称
        - total_avg_change: 最近5个交易日累计涨跌幅
        - total_up_count: 对应上涨股票数量
        - total_down_count: 对应下跌股票数量
    """
    try:
        trading_days = _get_trading_days(days + 7)
        analysis_days = trading_days[:days]  # 取最新的days个交易日
        
        logger.info(f"分析日期: {analysis_days}")
        
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
        
        # 加载概念映射
        concept_map = JsonUtil.load(CONCEPT_MAP_FILE) or {}
        
        # 构建概念到股票的映射
        concept_to_stocks = defaultdict(list)
        for code, concepts in concept_map.items():
            filtered_concepts = _filter_concepts(concepts)
            for concept in filtered_concepts:
                concept_to_stocks[concept].append(code)
        
        # 过滤掉股票列表中没有的股票
        valid_codes = set(stock_list['code'].tolist())
        for concept in list(concept_to_stocks.keys()):
            concept_to_stocks[concept] = [
                code for code in concept_to_stocks[concept] 
                if code in valid_codes
            ]
        
        logger.info(f"分析 {len(concept_to_stocks)} 个概念...")
        
        # 分析每个概念
        concept_results = []
        for concept_name, stock_codes in tqdm(concept_to_stocks.items(), desc="分析概念", unit="个"):
            result = _analyze_concept(
                concept_name, stock_codes, stock_list, analysis_days, concept_map
            )
            if result:
                concept_results.append(result)
        
        logger.info(f"找到 {len(concept_results)} 个有效概念")
        
        # 格式化JSON结果（简化版，只返回Top 10基本信息）
        result_json = _format_concept_result(concept_results, analysis_days, compress=compress)
        result_dict = JsonUtil.loads(result_json) or {}
        
        # 保存JSON文件
        json_output_path = None
        if result_dict:
            json_filename = f"concept_analysis_{datetime.now().strftime('%Y%m%d')}_{days}days_{market}_kc{include_kc}_cy{include_cy}.json"
            json_output_path = TOOLS_OUTPUT_DIR / json_filename
            JsonUtil.save(result_dict, json_output_path)
            result_dict["json_output_path"] = str(json_output_path)
        
        # 生成图表（使用完整数据）
        chart_output_path = None
        if save_chart and concept_results:
            if chart_path is None:
                chart_path = TOOLS_OUTPUT_DIR / f"concept_analysis_{datetime.now().strftime('%Y%m%d')}_{days}days_{market}_kc{include_kc}_cy{include_cy}.png"
            
            # 为图表生成准备完整数据
            chart_data = _format_concept_result_for_chart(concept_results, analysis_days)
            chart_result_dict = JsonUtil.loads(chart_data) or {}
            chart_output_path = generate_chart_from_results(chart_result_dict, save_path=chart_path, top_n=top_n)
            result_dict["chart_output_path"] = chart_output_path
        else:
            chart_output_path = "未找到概念数据"
        
        return JsonUtil.dumps(result_dict), chart_output_path
        
    except Exception as e:
        logger.error(f"分析概念数据失败: {e}")
        error_result = JsonUtil.dumps({"message": f"分析失败: {str(e)}"})
        return error_result, "分析失败"


# ==================== 兼容性函数（保持向后兼容）====================

def analyze_concepts_simple(days: int = 5, market: str = "all",
                           include_kc: bool = False, include_cy: bool = False,
                           compress: bool = False) -> str:
    """
    分析大盘概念趋势（兼容性函数，仅返回JSON）
    
    Args:
        days: 分析天数，默认5天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        compress: 是否压缩JSON格式，默认False
        
    Returns:
        JSON格式的字符串结果
    """
    result_json, _ = analyze_concepts(
        days=days, market=market,
        include_kc=include_kc, include_cy=include_cy,
        compress=compress
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
        "name": "analyze_concepts",
        "description": "分析大盘概念趋势（基于本地缓存数据）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "分析天数，默认5天",
                    "default": 5
                },
                "market": {
                    "type": "string",
                    "description": "市场类型：'all'全市场，'sh'上海，'sz'深圳",
                    "enum": ["all", "sh", "sz"],
                    "default": "all"
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
        days = arguments.get("days", 5)
        market = arguments.get("market", "all")
        include_kc = arguments.get("include_kc", False)
        include_cy = arguments.get("include_cy", False)
        compress = arguments.get("compress", False)
        
        result_json, _ = analyze_concepts(
            days=days, market=market,
            include_kc=include_kc, include_cy=include_cy,
            compress=compress
        )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": result_json
                }
            ],
            "isError": False
        }
    except Exception as e:
        logger.error(f"MCP调用失败: {e}")
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
    try:
        print("=" * 70)
        print("大盘概念趋势分析 - 开始分析")
        print("=" * 70)
        
        result_json, chart_content = analyze_concepts(
            days=5,
            market="all",
            include_kc=False,
            include_cy=False,
            compress=False,
            save_chart=True,
            top_n=10
        )
        
        result = JsonUtil.loads(result_json) or {}
        
        # 打印查询结果消息
        if result.get("concepts"):
            print(f"\n✅ 分析完成，找到 {len(result.get('concepts', []))} 个热门概念")
            print(f"📊 分析周期: {result.get('analysis_days', 5)} 个交易日")
            print(f"⏰ 查询时间: {result.get('query_time', 'N/A')}")
            print(f"📈 总概念数: {result.get('total_concepts', 0)}")
            
            # 打印前10个概念
            print("\n🔥 热门概念TOP10:")
            for i, concept in enumerate(result.get('concepts', [])[:10], 1):
                print(f"   {i}. {concept.get('concept_name', 'N/A')} - "
                      f"{concept.get('total_avg_change', '0%')} "
                      f"(上涨:{concept.get('total_up_count', 0)} "
                      f"下跌:{concept.get('total_down_count', 0)})")
            
            # 打印图表文件路径
            print(f"\n📊 图表已保存到: {chart_content}")
            
            # 输出JSON结果到文件
            json_output_path = TOOLS_OUTPUT_DIR / f"concept_analysis_{datetime.now().strftime('%Y%m%d')}.json"
            print(f"📄 准备保存JSON结果到: {json_output_path}")
            print(f"📄 result类型: {type(result)}, 是否为空: {not result}")
            
            save_success = JsonUtil.save(result, json_output_path)
            if save_success:
                print(f"📄 JSON结果已保存到: {json_output_path}")
                print(f"📄 文件是否存在: {json_output_path.exists()}")
            else:
                print(f"❌ JSON保存失败: {json_output_path}")
        else:
            print(f"\n⚠️  {result.get('message', '未找到概念数据')}")
        
        print("\n" + "=" * 70)
        print("分析完成")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
