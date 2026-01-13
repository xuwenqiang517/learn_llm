import pandas as pd
from typing import List, Optional


def analyze_volume_price_pattern(
    df: pd.DataFrame,
    price_change_col: str = "涨跌幅",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None,
    days: int = 3
) -> List[str]:
    """
    分析量价形态
    
    Args:
        df: 历史数据DataFrame，已按日期排序（从新到旧）
        price_change_col: 涨跌幅列名
        volume_col: 成交量列名
        turnover_col: 换手率列名（可选）
        days: 分析天数，默认3天
    
    Returns:
        量价形态列表，如["量价齐升", "价涨量缩"]
    """
    if len(df) < days:
        return []
    
    patterns = []
    recent_df = df.head(days)
    
    price_changes = recent_df[price_change_col].values
    volumes = recent_df[volume_col].values
    
    turnover_rates = None
    if turnover_col and turnover_col in df.columns:
        turnover_rates = recent_df[turnover_col].values
    
    # 判断是否连续上涨
    is_rising = all(pc > 0 for pc in price_changes)
    
    if not is_rising:
        return patterns
    
    # 计算成交量变化趋势
    volume_trend = []
    for i in range(len(volumes) - 1):
        if volumes[i] > volumes[i + 1]:
            volume_trend.append("increase")
        elif volumes[i] < volumes[i + 1]:
            volume_trend.append("decrease")
        else:
            volume_trend.append("stable")
    
    # 量价齐升：价格上涨且成交量增加
    if all(t == "increase" for t in volume_trend):
        patterns.append("量价齐升")
    
    # 价涨量缩：价格上涨但成交量减少
    elif all(t == "decrease" for t in volume_trend):
        patterns.append("价涨量缩")
    
    # 量价平稳：价格上涨但成交量平稳
    elif all(t == "stable" for t in volume_trend):
        patterns.append("量价平稳")
    
    # 量价震荡：成交量变化不一致
    else:
        patterns.append("量价震荡")
    
    # 如果有换手率数据，分析换手率趋势
    if turnover_rates is not None:
        # 分析换手率变化趋势
        turnover_trend = []
        for i in range(len(turnover_rates) - 1):
            if turnover_rates[i] > turnover_rates[i + 1]:
                turnover_trend.append("increase")
            elif turnover_rates[i] < turnover_rates[i + 1]:
                turnover_trend.append("decrease")
            else:
                turnover_trend.append("stable")
        
        # 换手率递增
        if all(t == "increase" for t in turnover_trend):
            patterns.append("换手递增")
        # 换手率递减
        elif all(t == "decrease" for t in turnover_trend):
            patterns.append("换手递减")
        # 换手率平稳
        elif all(t == "stable" for t in turnover_trend):
            patterns.append("换手平稳")
        # 换手率震荡
        else:
            patterns.append("换手震荡")
        
        # 换手率水平分类
        avg_turnover = turnover_rates.mean()
        if avg_turnover > 10:
            patterns.append("高换手")
        elif avg_turnover > 5:
            patterns.append("中换手")
        else:
            patterns.append("低换手")
    
    # 判断是否推荐
    is_recommended = False
    if "量价齐升" in patterns:
        if "换手递增" in patterns or "高换手" in patterns:
            is_recommended = True
    
    if is_recommended:
        patterns.append("【推荐】")
    
    return patterns


def get_volume_price_summary(
    df: pd.DataFrame,
    price_change_col: str = "涨跌幅",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None,
    days: int = 3
) -> str:
    """
    获取量价形态摘要
    
    Args:
        df: 历史数据DataFrame，已按日期排序（从新到旧）
        price_change_col: 涨跌幅列名
        volume_col: 成交量列名
        turnover_col: 换手率列名（可选）
        days: 分析天数，默认3天
    
    Returns:
        量价形态摘要字符串
    """
    patterns = analyze_volume_price_pattern(df, price_change_col, volume_col, turnover_col, days)
    
    if not patterns:
        return "无明显量价形态"
    
    return "、".join(patterns)
