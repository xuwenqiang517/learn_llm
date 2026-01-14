import pandas as pd
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class TechnicalScore:
    score: int
    ma_score: int
    volume_score: int
    macd_score: int
    rsi_score: int
    boll_score: int
    atr_score: int
    signals: List[str]
    summary: str
    raw_data: Dict = field(default_factory=dict)


def analyze_ma_pattern(
    df: pd.DataFrame,
    close_col: str = "收盘",
    ma5_col: str = "MA5",
    ma10_col: str = "MA10",
    ma20_col: str = "MA20"
) -> Dict:
    if len(df) < 5:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    signals = []
    score = 0
    
    row_idx = None
    for i in range(len(df)):
        ma5 = df.iloc[i].get(ma5_col, 0)
        ma10 = df.iloc[i].get(ma10_col, 0)
        ma20 = df.iloc[i].get(ma20_col, 0)
        if pd.notna(ma5) and pd.notna(ma10) and pd.notna(ma20) and ma5 > 0 and ma10 > 0 and ma20 > 0:
            row_idx = i
            break
    
    if row_idx is None:
        return {"score": 0, "signals": ["均线数据不足"], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[row_idx]
    close = latest.get(close_col, 0)
    ma5 = latest.get(ma5_col, 0)
    ma10 = latest.get(ma10_col, 0)
    ma20 = latest.get(ma20_col, 0)
    
    prev_idx = min(row_idx + 4, len(df) - 1)
    ma5_prev = df.iloc[prev_idx].get(ma5_col, 0) if pd.notna(df.iloc[prev_idx].get(ma5_col, 0)) else 0
    ma10_prev = df.iloc[min(row_idx + 9, len(df) - 1)].get(ma10_col, 0) if pd.notna(df.iloc[min(row_idx + 9, len(df) - 1)].get(ma10_col, 0)) else 0
    ma20_prev = df.iloc[min(row_idx + 19, len(df) - 1)].get(ma20_col, 0) if pd.notna(df.iloc[min(row_idx + 19, len(df) - 1)].get(ma20_col, 0)) else 0
    
    close_ma5_gap = ((close - ma5) / ma5 * 100) if ma5 > 0 else 0
    ma5_ma10_gap = ((ma5 - ma10) / ma10 * 100) if ma10 > 0 else 0
    ma10_ma20_gap = ((ma10 - ma20) / ma20 * 100) if ma20 > 0 else 0
    ma5_trend = ((ma5 - ma5_prev) / ma5_prev * 100) if ma5_prev > 0 else 0
    ma10_trend = ((ma10 - ma10_prev) / ma10_prev * 100) if ma10_prev > 0 else 0
    ma20_trend = ((ma20 - ma20_prev) / ma20_prev * 100) if ma20_prev > 0 else 0
    
    signals.append(f"收盘价={close:.2f}, MA5={ma5:.2f}, MA10={ma10:.2f}, MA20={ma20:.2f}")
    signals.append(f"价格-MA5差值={close_ma5_gap:+.2f}%, MA5-MA10差值={ma5_ma10_gap:+.2f}%, MA10-MA20差值={ma10_ma20_gap:+.2f}%")
    signals.append(f"MA5变化={ma5_trend:+.2f}%, MA10变化={ma10_trend:+.2f}%, MA20变化={ma20_trend:+.2f}%")
    
    if ma5 > ma10 > ma20:
        signals.append("均线多头排列(MA5>MA10>MA20)")
        score += 10
    elif ma5 < ma10 < ma20:
        signals.append("均线空头排列(MA5<MA10<MA20)")
        score -= 5
    elif ma5 > ma10 and ma10 < ma20:
        signals.append("均线震荡(MA5>MA10<MA20)")
        score += 2
    else:
        signals.append("均线纠缠(无明显趋势)")
        score += 0
    
    if close > ma5:
        signals.append(f"价格在MA5上方({close_ma5_gap:+.2f}%)")
        score += 3
        if close > ma10:
            signals.append(f"价格在MA10上方")
            score += 3
            if close > ma20:
                signals.append(f"价格在MA20上方")
                score += 4
    elif close < ma5:
        signals.append(f"价格在MA5下方({close_ma5_gap:+.2f}%)")
        score -= 2
        if close < ma20:
            signals.append(f"价格在MA20下方")
            score -= 3
    
    score = max(0, min(25, score))
    
    raw_data = {
        "close": close,
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "close_ma5_gap_pct": close_ma5_gap,
        "ma5_ma10_gap_pct": ma5_ma10_gap,
        "ma10_ma20_gap_pct": ma10_ma20_gap,
        "ma5_trend_pct": ma5_trend,
        "ma10_trend_pct": ma10_trend,
        "ma20_trend_pct": ma20_trend
    }
    
    summary = f"MA5={ma5:.2f}({ma5_trend:+.1f}%), MA10={ma10:.2f}({ma10_trend:+.1f}%), MA20={ma20:.2f}({ma20_trend:+.1f}%)"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_volume_ma_pattern(
    df: pd.DataFrame,
    volume_col: str = "成交量",
    vol_ma5_col: str = "VOL_MA5",
    vol_ma10_col: str = "VOL_MA10",
    vol_ma20_col: str = "VOL_MA20"
) -> Dict:
    if len(df) < 5:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    signals = []
    score = 0
    
    row_idx = None
    for i in range(len(df)):
        vol_ma5 = df.iloc[i].get(vol_ma5_col, 0)
        vol_ma10 = df.iloc[i].get(vol_ma10_col, 0)
        vol_ma20 = df.iloc[i].get(vol_ma20_col, 0)
        if pd.notna(vol_ma5) and pd.notna(vol_ma10) and pd.notna(vol_ma20) and vol_ma5 > 0 and vol_ma10 > 0 and vol_ma20 > 0:
            row_idx = i
            break
    
    if row_idx is None:
        return {"score": 0, "signals": ["量能均线数据不足"], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[row_idx]
    volume = latest.get(volume_col, 0)
    vol_ma5 = latest.get(vol_ma5_col, 0)
    vol_ma10 = latest.get(vol_ma10_col, 0)
    vol_ma20 = latest.get(vol_ma20_col, 0)
    
    vol_ma5_prev = df.iloc[min(row_idx + 4, len(df) - 1)].get(vol_ma5_col, 0) if pd.notna(df.iloc[min(row_idx + 4, len(df) - 1)].get(vol_ma5_col, 0)) else 0
    vol_ma10_prev = df.iloc[min(row_idx + 9, len(df) - 1)].get(vol_ma10_col, 0) if pd.notna(df.iloc[min(row_idx + 9, len(df) - 1)].get(vol_ma10_col, 0)) else 0
    
    volume_ma5_ratio = volume / vol_ma5 if vol_ma5 > 0 else 0
    volume_ma10_ratio = volume / vol_ma10 if vol_ma10 > 0 else 0
    vol_ma5_trend = ((vol_ma5 - vol_ma5_prev) / vol_ma5_prev * 100) if vol_ma5_prev > 0 else 0
    vol_ma10_trend = ((vol_ma10 - vol_ma10_prev) / vol_ma10_prev * 100) if vol_ma10_prev > 0 else 0
    
    signals.append(f"成交量={volume/1e6:.2f}M, VMA5={vol_ma5/1e6:.2f}M, VMA10={vol_ma10/1e6:.2f}M, VMA20={vol_ma20/1e6:.2f}M")
    signals.append(f"量能倍数: 相比VMA5={volume_ma5_ratio:.2f}x, 相比VMA10={volume_ma10_ratio:.2f}x")
    signals.append(f"VMA5变化={vol_ma5_trend:+.2f}%, VMA10变化={vol_ma10_trend:+.2f}%")
    
    if volume > vol_ma5:
        signals.append(f"量能在MA5上方({volume_ma5_ratio:.2f}x)")
        score += 3
        if volume > vol_ma10:
            signals.append(f"量能在MA10上方({volume_ma10_ratio:.2f}x)")
            score += 3
            if volume > vol_ma20:
                signals.append("量能在MA20上方")
                score += 4
    elif volume < vol_ma5:
        signals.append(f"量能在MA5下方({volume_ma5_ratio:.2f}x)")
        score -= 2
    
    if vol_ma5 > vol_ma10 > vol_ma20:
        signals.append("成交量均线多头排列(VMA5>VMA10>VMA20)")
        score += 3
    elif vol_ma5 < vol_ma10 < vol_ma20:
        signals.append("成交量均线空头排列(VMA5<VMA10<VMA20)")
        score -= 2
    
    if volume_ma5_ratio > 2:
        signals.append(f"明显放量({volume_ma5_ratio:.1f}倍)")
        score += 2
    elif volume_ma5_ratio > 1.5:
        signals.append(f"温和放量({volume_ma5_ratio:.1f}倍)")
        score += 1
    elif volume_ma5_ratio < 0.5:
        signals.append(f"明显缩量({volume_ma5_ratio:.1f}倍)")
        score -= 1
    
    score = max(0, min(15, score))
    
    raw_data = {
        "volume": volume,
        "vol_ma5": vol_ma5,
        "vol_ma10": vol_ma10,
        "vol_ma20": vol_ma20,
        "volume_ma5_ratio": volume_ma5_ratio,
        "volume_ma10_ratio": volume_ma10_ratio,
        "vol_ma5_trend_pct": vol_ma5_trend,
        "vol_ma10_trend_pct": vol_ma10_trend
    }
    
    summary = f"量={volume/1e6:.1f}M, VMA5={vol_ma5/1e6:.1f}M, VMA10={vol_ma10/1e6:.1f}M"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_macd_pattern(
    df: pd.DataFrame,
    dif_col: str = "DIF",
    dea_col: str = "DEA",
    macd_col: str = "MACD"
) -> Dict:
    if len(df) < 10:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[0]
    signals = []
    score = 0
    
    dif = latest.get(dif_col, 0)
    dea = latest.get(dea_col, 0)
    macd = latest.get(macd_col, 0)
    
    prev_dif = df.iloc[1].get(dif_col, 0) if len(df) >= 2 else 0
    prev_dea = df.iloc[1].get(dea_col, 0) if len(df) >= 2 else 0
    prev_macd = df.iloc[1].get(macd_col, 0) if len(df) >= 2 else 0
    
    dif_dea_diff = dif - dea
    dif_dea_diff_pct = (dif_dea_diff / abs(dea) * 100) if dea != 0 else 0
    macd_change = macd - prev_macd if prev_macd != 0 else 0
    macd_change_pct = (macd_change / abs(prev_macd) * 100) if prev_macd != 0 else 0
    
    dif_trend = dif - prev_dif
    dea_trend = dea - prev_dea
    
    signals.append(f"MACD指标: DIF={dif:.4f}, DEA={dea:.4f}, MACD={macd:.4f}")
    signals.append(f"DIF-DEA差值={dif_dea_diff:.4f}({dif_dea_diff_pct:+.2f}%)")
    signals.append(f"MACD柱变化={macd_change:+.4f}({macd_change_pct:+.1f}%), DIF变化={dif_trend:+.4f}, DEA变化={dea_trend:+.4f}")
    
    if dif > dea and prev_dif <= prev_dea:
        signals.append("MACD金叉(DIF上穿DEA)")
        score += 8
    elif dif < dea and prev_dif >= prev_dea:
        signals.append("MACD死叉(DIF下穿DEA)")
        score -= 8
    
    if macd > 0:
        signals.append(f"MACD红柱({macd:.4f})")
        score += 3
        if macd > prev_macd:
            signals.append("红柱放大")
            score += 2
        elif macd < prev_macd:
            signals.append("红柱缩短")
            score -= 1
    elif macd < 0:
        signals.append(f"MACD绿柱({macd:.4f})")
        score -= 3
        if macd < prev_macd:
            signals.append("绿柱缩短(转强)")
            score += 1
        elif macd > prev_macd:
            signals.append("绿柱放大(转弱)")
            score -= 2
    
    if dif > dea:
        signals.append("DIF在DEA上方")
        score += 3
    else:
        signals.append("DIF在DEA下方")
        score -= 3
    
    if dif > 0 and dea > 0:
        signals.append("MACD在零轴上方")
        score += 2
    elif dif < 0 and dea < 0:
        signals.append("MACD在零轴下方")
        score -= 2
    
    score = max(0, min(20, score))
    
    raw_data = {
        "dif": dif,
        "dea": dea,
        "macd": macd,
        "dif_dea_diff": dif_dea_diff,
        "dif_dea_diff_pct": dif_dea_diff_pct,
        "macd_change": macd_change,
        "dif_trend": dif_trend,
        "dea_trend": dea_trend
    }
    
    summary = f"DIF={dif:.3f}, DEA={dea:.3f}, MACD={macd:.3f}"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_rsi_pattern(
    df: pd.DataFrame,
    rsi_col: str = "RSI"
) -> Dict:
    if len(df) < 14:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[0]
    signals = []
    score = 0
    
    rsi = latest.get(rsi_col, 50)
    
    rsi_values = df.head(5)[rsi_col].values if len(df) >= 5 else [rsi] * 5
    rsi_prev3 = df.head(3)[rsi_col].values if len(df) >= 3 else [rsi] * 3
    rsi_5d_ago = df.iloc[4].get(rsi_col, rsi) if len(df) >= 5 else rsi
    
    rsi_trend = rsi - rsi_5d_ago
    rsi_3d_avg = sum(rsi_prev3) / len(rsi_prev3)
    rsi_momentum = rsi - rsi_3d_avg
    
    signals.append(f"RSI(14)={rsi:.2f}")
    signals.append(f"RSI5日变化={rsi_trend:+.2f}, RSI3日均值={rsi_3d_avg:.2f}, RSI动量={rsi_momentum:+.2f}")
    
    if rsi >= 80:
        signals.append(f"RSI超买区域({rsi:.1f})")
        score -= 5
    elif rsi >= 70:
        signals.append(f"RSI偏强区域({rsi:.1f})")
        score -= 2
    elif rsi <= 20:
        signals.append(f"RSI超卖区域({rsi:.1f})")
        score += 5
    elif rsi <= 30:
        signals.append(f"RSI偏弱区域({rsi:.1f})")
        score += 2
    else:
        signals.append(f"RSI中性区域({rsi:.1f})")
        score += 3
    
    if 40 <= rsi <= 60:
        signals.append("RSI在40-60中性区间震荡")
        score += 2
    
    if rsi_trend > 10:
        signals.append("RSI上升趋势明显")
        score += 2
    elif rsi_trend > 5:
        signals.append("RSI温和上升")
        score += 1
    elif rsi_trend < -10:
        signals.append("RSI下降趋势明显")
        score -= 2
    elif rsi_trend < -5:
        signals.append("RSI温和下降")
        score -= 1
    
    if rsi > 70 and rsi_trend < 0:
        signals.append("RSI从超买回落")
        score -= 2
    elif rsi < 30 and rsi_trend > 0:
        signals.append("RSI从超卖回升")
        score += 2
    
    score = max(0, min(15, score))
    
    raw_data = {
        "rsi": rsi,
        "rsi_trend": rsi_trend,
        "rsi_3d_avg": rsi_3d_avg,
        "rsi_momentum": rsi_momentum,
        "rsi_5d_ago": rsi_5d_ago
    }
    
    summary = f"RSI={rsi:.1f}({rsi_trend:+.1f})"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_boll_pattern(
    df: pd.DataFrame,
    close_col: str = "收盘",
    boll_mid_col: str = "BOLL_MID",
    boll_up_col: str = "BOLL_UP",
    boll_low_col: str = "BOLL_LOW"
) -> Dict:
    if len(df) < 20:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[0]
    signals = []
    score = 0
    
    close = latest.get(close_col, 0)
    boll_mid = latest.get(boll_mid_col, 0)
    boll_up = latest.get(boll_up_col, 0)
    boll_low = latest.get(boll_low_col, 0)
    
    prev_boll_up = df.iloc[9].get(boll_up_col, 0) if len(df) >= 10 else 0
    prev_boll_low = df.iloc[9].get(boll_low_col, 0) if len(df) >= 10 else 0
    
    boll_width = boll_up - boll_low
    prev_boll_width = prev_boll_up - prev_boll_low
    
    close_to_mid = ((close - boll_mid) / boll_mid * 100) if boll_mid > 0 else 0
    close_to_up = ((close - boll_up) / boll_up * 100) if boll_up > 0 else 0
    close_to_low = ((close - boll_low) / boll_low * 100) if boll_low > 0 else 0
    position_ratio = (close - boll_low) / boll_width if boll_width > 0 else 0.5
    
    width_change_pct = ((boll_width - prev_boll_width) / prev_boll_width * 100) if prev_boll_width > 0 else 0
    
    signals.append(f"布林带: 上轨={boll_up:.2f}, 中轨={boll_mid:.2f}, 下轨={boll_low:.2f}, 宽度={boll_width:.2f}")
    signals.append(f"价格相对位置: 距中轨{close_to_mid:+.2f}%, 距上轨{close_to_up:+.2f}%, 距下轨{close_to_low:+.2f}%")
    signals.append(f"价格位置比例={position_ratio:.2%}(0=下轨, 0.5=中轨, 1=上轨)")
    signals.append(f"布林带宽度变化={width_change_pct:+.2f}%")
    
    if position_ratio > 0.8:
        signals.append(f"价格接近上轨({position_ratio*100:.0f}%)")
        score += 2
        if position_ratio >= 1:
            signals.append("价格突破上轨")
            score += 3
    elif position_ratio < 0.2:
        signals.append(f"价格接近下轨({position_ratio*100:.0f}%)")
        score += 2
        if position_ratio <= 0:
            signals.append("价格跌破下轨")
            score -= 3
    else:
        signals.append(f"价格在中轨附近({position_ratio*100:.0f}%)")
        score += 3
    
    if close > boll_mid:
        signals.append("价格在中轨上方")
        score += 2
        if close > boll_up:
            signals.append("价格在上轨外")
            score += 1
    elif close < boll_mid:
        signals.append("价格在中轨下方")
        score -= 2
        if close < boll_low:
            signals.append("价格在下轨外")
            score -= 1
    
    if width_change_pct < -20:
        signals.append("布林带收窄(波动可能放大)")
        score += 2
    elif width_change_pct > 20:
        signals.append("布林带扩张(波动加大)")
        score += 1
    
    score = max(0, min(15, score))
    
    raw_data = {
        "boll_up": boll_up,
        "boll_mid": boll_mid,
        "boll_low": boll_low,
        "boll_width": boll_width,
        "position_ratio": position_ratio,
        "close_to_mid_pct": close_to_mid,
        "close_to_up_pct": close_to_up,
        "close_to_low_pct": close_to_low,
        "width_change_pct": width_change_pct
    }
    
    summary = f"中轨={boll_mid:.2f}, 上轨={boll_up:.2f}, 下轨={boll_low:.2f}, 位置={position_ratio:.0%}"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_atr_pattern(
    df: pd.DataFrame,
    atr_col: str = "ATR"
) -> Dict:
    if len(df) < 14:
        return {"score": 0, "signals": [], "summary": "数据不足", "raw_data": {}}
    
    latest = df.iloc[0]
    signals = []
    score = 0
    
    atr = latest.get(atr_col, 0)
    
    if atr == 0:
        return {"score": 0, "signals": [], "summary": "ATR数据异常", "raw_data": {}}
    
    atr_values = df.head(20)[atr_col].values
    atr_mean = sum(atr_values) / len(atr_values)
    atr_5d_ago = df.iloc[4].get(atr_col, atr) if len(df) >= 5 else atr
    
    atr_ratio = atr / atr_mean if atr_mean > 0 else 1
    atr_trend = atr - atr_5d_ago
    atr_trend_pct = (atr_trend / atr_5d_ago * 100) if atr_5d_ago > 0 else 0
    atr_std = (sum((x - atr_mean) ** 2 for x in atr_values) / len(atr_values)) ** 0.5 if len(atr_values) > 0 else 0
    
    signals.append(f"ATR(14)={atr:.2f}")
    signals.append(f"ATR20日均值={atr_mean:.2f}, ATR当前/均值={atr_ratio:.2f}")
    signals.append(f"ATR5日变化={atr_trend:+.2f}({atr_trend_pct:+.1f}%)")
    signals.append(f"ATR20日标准差={atr_std:.2f}")
    
    if atr_ratio > 1.5:
        signals.append(f"高波动期(ATR={atr:.2f}, 是均值的{atr_ratio:.1f}倍)")
        score += 2
    elif atr_ratio > 1.2:
        signals.append(f"波动放大(ATR={atr:.2f})")
        score += 1
    elif atr_ratio < 0.7:
        signals.append(f"低波动期(ATR={atr:.2f}, 是均值的{atr_ratio:.1f}倍)")
        score += 2
    elif atr_ratio < 0.9:
        signals.append(f"波动收窄(ATR={atr:.2f})")
        score += 1
    
    if atr_trend > 0:
        signals.append("ATR上升(波动可能扩大)")
        score += 2
    elif atr_trend < 0:
        signals.append("ATR下降(波动可能收窄)")
        score += 1
    
    score = max(0, min(10, score))
    
    raw_data = {
        "atr": atr,
        "atr_mean": atr_mean,
        "atr_ratio": atr_ratio,
        "atr_trend": atr_trend,
        "atr_trend_pct": atr_trend_pct,
        "atr_std": atr_std
    }
    
    summary = f"ATR={atr:.2f}({atr_ratio:.1f}×均值, {atr_trend:+.1f}%)"
    
    return {"score": score, "signals": signals, "summary": summary, "raw_data": raw_data}


def analyze_technical_pattern(
    df: pd.DataFrame,
    close_col: str = "收盘",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None
) -> TechnicalScore:
    all_signals = []
    ma_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    volume_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    macd_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    rsi_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    boll_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    atr_result = {"score": 0, "signals": [], "summary": "", "raw_data": {}}
    
    try:
        ma_result = analyze_ma_pattern(df, close_col)
        all_signals.extend(ma_result["signals"])
    except Exception as e:
        pass
    
    try:
        volume_result = analyze_volume_ma_pattern(df, volume_col)
        all_signals.extend(volume_result["signals"])
    except Exception as e:
        pass
    
    try:
        macd_result = analyze_macd_pattern(df)
        all_signals.extend(macd_result["signals"])
    except Exception as e:
        pass
    
    try:
        rsi_result = analyze_rsi_pattern(df)
        all_signals.extend(rsi_result["signals"])
    except Exception as e:
        pass
    
    try:
        boll_result = analyze_boll_pattern(df, close_col)
        all_signals.extend(boll_result["signals"])
    except Exception as e:
        pass
    
    try:
        atr_result = analyze_atr_pattern(df)
        all_signals.extend(atr_result["signals"])
    except Exception as e:
        pass
    
    total_score = (
        ma_result["score"] +
        volume_result["score"] +
        macd_result["score"] +
        rsi_result["score"] +
        boll_result["score"] +
        atr_result["score"]
    )
    
    if total_score >= 70:
        summary = "技术面强势"
    elif total_score >= 50:
        summary = "技术面偏强"
    elif total_score >= 30:
        summary = "技术面中性"
    elif total_score >= 15:
        summary = "技术面偏弱"
    else:
        summary = "技术面弱势"
    
    is_recommended = (
        total_score >= 50 and
        any("均线多头排列" in s for s in all_signals) and
        any("MACD金叉" in s for s in all_signals)
    )
    
    if is_recommended:
        summary += "【关注】"
    
    all_raw_data = {}
    all_raw_data.update(ma_result.get("raw_data", {}))
    all_raw_data.update(volume_result.get("raw_data", {}))
    all_raw_data.update(macd_result.get("raw_data", {}))
    all_raw_data.update(rsi_result.get("raw_data", {}))
    all_raw_data.update(boll_result.get("raw_data", {}))
    all_raw_data.update(atr_result.get("raw_data", {}))
    
    return TechnicalScore(
        score=total_score,
        ma_score=ma_result["score"],
        volume_score=volume_result["score"],
        macd_score=macd_result["score"],
        rsi_score=rsi_result["score"],
        boll_score=boll_result["score"],
        atr_score=atr_result["score"],
        signals=all_signals,
        summary=summary,
        raw_data=all_raw_data
    )


def get_technical_summary(
    df: pd.DataFrame,
    close_col: str = "收盘",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None
) -> str:
    result = analyze_technical_pattern(df, close_col, volume_col, turnover_col)
    
    ma_summary = ma_result["summary"] if result.ma_score > 0 else "数据不足"
    volume_summary = volume_result["summary"] if result.volume_score > 0 else "数据不足"
    macd_summary = macd_result["summary"] if result.macd_score > 0 else "数据不足"
    rsi_summary = rsi_result["summary"] if result.rsi_score > 0 else "数据不足"
    boll_summary = boll_result["summary"] if result.boll_score > 0 else "数据不足"
    atr_summary = atr_result["summary"] if result.atr_score > 0 else "数据不足"
    
    lines = [
        f"综合评分: {result.score}/100",
        f"均线(MA): {result.ma_score}/25 - {ma_summary}",
        f"成交量: {result.volume_score}/15 - {volume_summary}",
        f"MACD: {result.macd_score}/20 - {macd_summary}",
        f"RSI: {result.rsi_score}/15 - {rsi_summary}",
        f"布林带: {result.boll_score}/15 - {boll_summary}",
        f"ATR: {result.atr_score}/10 - {atr_summary}",
        "",
        "=== 详细数据 ===",
    ]
    
    for signal in result.signals:
        lines.append(f"- {signal}")
    
    lines.extend([
        "",
        f"综合评价: {result.summary}"
    ])
    
    return "\n".join(lines)


def get_technical_signals(
    df: pd.DataFrame,
    close_col: str = "收盘",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None
) -> Dict:
    result = analyze_technical_pattern(df, close_col, volume_col, turnover_col)
    raw = result.raw_data
    
    signals = {
        "ma_pattern": "",
        "ma_gaps": "",
        "ma_trend": "",
        "volume_pattern": "",
        "volume_level": "",
        "volume_ma_pattern": "",
        "macd_status": "",
        "macd_diff": 0,
        "rsi_status": "",
        "rsi_value": 0,
        "boll_position": "",
        "boll_ratio": 0
    }
    
    if raw:
        ma_pattern = ""
        if any("均线多头排列" in s for s in result.signals):
            ma_pattern = "多头排列"
        elif any("均线空头排列" in s for s in result.signals):
            ma_pattern = "空头排列"
        elif any("均线震荡" in s for s in result.signals):
            ma_pattern = "震荡"
        elif any("均线纠缠" in s for s in result.signals):
            ma_pattern = "纠缠"
        signals["ma_pattern"] = ma_pattern
        
        close_ma5 = raw.get("close_ma5_gap_pct", 0)
        ma5_ma10 = raw.get("ma5_ma10_gap_pct", 0)
        ma10_ma20 = raw.get("ma10_ma20_gap_pct", 0)
        signals["ma_gaps"] = f"P-M5:{close_ma5:+.1f}%,M5-M10:{ma5_ma10:+.1f}%,M10-M20:{ma10_ma20:+.1f}%"
        
        ma5_t = raw.get("ma5_trend_pct", 0)
        ma10_t = raw.get("ma10_trend_pct", 0)
        ma20_t = raw.get("ma20_trend_pct", 0)
        signals["ma_trend"] = f"M5{ma5_t:+.1f}%,M10{ma10_t:+.1f}%,M20{ma20_t:+.1f}%"
        
        vol_ratio = raw.get("volume_ma5_ratio", 0)
        if vol_ratio > 1.5:
            signals["volume_pattern"] = "放量"
        elif vol_ratio < 0.7:
            signals["volume_pattern"] = "缩量"
        else:
            signals["volume_pattern"] = "正常"
        signals["volume_level"] = f"{vol_ratio:.1f}x"
        
        vol_ma5 = raw.get("vol_ma5", 0)
        vol_ma10 = raw.get("vol_ma10", 0)
        vol_ma20 = raw.get("vol_ma20", 0)
        if vol_ma5 > 0 and vol_ma10 > 0 and vol_ma20 > 0:
            if vol_ma5 > vol_ma10 > vol_ma20:
                signals["volume_ma_pattern"] = "多头"
            elif vol_ma5 < vol_ma10 < vol_ma20:
                signals["volume_ma_pattern"] = "空头"
            else:
                signals["volume_ma_pattern"] = "-"
        else:
            signals["volume_ma_pattern"] = "-"
        
        macd_diff = raw.get("dif_dea_diff", 0)
        signals["macd_diff"] = round(macd_diff, 4)
        if macd_diff > 0:
            if any("MACD金叉" in s for s in result.signals):
                signals["macd_status"] = "金叉"
            elif macd_diff > 0.5:
                signals["macd_status"] = "强势"
            else:
                signals["macd_status"] = "偏强"
        else:
            if any("MACD死叉" in s for s in result.signals):
                signals["macd_status"] = "死叉"
            elif macd_diff < -0.5:
                signals["macd_status"] = "弱势"
            else:
                signals["macd_status"] = "偏弱"
        
        rsi = raw.get("rsi", 50)
        signals["rsi_value"] = round(rsi, 1)
        if rsi >= 70:
            signals["rsi_status"] = "超买"
        elif rsi >= 60:
            signals["rsi_status"] = "偏强"
        elif rsi <= 30:
            signals["rsi_status"] = "超卖"
        elif rsi <= 40:
            signals["rsi_status"] = "偏弱"
        else:
            signals["rsi_status"] = "中性"
        
        boll_ratio = raw.get("position_ratio", 0.5)
        signals["boll_ratio"] = round(boll_ratio, 2)
        if boll_ratio > 0.8:
            signals["boll_position"] = "上轨"
        elif boll_ratio < 0.2:
            signals["boll_position"] = "下轨"
        else:
            signals["boll_position"] = "中轨"
    
    return signals


def get_technical_raw_output(
    df: pd.DataFrame,
    close_col: str = "收盘",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None
) -> Dict:
    result = analyze_technical_pattern(df, close_col, volume_col, turnover_col)
    
    return {
        "scores": {
            "total": result.score,
            "ma": result.ma_score,
            "volume": result.volume_score,
            "macd": result.macd_score,
            "rsi": result.rsi_score,
            "boll": result.boll_score,
            "atr": result.atr_score
        },
        "raw_data": result.raw_data,
        "signals": result.signals,
        "summary": result.summary
    }


def analyze_volume_price_pattern(
    df: pd.DataFrame,
    price_change_col: str = "涨跌幅",
    volume_col: str = "成交量",
    turnover_col: Optional[str] = None,
    days: int = 3
) -> List[str]:
    if len(df) < days:
        return []
    
    patterns = []
    recent_df = df.head(days)
    
    price_changes = recent_df[price_change_col].values
    volumes = recent_df[volume_col].values
    
    turnover_rates = None
    if turnover_col and turnover_col in df.columns:
        turnover_rates = recent_df[turnover_col].values
    
    is_rising = all(pc > 0 for pc in price_changes)
    
    if not is_rising:
        return patterns
    
    volume_trend = []
    for i in range(len(volumes) - 1):
        if volumes[i] > volumes[i + 1]:
            volume_trend.append("increase")
        elif volumes[i] < volumes[i + 1]:
            volume_trend.append("decrease")
        else:
            volume_trend.append("stable")
    
    if all(t == "increase" for t in volume_trend):
        patterns.append("量价齐升")
    elif all(t == "decrease" for t in volume_trend):
        patterns.append("价涨量缩")
    elif all(t == "stable" for t in volume_trend):
        patterns.append("量价平稳")
    else:
        patterns.append("量价震荡")
    
    if turnover_rates is not None:
        turnover_trend = []
        for i in range(len(turnover_rates) - 1):
            if turnover_rates[i] > turnover_rates[i + 1]:
                turnover_trend.append("increase")
            elif turnover_rates[i] < turnover_rates[i + 1]:
                turnover_trend.append("decrease")
            else:
                turnover_trend.append("stable")
        
        if all(t == "increase" for t in turnover_trend):
            patterns.append("换手递增")
        elif all(t == "decrease" for t in turnover_trend):
            patterns.append("换手递减")
        elif all(t == "stable" for t in turnover_trend):
            patterns.append("换手平稳")
        else:
            patterns.append("换手震荡")
        
        avg_turnover = turnover_rates.mean()
        if avg_turnover > 10:
            patterns.append("高换手")
        elif avg_turnover > 5:
            patterns.append("中换手")
        else:
            patterns.append("低换手")
    
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
    patterns = analyze_volume_price_pattern(df, price_change_col, volume_col, turnover_col, days)
    
    if not patterns:
        return "无明显量价形态"
    
    return "、".join(patterns)
