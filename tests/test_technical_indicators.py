"""
技术指标计算准确性测试

独立验证各技术指标的计算逻辑，不依赖真实数据文件
通过生成模拟数据，手动计算预期值，与实际函数输出对比
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, '/Users/wq/Documents/github/learn_llm')

from agent.tool.load_base_data import (
    _calculate_ema, _calculate_macd, _calculate_rsi,
    _calculate_boll, _calculate_atr, _calculate_consecutive_rise,
    _calculate_indicators, _get_need_calc_rows, _calculate_indicators_incremental
)


def generate_mock_stock_data(days: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成模拟股票数据"""
    np.random.seed(seed)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    base_price = 100
    price_changes = np.random.randn(days) * 2
    price_changes[0] = 0
    
    close_prices = base_price + np.cumsum(price_changes)
    close_prices = np.maximum(close_prices, 1)
    
    df = pd.DataFrame({
        '日期': dates,
        '开盘': close_prices * (1 + np.random.randn(days) * 0.02),
        '收盘': close_prices,
        '最高': close_prices * (1 + np.abs(np.random.randn(days) * 0.03)),
        '最低': close_prices * (1 - np.abs(np.random.randn(days) * 0.03)),
        '成交量': np.random.randint(100000, 10000000, days),
        '涨跌幅': np.random.randn(days) * 2
    })
    
    df['开盘'] = np.maximum(df['开盘'], 1)
    df['最高'] = np.maximum(df['最高'], 1)
    df['最低'] = np.maximum(df['最低'], 1)
    
    df.loc[0, '涨跌幅'] = 0
    df['涨跌幅'] = (df['收盘'].pct_change() * 100).round(2)
    
    return df


def manual_ma(close: pd.Series, window: int) -> pd.Series:
    """手动计算移动平均线"""
    return close.rolling(window=window).mean().round(2)


def manual_ema(close: pd.Series, span: int) -> pd.Series:
    """手动计算EMA（不四舍五入，与实现一致）"""
    return close.ewm(span=span, adjust=False).mean()


def manual_macd(close: pd.Series) -> tuple:
    """手动计算MACD（四舍五入到2位小数）"""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = (ema12 - ema26).round(2)
    dea = dif.ewm(span=9, adjust=False).mean().round(2)
    macd = ((dif - dea) * 2).round(2)
    return dif, dea, macd


def manual_rsi(close: pd.Series, period: int = 6) -> pd.Series:
    """手动计算RSI（period=6，与实现一致）"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = (100 - 100 / (1 + rs)).round(2)
    
    return rsi


def manual_boll(close: pd.Series, window: int = 20, nbdev: int = 2) -> tuple:
    """手动计算布林带"""
    mid = close.rolling(window=window).mean().round(2)
    std = close.rolling(window=window).std()
    
    up = (mid + (std * nbdev)).round(2)
    low = (mid - (std * nbdev)).round(2)
    
    return mid, up, low


def manual_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """手动计算ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean().round(2)
    
    return atr


def manual_consecutive_rise(pct_change: pd.Series) -> pd.Series:
    """手动计算连涨天数"""
    is_rising = (pct_change > 0).astype(int)
    consecutive = is_rising.cumsum() - is_rising.cumsum().where(is_rising == 0).ffill().fillna(0).astype(int)
    return consecutive


def manual_cumulative_returns(pct_change: pd.Series, window: int) -> pd.Series:
    """手动计算累计涨幅"""
    return pct_change.rolling(window=window).sum().round(2)


class TestResult:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def add_pass(self, name: str):
        self.tests_passed += 1
        print(f"  ✅ {name}")
    
    def add_fail(self, name: str, details: str):
        self.tests_failed += 1
        self.failures.append((name, details))
        print(f"  ❌ {name}: {details}")
    
    def summary(self):
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print(f"测试结果: {self.tests_passed}/{total} 通过")
        if self.tests_failed > 0:
            print(f"失败: {self.tests_failed}")
            for name, details in self.failures:
                print(f"  - {name}: {details}")
        print(f"{'='*60}")
        return self.tests_failed == 0


def test_ma_indicators(df: pd.DataFrame, result: TestResult):
    """测试移动平均线指标"""
    print("\n📊 测试MA指标...")
    
    for window in [5, 10, 20]:
        col_name = f"MA{window}"
        actual = df[col_name]
        expected = manual_ma(df['收盘'], window)
        
        diff = (actual.fillna(0) - expected.fillna(0)).abs().sum()
        if diff < 0.01:
            result.add_pass(f"MA{window}")
        else:
            result.add_fail(f"MA{window}", f"差异={diff:.2f}")


def test_ema_indicators(df: pd.DataFrame, result: TestResult):
    """测试EMA指标"""
    print("\n📈 测试EMA指标...")
    
    for span in [12, 26]:
        col_name = f"EMA{span}"
        actual = _calculate_ema(df, '收盘', span)
        
        expected = manual_ema(df['收盘'], span)
        
        diff = (actual.fillna(0) - expected.fillna(0)).abs().sum()
        
        if diff < 0.01:
            result.add_pass(f"EMA{span}")
        else:
            result.add_fail(f"EMA{span}", f"差异={diff:.2f}")


def test_macd_indicators(df: pd.DataFrame, result: TestResult):
    """测试MACD指标"""
    print("\n📉 测试MACD指标...")
    
    df_with_macd = _calculate_macd(df.copy())
    
    dif_expected, dea_expected, macd_expected = manual_macd(df['收盘'])
    
    dif_diff = (df_with_macd['DIF'].fillna(0) - dif_expected.fillna(0)).abs().sum()
    dea_diff = (df_with_macd['DEA'].fillna(0) - dea_expected.fillna(0)).abs().sum()
    macd_diff = (df_with_macd['MACD'].fillna(0) - macd_expected.fillna(0)).abs().sum()
    
    if dif_diff < 0.01:
        result.add_pass("DIF")
    else:
        result.add_fail("DIF", f"差异={dif_diff:.2f}")
    
    if dea_diff < 0.01:
        result.add_pass("DEA")
    else:
        result.add_fail("DEA", f"差异={dea_diff:.2f}")
    
    if macd_diff < 0.01:
        result.add_pass("MACD")
    else:
        result.add_fail("MACD", f"差异={macd_diff:.2f}")


def test_rsi_indicators(df: pd.DataFrame, result: TestResult):
    """测试RSI指标"""
    print("\n📊 测试RSI指标...")
    
    df_with_rsi = _calculate_rsi(df.copy(), period=6)
    rsi_expected = manual_rsi(df['收盘'], period=6)
    
    rsi_diff = (df_with_rsi['RSI'].fillna(0) - rsi_expected.fillna(0)).abs().sum()
    
    if rsi_diff < 0.01:
        result.add_pass("RSI")
    else:
        result.add_fail("RSI", f"差异={rsi_diff:.2f}")


def test_boll_indicators(df: pd.DataFrame, result: TestResult):
    """测试布林带指标"""
    print("\n📊 测试BOLL指标...")
    
    df_with_boll = _calculate_boll(df.copy())
    mid_expected, up_expected, low_expected = manual_boll(df['收盘'])
    
    mid_diff = (df_with_boll['BOLL_MID'].fillna(0) - mid_expected.fillna(0)).abs().sum()
    up_diff = (df_with_boll['BOLL_UP'].fillna(0) - up_expected.fillna(0)).abs().sum()
    low_diff = (df_with_boll['BOLL_LOW'].fillna(0) - low_expected.fillna(0)).abs().sum()
    
    if mid_diff < 0.01:
        result.add_pass("BOLL_MID")
    else:
        result.add_fail("BOLL_MID", f"差异={mid_diff:.2f}")
    
    if up_diff < 0.01:
        result.add_pass("BOLL_UP")
    else:
        result.add_fail("BOLL_UP", f"差异={up_diff:.2f}")
    
    if low_diff < 0.01:
        result.add_pass("BOLL_LOW")
    else:
        result.add_fail("BOLL_LOW", f"差异={low_diff:.2f}")


def test_atr_indicators(df: pd.DataFrame, result: TestResult):
    """测试ATR指标"""
    print("\n📊 测试ATR指标...")
    
    df_with_atr = _calculate_atr(df.copy())
    atr_expected = manual_atr(df['最高'], df['最低'], df['收盘'])
    
    atr_diff = (df_with_atr['ATR'].fillna(0) - atr_expected.fillna(0)).abs().sum()
    
    if atr_diff < 0.01:
        result.add_pass("ATR")
    else:
        result.add_fail("ATR", f"差异={atr_diff:.2f}")


def test_consecutive_rise(df: pd.DataFrame, result: TestResult):
    """测试连涨天数指标"""
    print("\n🔢 测试连涨天数指标...")
    
    consecutive_series = _calculate_consecutive_rise(df.copy())
    consecutive_expected = manual_consecutive_rise(df['涨跌幅'])
    
    match_count = (consecutive_series.reset_index(drop=True) == consecutive_expected.reset_index(drop=True)).sum()
    total_count = len(consecutive_series)
    
    if match_count == total_count:
        result.add_pass("连涨天数")
    else:
        mismatch_indices = consecutive_series[consecutive_series != consecutive_expected].index.tolist()
        result.add_fail("连涨天数", f"不匹配数量={total_count - match_count}, 位置={mismatch_indices[:5]}")
    
    sample_test(df, consecutive_series, result)


def sample_test(df: pd.DataFrame, consecutive_series: pd.Series, result: TestResult):
    """抽样验证连涨天数逻辑"""
    sample_df = pd.DataFrame({
        '涨跌幅': df['涨跌幅'].values,
        '连涨天数': consecutive_series.values
    }).tail(20)
    
    correct_count = 0
    total_check = 0
    
    for _, row in sample_df.iterrows():
        pct = row['涨跌幅']
        consecutive = row['连涨天数']
        
        if pd.isna(pct) or pd.isna(consecutive):
            continue
            
        total_check += 1
        
        if pct > 0:
            if consecutive >= 1:
                correct_count += 1
        else:
            if consecutive == 0:
                correct_count += 1
    
    if total_check > 0 and correct_count == total_check:
        result.add_pass("连涨天数逻辑抽样验证")
    else:
        result.add_fail("连涨天数逻辑抽样验证", f"正确={correct_count}/{total_check}")


def test_cumulative_returns(df: pd.DataFrame, result: TestResult):
    """测试累计涨幅指标"""
    print("\n📈 测试累计涨幅指标...")
    
    for window in [3, 5]:
        col_name = f"{window}日涨幅"
        expected = manual_cumulative_returns(df['涨跌幅'], window)
        
        actual = df[col_name]
        diff = (actual.fillna(0) - expected.fillna(0)).abs().sum()
        
        if diff < 0.01:
            result.add_pass(col_name)
        else:
            result.add_fail(col_name, f"差异={diff:.2f}")


def test_incremental_update(df: pd.DataFrame, result: TestResult):
    """测试增量更新逻辑"""
    print("\n🔄 测试增量更新逻辑...")
    
    df_with_indicators = _calculate_indicators(df.copy())
    
    start_idx, count = _get_need_calc_rows(df_with_indicators)
    
    if count == 0:
        result.add_pass("全量数据无需增量计算")
    else:
        result.add_fail("全量数据无需增量计算", f"返回需要计算{count}行")
    
    print("\n测试2: 检测缺失指标的数据")
    df_missing = df.copy()
    df_missing.loc[df_missing.index[-5]:, 'MA20'] = np.nan
    
    start_idx2, count2 = _get_need_calc_rows(df_missing)
    
    if count2 >= 5:
        result.add_pass("缺失MA20触发增量计算")
    else:
        result.add_fail("缺失MA20触发增量计算", f"应需计算>=5行，实际{count2}行")
    
    print("\n测试3: 检测连涨天数全为0的异常数据")
    df_bug = df.copy()
    df_bug['连涨天数'] = 0
    
    start_idx3, count3 = _get_need_calc_rows(df_bug)
    
    if count3 == len(df_bug):
        result.add_pass("连涨天数全0触发全量计算")
    else:
        result.add_fail("连涨天数全0触发全量计算", f"应需全量计算，实际count={count3}")


def test_edge_cases(result: TestResult):
    """测试边界情况"""
    print("\n⚠️ 测试边界情况...")
    
    empty_df = pd.DataFrame(columns=['收盘', '涨跌幅', '最高', '最低'])
    start_idx, count = _get_need_calc_rows(empty_df)
    
    if count == 0 or count == len(empty_df):
        result.add_pass("空数据处理")
    else:
        result.add_fail("空数据处理", f"返回{count}行")
    
    short_df = pd.DataFrame({
        '收盘': [100, 101, 102, 103, 104],
        '涨跌幅': [0, 1, 1, 1, 1],
        '最高': [101, 102, 103, 104, 105],
        '最低': [99, 100, 101, 102, 103]
    })
    
    start_idx, count = _get_need_calc_rows(short_df)
    
    if count == len(short_df):
        result.add_pass("短数据全量计算")
    else:
        result.add_fail("短数据全量计算", f"应返回全量{len(short_df)}行，实际{count}行")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("技术指标计算准确性测试")
    print("="*60)
    
    result = TestResult()
    
    print("\n生成测试数据...")
    df = generate_mock_stock_data(days=100, seed=42)
    print(f"测试数据: {len(df)}行")
    print(f"日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
    
    print("\n计算技术指标...")
    df = _calculate_indicators(df.copy())
    
    test_ma_indicators(df, result)
    test_ema_indicators(df, result)
    test_macd_indicators(df, result)
    test_rsi_indicators(df, result)
    test_boll_indicators(df, result)
    test_atr_indicators(df, result)
    test_consecutive_rise(df, result)
    test_cumulative_returns(df, result)
    test_incremental_update(df, result)
    test_edge_cases(result)
    
    success = result.summary()
    
    if success:
        print("\n🎉 所有测试通过！技术指标计算准确。")
    else:
        print("\n⚠️ 部分测试失败，请检查上述问题。")
    
    return success


def test_consecutive_rise_edge_cases():
    """专门测试连涨天数边界情况"""
    print("\n" + "="*60)
    print("连涨天数边界情况专项测试")
    print("="*60)
    
    test_cases = [
        ("持续上涨", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ("持续下跌", [-1, -2, -3, -4, -5], [0, 0, 0, 0, 0]),
        ("涨跌交替", [1, -1, 1, -1, 1], [1, 0, 1, 0, 1]),
        ("先跌后涨", [-1, -1, 1, 2, 3], [0, 0, 1, 2, 3]),
        ("先涨后跌", [1, 2, 3, -1, -1], [1, 2, 3, 0, 0]),
        ("零涨幅", [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
    ]
    
    result = TestResult()
    
    for name, changes, expected in test_cases:
        print(f"\n测试: {name}")
        
        df = pd.DataFrame({
            '日期': pd.date_range('2026-01-01', periods=len(changes)),
            '涨跌幅': changes
        })
        
        df_result = _calculate_consecutive_rise(df.copy())
        actual = df_result.tolist()
        
        if actual == expected:
            result.add_pass(name)
        else:
            result.add_fail(name, f"期望{expected}，实际{actual}")
    
    result.summary()
    return result.tests_failed == 0


def test_incremental_logic():
    """测试增量计算逻辑"""
    print("\n" + "="*60)
    print("增量计算逻辑专项测试")
    print("="*60)
    
    result = TestResult()
    
    df = generate_mock_stock_data(days=50, seed=123)
    df = _calculate_indicators(df.copy())
    
    print("\n测试1: 全量数据不应触发增量计算")
    start, count = _get_need_calc_rows(df)
    if count == 0:
        result.add_pass("全量数据无需增量")
    else:
        result.add_fail("全量数据无需增量", f"count={count}")
    
    print("\n测试2: 缺失指标数据应触发增量计算")
    df_missing = df.copy()
    df_missing.loc[df_missing.index[-10]:, 'ATR'] = np.nan
    start, count = _get_need_calc_rows(df_missing)
    if count >= 10:
        result.add_pass("缺失ATR触发增量")
    else:
        result.add_fail("缺失ATR触发增量", f"count={count}")
    
    print("\n测试3: 增量计算结果正确性")
    df_base = generate_mock_stock_data(days=80, seed=456)
    df_base = _calculate_indicators(df_base.copy())
    
    df_extended = generate_mock_stock_data(days=100, seed=456)
    
    result_incremental = _calculate_indicators_incremental(df_extended.copy())
    result_full = _calculate_indicators(df_extended.copy())
    
    ma20_incremental = result_incremental['MA20'].iloc[-1]
    ma20_full = result_full['MA20'].iloc[-1]
    
    if abs(ma20_full - ma20_incremental) < 0.01:
        result.add_pass("增量与全量计算一致")
    else:
        result.add_fail("增量与全量计算一致", f"full={ma20_full}, incremental={ma20_incremental}")
    
    result.summary()
    return result.tests_failed == 0


if __name__ == "__main__":
    success1 = run_all_tests()
    success2 = test_consecutive_rise_edge_cases()
    success3 = test_incremental_logic()
    
    if success1 and success2 and success3:
        print("\n" + "="*60)
        print("🎉 全部测试通过！")
        print("="*60)
        exit(0)
    else:
        print("\n" + "="*60)
        print("❌ 测试失败")
        print("="*60)
        exit(1)
