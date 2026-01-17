import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agent.tool.load_base_data import (
    _get_trading_days, STOCK_DATA_DIR, ETF_DATA_DIR
)


def _calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _validate_all_indicators():
    from utils.log_util import print_yellow, print_green, print_red
    print_yellow("=" * 70)
    print_yellow("数据准确性验证 - 技术指标")
    print_yellow("=" * 70)

    stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
    etf_files = list(ETF_DATA_DIR.glob("*.csv"))
    all_files = stock_files + etf_files

    if len(all_files) == 0:
        print_red("没有数据文件")
        return False

    sample_files = all_files[:100]
    print_yellow(f"验证样本: {len(sample_files)} 个文件")

    indicator_stats = {
        'MA5': {'passed': 0, 'failed': 0},
        'MA10': {'passed': 0, 'failed': 0},
        'MA20': {'passed': 0, 'failed': 0},
        'VOL_MA5': {'passed': 0, 'failed': 0},
        'VOL_MA10': {'passed': 0, 'failed': 0},
        'VOL_MA20': {'passed': 0, 'failed': 0},
        '连涨天数': {'passed': 0, 'failed': 0},
        '连涨天数逻辑': {'passed': 0, 'failed': 0},
        '3日涨幅': {'passed': 0, 'failed': 0},
        '5日涨幅': {'passed': 0, 'failed': 0},
        'DIF': {'passed': 0, 'failed': 0},
        'DEA': {'passed': 0, 'failed': 0},
        'MACD': {'passed': 0, 'failed': 0},
        'RSI': {'passed': 0, 'failed': 0},
        'BOLL_MID': {'passed': 0, 'failed': 0},
        'BOLL_UP': {'passed': 0, 'failed': 0},
        'BOLL_LOW': {'passed': 0, 'failed': 0},
        'ATR': {'passed': 0, 'failed': 0},
    }

    for file_path in tqdm(sample_files, desc="验证指标", unit="个"):
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', parse_dates=['日期'])
            if len(df) < 30:
                continue

            results = _check_indicators(df, file_path.name)
            for indicator, passed in results.items():
                if passed:
                    indicator_stats[indicator]['passed'] += 1
                else:
                    indicator_stats[indicator]['failed'] += 1

        except Exception as e:
            print_red(f"验证失败 {file_path.name}: {e}")

    print_yellow("\n" + "=" * 70)
    print_yellow("【各指标验证结果】")
    print_yellow("=" * 70)
    print_yellow(f"{'指标':<15} {'通过':<10} {'失败':<10} {'状态'}")
    print_yellow("-" * 50)

    all_passed = True
    for indicator, stats in indicator_stats.items():
        total = stats['passed'] + stats['failed']
        if stats['failed'] > 0:
            status = "❌"
            all_passed = False
        else:
            status = "✅"
        print_yellow(f"{indicator:<15} {stats['passed']:<10} {stats['failed']:<10} {status}")

    print_yellow("-" * 50)
    if all_passed:
        print_green("✅ 所有指标验证通过")
    else:
        print_red("❌ 部分指标验证失败")
    print_yellow("=" * 70)

    return all_passed


def _check_indicators(df: pd.DataFrame, file_name: str) -> dict:
    results = {}

    close = df['收盘']
    pct_change = df['涨跌幅']
    high = df['最高']
    low = df['最低']
    volume = df.get('成交量')

    # MA5
    ma5_expected = close.rolling(5).mean().round(2)
    results['MA5'] = (df['MA5'].fillna(0) - ma5_expected.fillna(0)).abs().sum() <= 0.01

    # MA10
    ma10_expected = close.rolling(10).mean().round(2)
    results['MA10'] = (df['MA10'].fillna(0) - ma10_expected.fillna(0)).abs().sum() <= 0.01

    # MA20
    ma20_expected = close.rolling(20).mean().round(2)
    results['MA20'] = (df['MA20'].fillna(0) - ma20_expected.fillna(0)).abs().sum() <= 0.01

    # VOL_MA
    if volume is not None:
        vol_ma5 = volume.rolling(5).mean().round(2)
        results['VOL_MA5'] = (df['VOL_MA5'].fillna(0) - vol_ma5.fillna(0)).abs().sum() <= 0.01

        vol_ma10 = volume.rolling(10).mean().round(2)
        results['VOL_MA10'] = (df['VOL_MA10'].fillna(0) - vol_ma10.fillna(0)).abs().sum() <= 0.01

        vol_ma20 = volume.rolling(20).mean().round(2)
        results['VOL_MA20'] = (df['VOL_MA20'].fillna(0) - vol_ma20.fillna(0)).abs().sum() <= 0.01
    else:
        results['VOL_MA5'] = results['VOL_MA10'] = results['VOL_MA20'] = True

    # 连涨天数
    is_rising = (pct_change > 0).astype(int)
    consec_expected = is_rising.cumsum() - is_rising.cumsum().where(is_rising == 0).ffill().fillna(0).astype(int)
    results['连涨天数'] = df['连涨天数'].reset_index(drop=True).equals(consec_expected.reset_index(drop=True))

    # 连涨天数逻辑检查
    logic_passed = True
    for idx in range(len(df)):
        pct = pct_change.iloc[idx]
        consec = df['连涨天数'].iloc[idx]
        if pd.notna(pct) and pd.notna(consec):
            if pct > 0 and consec <= 0:
                logic_passed = False
                break
            elif pct < 0 and consec != 0:
                logic_passed = False
                break
    results['连涨天数逻辑'] = logic_passed

    # 3日涨幅
    d3_expected = pct_change.rolling(3).sum().round(2)
    results['3日涨幅'] = (df['3日涨幅'].fillna(0) - d3_expected.fillna(0)).abs().sum() <= 0.01

    # 5日涨幅
    d5_expected = pct_change.rolling(5).sum().round(2)
    results['5日涨幅'] = (df['5日涨幅'].fillna(0) - d5_expected.fillna(0)).abs().sum() <= 0.01

    # MACD
    ema12 = _calculate_ema(close, 12)
    ema26 = _calculate_ema(close, 26)
    dif_expected = (ema12 - ema26).round(2)
    results['DIF'] = (df['DIF'].fillna(0) - dif_expected.fillna(0)).abs().sum() <= 0.01

    dea_expected = _calculate_ema(dif_expected, 9).round(2)
    results['DEA'] = (df['DEA'].fillna(0) - dea_expected.fillna(0)).abs().sum() <= 0.01

    macd_expected = ((dif_expected - dea_expected) * 2).round(2)
    results['MACD'] = (df['MACD'].fillna(0) - macd_expected.fillna(0)).abs().sum() <= 0.01

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi_expected = (100 - 100 / (1 + rs)).round(2)
    results['RSI'] = (df['RSI'].fillna(0) - rsi_expected.fillna(0)).abs().sum() <= 0.01

    # BOLL
    boll_mid_expected = close.rolling(20).mean().round(2)
    results['BOLL_MID'] = (df['BOLL_MID'].fillna(0) - boll_mid_expected.fillna(0)).abs().sum() <= 0.01

    std = close.rolling(20).std()
    boll_up_expected = (boll_mid_expected + 2 * std).round(2)
    results['BOLL_UP'] = (df['BOLL_UP'].fillna(0) - boll_up_expected.fillna(0)).abs().sum() <= 0.01

    boll_low_expected = (boll_mid_expected - 2 * std).round(2)
    results['BOLL_LOW'] = (df['BOLL_LOW'].fillna(0) - boll_low_expected.fillna(0)).abs().sum() <= 0.01

    # ATR
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_expected = tr.rolling(14).mean().round(2)
    results['ATR'] = (df['ATR'].fillna(0) - atr_expected.fillna(0)).abs().sum() <= 0.01

    return results


def _validate_data_completeness():
    from utils.log_util import print_yellow, print_green, print_red
    print_yellow("=" * 70)
    print_yellow("数据完整性验证")
    print_yellow("=" * 70)
    trading_days = _get_trading_days()
    if len(trading_days) == 0:
        print_red("获取交易日列表失败")
        return False
    last_day = trading_days[-1]
    print_yellow(f"最后一个交易日: {last_day}")
    stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
    etf_files = list(ETF_DATA_DIR.glob("*.csv"))
    print_yellow(f"\n验证股票数据: {len(stock_files)} 个文件")
    missing_stocks = []
    for file_path in tqdm(stock_files[:500], desc="检查股票", unit="个"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", parse_dates=["日期"])
            if not df.empty:
                last_date = df["日期"].max().strftime("%Y%m%d")
                if last_date != last_day:
                    missing_stocks.append((file_path.stem, last_date))
        except Exception:
            continue
    print_yellow(f"\n验证ETF数据: {len(etf_files)} 个文件")
    missing_etfs = []
    for file_path in tqdm(etf_files, desc="检查ETF", unit="个"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig", parse_dates=["日期"])
            if not df.empty:
                last_date = df["日期"].max().strftime("%Y%m%d")
                if last_date != last_day:
                    missing_etfs.append((file_path.stem, last_date))
        except Exception:
            continue
    print_yellow("\n" + "=" * 70)
    print_yellow("【验证结果】")
    print_yellow(f"股票缺失: {len(missing_stocks)} / {len(stock_files)}")
    print_yellow(f"ETF缺失:  {len(missing_etfs)} / {len(etf_files)}")
    if len(missing_stocks) > 0:
        print_yellow(f"\n缺失数据的股票(前10个):")
        for code, last_date in missing_stocks[:10]:
            print_yellow(f"  {code}: 最后日期 {last_date}")
    if len(missing_etfs) > 0:
        print_yellow(f"\n缺失数据的ETF(前10个):")
        for code, last_date in missing_etfs[:10]:
            print_yellow(f"  {code}: 最后日期 {last_date}")
    total_missing = len(missing_stocks) + len(missing_etfs)
    total_files = len(stock_files) + len(etf_files)
    completeness = (total_files - total_missing) / total_files * 100
    print_yellow("\n" + "=" * 70)
    if completeness >= 99:
        print_green(f"数据完整性: {completeness:.1f}% ✅")
    elif completeness >= 95:
        print_yellow(f"数据完整性: {completeness:.1f}% ⚠️")
    else:
        print_red(f"数据完整性: {completeness:.1f}% ❌")
    print_yellow("=" * 70)
    return completeness >= 95


if __name__ == "__main__":
    _validate_data_completeness()
    print()
    _validate_all_indicators()
