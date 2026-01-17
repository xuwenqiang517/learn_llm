import akshare as ak
import sys
import time
from pathlib import Path
import pandas as pd
from datetime import date, datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil, print_green, print_red, print_yellow
from utils.file_util import FileUtil
from utils.data_path_util import (
    get_base_data_dir, get_etf_data_dir, get_stock_data_dir,
    get_etf_list_file, get_stock_list_file, get_last_trading_day,
    _delete_old_list_files, _is_same_day_data,
    get_industry_cache_file, get_lrb_cache_file,
    get_stock_spot_cache_file, get_etf_spot_cache_file,
    _is_cache_expired, load_cache, save_cache
)

logger = LogUtil.get_logger(__name__)

BASE_DATA_DIR = get_base_data_dir()
ETF_DATA_DIR = get_etf_data_dir()
STOCK_DATA_DIR = get_stock_data_dir()

FileUtil.ensure_dirs(BASE_DATA_DIR, ETF_DATA_DIR, STOCK_DATA_DIR)

_spot_cache = {}
_lrb_cache = {}
_industry_cache = {}

THREAD_POOL_IO = 8
THREAD_POOL_CPU = 8
API_CALL_INTERVAL = 0.1
FAST_TEST_MODE = False
FAST_TEST_SAMPLE_SIZE = 300


def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df["trade_date"] <= date.today()].tail(days)["trade_date"].tolist()
    print_green(f"获取交易日列表成功: {len(trading_days)} 条")
    return [str(day).replace("-", "") for day in trading_days]


def _update_etf_list() -> None:
    ETF_LIST_FILE = get_etf_list_file()
    if _is_same_day_data(ETF_LIST_FILE):
        print_green(f"ETF列表已是最新（{ETF_LIST_FILE.name}），跳过更新")
        return
    try:
        print_green("开始更新 ETF 列表...")
        df = ak.fund_etf_spot_em()
        if "总市值" in df.columns:
            df["总市值"] = (df["总市值"] / 100000000).round(2).astype(str) + "亿"
        df[["代码", "名称", "总市值", "换手率", "量比"]].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
        print_green(f"ETF 列表已保存到: {ETF_LIST_FILE}")
        last_trading = get_last_trading_day()
        _delete_old_list_files("etf_list", last_trading)
        print_green("已清理旧版 ETF 列表文件")
    except Exception as e:
        print_red(f"更新 ETF 列表失败: {e}")


def _update_code_list() -> None:
    STOCK_LIST_FILE = get_stock_list_file()
    if _is_same_day_data(STOCK_LIST_FILE):
        print_green(f"股票列表已是最新（{STOCK_LIST_FILE.name}），跳过更新")
        return
    try:
        big_df = pd.DataFrame()
        logger.info("获取上海主板A股...")
        stock_sh = ak.stock_info_sh_name_code(symbol="主板A股")
        if not stock_sh.empty:
            stock_sh = stock_sh[["证券代码", "证券简称"]].copy()
            stock_sh["type"] = "上海主板"
            stock_sh.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_sh], ignore_index=True)

        logger.info("获取深圳A股...")
        stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
        if not stock_sz.empty:
            stock_sz = stock_sz.copy()
            stock_sz["A股代码"] = stock_sz["A股代码"].astype(str).str.zfill(6)

            stock_sz_main = stock_sz[stock_sz["A股代码"].str.startswith("000")][["A股代码", "A股简称"]].copy()
            if not stock_sz_main.empty:
                stock_sz_main["type"] = "深圳主板"
                stock_sz_main.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_main], ignore_index=True)

            stock_sz_sme = stock_sz[stock_sz["A股代码"].str.startswith("002")][["A股代码", "A股简称"]].copy()
            if not stock_sz_sme.empty:
                stock_sz_sme["type"] = "中小板"
                stock_sz_sme.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_sme], ignore_index=True)

            stock_sz_gem = stock_sz[stock_sz["A股代码"].str.startswith("300")][["A股代码", "A股简称"]].copy()
            if not stock_sz_gem.empty:
                stock_sz_gem["type"] = "创业板"
                stock_sz_gem.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_gem], ignore_index=True)

            stock_sz_gem_reg = stock_sz[stock_sz["A股代码"].str.startswith("301")][["A股代码", "A股简称"]].copy()
            if not stock_sz_gem_reg.empty:
                stock_sz_gem_reg["type"] = "创业板"
                stock_sz_gem_reg.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_gem_reg], ignore_index=True)

        logger.info("获取科创板...")
        stock_kcb = ak.stock_info_sh_name_code(symbol="科创板")
        if not stock_kcb.empty:
            stock_kcb = stock_kcb[["证券代码", "证券简称"]].copy()
            stock_kcb["type"] = "科创板"
            stock_kcb.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_kcb], ignore_index=True)

        logger.info("获取北京股...")
        stock_bse = ak.stock_info_bj_name_code()
        if not stock_bse.empty:
            stock_bse = stock_bse[["证券代码", "证券简称"]].copy()
            stock_bse["type"] = "北京股"
            stock_bse.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_bse], ignore_index=True)

        big_df.to_csv(STOCK_LIST_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"股票列表已更新（{len(big_df)}只）")
        logger.info(big_df["板块类型"].value_counts())
        last_trading = get_last_trading_day()
        _delete_old_list_files("stock_list", last_trading)
        logger.info("已清理旧版股票列表文件")
    except Exception as e:
        logger.error(f"更新股票列表失败: {e}")
        raise


def _has_missing_days(cache_file: Path, trading_days: list) -> bool:
    if not cache_file.exists():
        return True
    cached_df = pd.read_csv(cache_file, encoding="utf-8-sig", parse_dates=["日期"])
    if cached_df.empty:
        return True
    cached_dates = set(cached_df["日期"].dt.strftime("%Y%m%d").tolist())
    recent_days = trading_days[-3:]
    missing_days = [day for day in recent_days if day not in cached_dates]
    return len(missing_days) > 0


def _update_daily_data(list_file: Path, data_dir: Path, desc: str, fetch_func, max_workers: int = THREAD_POOL_IO) -> tuple:
    trading_days = _get_trading_days(40)
    list_df = pd.read_csv(list_file, encoding="utf-8-sig", dtype={"代码": str})
    updated_files = []
    stats = {"remote": 0, "cached": 0, "failed": 0}

    def _update_single(code_name: tuple) -> str:
        code, name = code_name
        cache_file = data_dir / f"{code}.csv"
        if _has_missing_days(cache_file, trading_days):
            try:
                hist_df = fetch_func(symbol=code, start_date=trading_days[0], end_date=trading_days[-1])
                if not hist_df.empty:
                    if "涨跌额" in hist_df.columns:
                        hist_df = hist_df.drop(columns=["涨跌额"])
                    hist_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
                    time.sleep(API_CALL_INTERVAL)
                    return str(cache_file)
                else:
                    stats["failed"] += 1
            except Exception:
                stats["failed"] += 1
        else:
            stats["cached"] += 1
        return ""

    code_names = list_df[["代码", "名称"]].values.tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_update_single, code_names), desc=desc, unit="条", total=len(code_names)))

    updated_files = [Path(f) for f in results if f]
    stats["remote"] = len(updated_files)
    print_green(f"{desc}完成")
    print_green(f"  远程获取: {stats['remote']} 个")
    print_green(f"  缓存跳过: {stats['cached']} 个")
    if stats["failed"] > 0:
        print_red(f"  获取失败: {stats['failed']} 个")
    return updated_files, stats


def _fetch_stock_spot() -> pd.DataFrame:
    cache_file = get_stock_spot_cache_file()
    cached = load_cache(cache_file)
    if cached is not None and not _is_cache_expired(cache_file, days=1):
        _spot_cache["spot"] = cached
        print_green(f"从缓存加载A股实时行情: {len(cached)} 条")
        return cached
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值", "涨跌幅", "换手率"]]
        df.columns = ["代码", "PE", "PB", "总市值", "流通市值", "涨跌幅", "换手率"]
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        _spot_cache["spot"] = df
        save_cache(df, cache_file)
        print_green(f"获取A股实时行情成功: {len(df)} 条")
        return df
    except Exception as e:
        print_red(f"获取A股实时行情失败: {e}")
        return pd.DataFrame()


def _fetch_etf_spot() -> pd.DataFrame:
    cache_file = get_etf_spot_cache_file()
    cached = load_cache(cache_file)
    if cached is not None and not _is_cache_expired(cache_file, days=1):
        _spot_cache["etf"] = cached
        print_green(f"从缓存加载ETF实时行情: {len(cached)} 条")
        return cached
    try:
        df = ak.fund_etf_spot_em()
        df = df[["代码", "名称", "涨跌幅", "换手率", "IOPV实时估值", "总市值", "流通市值"]]
        df.columns = ["代码", "名称", "涨跌幅", "换手率", "净值", "总市值", "流通市值"]
        _spot_cache["etf"] = df
        save_cache(df, cache_file)
        print_green(f"获取ETF实时行情成功: {len(df)} 条")
        return df
    except Exception as e:
        print_red(f"获取ETF实时行情失败: {e}")
        return pd.DataFrame()


def _fetch_lrb_data() -> pd.DataFrame:
    cache_file = get_lrb_cache_file()
    cached = load_cache(cache_file)
    if cached is not None and not _is_cache_expired(cache_file, days=30):
        _lrb_cache["lrb"] = cached
        print_green(f"从缓存加载利润表数据: {len(cached)} 条")
        return cached
    try:
        df = ak.stock_lrb_em(date="20240930")
        df = df[["股票代码", "净利润同比", "营业总收入同比"]]
        df.columns = ["代码", "净利润同比", "营收同比"]
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        _lrb_cache["lrb"] = df
        save_cache(df, cache_file)
        print_green(f"获取利润表数据成功: {len(df)} 条，已缓存30天")
        return df
    except Exception as e:
        print_red(f"获取利润表数据失败: {e}")
        return pd.DataFrame()


def _fetch_industry_mapping() -> dict:
    cache_file = get_industry_cache_file()
    cached = load_cache(cache_file)
    if cached is not None and not _is_cache_expired(cache_file, days=7):
        _industry_cache.update(cached)
        print_green(f"从缓存加载行业映射: {len(cached)} 只股票")
        return _industry_cache
    try:
        industry_df = ak.stock_board_industry_name_em()
        if industry_df.empty:
            return {}
        for _, row in industry_df.iterrows():
            code = row["板块代码"]
            name = row["板块名称"]
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=code)
                if not cons_df.empty:
                    for _, stock in cons_df.iterrows():
                        stock_code = str(stock["代码"]).zfill(6)
                        _industry_cache[stock_code] = name
            except Exception:
                continue
            time.sleep(API_CALL_INTERVAL)
        save_cache(_industry_cache, cache_file)
        print_green(f"获取行业映射成功: {len(_industry_cache)} 只股票，已缓存7天")
    except Exception as e:
        print_red(f"获取行业映射失败: {e}")
    return _industry_cache


def _update_fundamentals(file_path: Path, spot_df: pd.DataFrame, lrb_df: pd.DataFrame, industry_map: dict, is_etf: bool = False) -> None:
    try:
        code = file_path.stem
        if is_etf:
            if code not in spot_df["代码"].values:
                return
            fund_data = spot_df[spot_df["代码"] == code].iloc[0]
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            if df.empty:
                return
            nav = fund_data["净值"]
            market_cap = f"{round(fund_data['总市值'] / 1e8, 2)}亿" if pd.notna(fund_data.get("总市值")) else None
            circ_market_cap = f"{round(fund_data['流通市值'] / 1e8, 2)}亿" if pd.notna(fund_data.get("流通市值")) else None
            last_row = df.iloc[-1] if len(df) > 0 else None
            if last_row is not None and last_row.get("净值") == nav and last_row.get("总市值") == market_cap and last_row.get("流通市值") == circ_market_cap:
                return
            df["净值"] = nav
            df["PE"] = None
            df["PB"] = None
            df["净利润同比"] = None
            df["营收同比"] = None
            df["总市值"] = market_cap
            df["流通市值"] = circ_market_cap
        else:
            if code not in spot_df["代码"].values:
                return
            spot_data = spot_df[spot_df["代码"] == code].iloc[0]
            lrb_data = None
            if code in lrb_df["代码"].values:
                lrb_data = lrb_df[lrb_df["代码"] == code].iloc[0]
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            if df.empty:
                return
            pe = round(spot_data["PE"], 2) if pd.notna(spot_data["PE"]) else None
            pb = round(spot_data["PB"], 2) if pd.notna(spot_data["PB"]) else None
            market_cap = f"{round(spot_data['总市值'] / 1e8, 2)}亿" if pd.notna(spot_data['总市值']) else None
            circ_market_cap = f"{round(spot_data['流通市值'] / 1e8, 2)}亿" if pd.notna(spot_data['流通市值']) else None
            net_profit_yoy = f"{round(lrb_data['净利润同比'], 2)}%" if lrb_data is not None and pd.notna(lrb_data["净利润同比"]) else None
            revenue_yoy = f"{round(lrb_data['营收同比'], 2)}%" if lrb_data is not None and pd.notna(lrb_data["营收同比"]) else None
            industry = industry_map.get(code, None)
            last_row = df.iloc[-1] if len(df) > 0 else None
            if last_row is not None and last_row.get("PE") == pe and last_row.get("PB") == pb and last_row.get("总市值") == market_cap and last_row.get("流通市值") == circ_market_cap and last_row.get("净利润同比") == net_profit_yoy and last_row.get("营收同比") == revenue_yoy and last_row.get("行业") == industry:
                return
            df["PE"] = pe
            df["PB"] = pb
            df["总市值"] = market_cap
            df["流通市值"] = circ_market_cap
            df["净利润同比"] = net_profit_yoy
            df["营收同比"] = revenue_yoy
            df["行业"] = industry

        df = _reorder_columns(df)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print_red(f"更新基本面失败 {file_path}: {e}")


def _update_stock_fundamentals(file_path: Path, spot_df: pd.DataFrame, lrb_df: pd.DataFrame, industry_map: dict) -> None:
    _update_fundamentals(file_path, spot_df, lrb_df, industry_map, is_etf=False)


def _update_etf_fundamentals(file_path: Path, etf_df: pd.DataFrame) -> None:
    _update_fundamentals(file_path, etf_df, pd.DataFrame(), {}, is_etf=True)


def _fill_fundamentals_aspect(max_workers: int = THREAD_POOL_IO, stock_files: list = None, etf_files: list = None) -> None:
    print_green("开始获取基本面数据...")
    stock_spot = _fetch_stock_spot()
    etf_spot = _fetch_etf_spot()
    lrb_data = _fetch_lrb_data()
    industry_map = _fetch_industry_mapping()
    if stock_files is None:
        stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
    if etf_files is None:
        etf_files = list(ETF_DATA_DIR.glob("*.csv"))
    print_green(f"开始更新股票基本面，共 {len(stock_files)} 个文件")
    if stock_files:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(lambda f: _update_stock_fundamentals(f, stock_spot, lrb_data, industry_map), stock_files), desc="更新股票基本面", unit="个", total=len(stock_files)))
    print_green("股票基本面更新完成")
    print_green(f"开始更新ETF基本面，共 {len(etf_files)} 个文件")
    if etf_files:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(lambda f: _update_etf_fundamentals(f, etf_spot), etf_files), desc="更新ETF基本面", unit="个", total=len(etf_files)))
    print_green("ETF基本面更新完成")


def _calculate_ema(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema12 = _calculate_ema(df, "收盘", 12)
    ema26 = _calculate_ema(df, "收盘", 26)
    df["DIF"] = (ema12 - ema26).round(2)
    df["DEA"] = _calculate_ema(df, "DIF", 9).round(2)
    df["MACD"] = ((df["DIF"] - df["DEA"]) * 2).round(2)
    return df


def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = (100 - 100 / (1 + rs)).round(2)
    return df


def _calculate_consecutive_rise(df: pd.DataFrame) -> pd.Series:
    is_rising = (df["涨跌幅"] > 0).astype(int)
    consecutive = is_rising.cumsum() - is_rising.cumsum().where(is_rising == 0).ffill().fillna(0).astype(int)
    return consecutive


def _calculate_boll(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["BOLL_MID"] = df["收盘"].rolling(window=period).mean().round(2)
    std = df["收盘"].rolling(window=period).std()
    df["BOLL_UP"] = (df["BOLL_MID"] + 2 * std).round(2)
    df["BOLL_LOW"] = (df["BOLL_MID"] - 2 * std).round(2)
    return df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high = df["最高"]
    low = df["最低"]
    close = df["收盘"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean().round(2)
    return df


def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["收盘"].rolling(window=5).mean().round(2)
    df["MA10"] = df["收盘"].rolling(window=10).mean().round(2)
    df["MA20"] = df["收盘"].rolling(window=20).mean().round(2)
    if "成交量" in df.columns:
        df["VOL_MA5"] = df["成交量"].rolling(window=5).mean().round(2)
        df["VOL_MA10"] = df["成交量"].rolling(window=10).mean().round(2)
        df["VOL_MA20"] = df["成交量"].rolling(window=20).mean().round(2)
    df["连涨天数"] = _calculate_consecutive_rise(df)
    df["3日涨幅"] = df["涨跌幅"].rolling(window=3).sum().round(2)
    df["5日涨幅"] = df["涨跌幅"].rolling(window=5).sum().round(2)
    df = _calculate_macd(df)
    df = _calculate_rsi(df)
    df = _calculate_boll(df)
    df = _calculate_atr(df)
    return df


def _get_need_calc_rows(df: pd.DataFrame) -> tuple:
    required_cols = ["MA5", "MA10", "MA20", "DIF", "DEA", "MACD", "RSI", "BOLL_MID", "BOLL_UP", "BOLL_LOW", "ATR", "连涨天数"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > len(required_cols) * 0.5:
        return 0, len(df)
    lookback = 25
    if len(df) <= lookback:
        return 0, len(df)
    consecutive_all_zeros = (df["连涨天数"] == 0).all() if "连涨天数" in df.columns else False
    has_consecutive_error = False
    if "连涨天数" in df.columns and "涨跌幅" in df.columns:
        pct_change = df["涨跌幅"]
        consecutive = df["连涨天数"]
        for idx in range(len(df)):
            if idx < len(pct_change) and idx < len(consecutive):
                pct = pct_change.iloc[idx]
                consec = consecutive.iloc[idx]
                if pd.notna(pct) and pd.notna(consec):
                    if pct > 0 and consec <= 0:
                        has_consecutive_error = True
                        break
                    if pct < 0 and consec != 0:
                        has_consecutive_error = True
                        break
    if has_consecutive_error:
        return 0, len(df)
    for i in range(len(df) - 1, max(0, len(df) - lookback - 1), -1):
        row = df.iloc[i]
        ma20_valid = pd.notna(row.get("MA20"))
        atr_valid = pd.notna(row.get("ATR"))
        if ma20_valid and atr_valid:
            if consecutive_all_zeros:
                return 0, len(df)
            if i == len(df) - 1:
                return len(df), 0
            return i + 1, len(df) - i - 1
    return 0, len(df)


def _calculate_indicators_incremental(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_idx, calc_count = _get_need_calc_rows(df)
    if calc_count == 0:
        return df
    if start_idx == 0 and calc_count == len(df):
        return _calculate_indicators(df)
    end_idx = len(df)
    lookback = 25
    calc_start = max(0, start_idx - lookback)
    temp_df = df.iloc[calc_start:end_idx].copy().reset_index(drop=True)
    temp_df = _calculate_indicators(temp_df)
    result_df = df.copy()
    for i in range(calc_start, end_idx):
        temp_idx = i - calc_start
        result_df.iloc[i] = temp_df.iloc[temp_idx]
    return result_df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_order = ["日期", "股票代码", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "连涨天数", "3日涨幅", "5日涨幅", "换手率", "PE", "PB", "总市值", "流通市值", "净利润同比", "营收同比", "行业", "MA5", "MA10", "MA20", "VOL_MA5", "VOL_MA10", "VOL_MA20", "DIF", "DEA", "MACD", "RSI", "BOLL_MID", "BOLL_UP", "BOLL_LOW", "ATR"]
    df = df.copy()
    if "涨跌额" in df.columns:
        df = df.drop(columns=["涨跌额"])
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    return df[column_order]


def _fill_technical_aspect(files: list = None) -> None:
    if files is None:
        etf_files = list(ETF_DATA_DIR.glob("*.csv"))
        stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
        files = etf_files + stock_files
    if not files:
        return
    print_green(f"开始计算技术指标，共 {len(files)} 个文件")
    file_data_map = {}
    files_to_write = []
    for file_path in tqdm(files, desc="读取文件", unit="个"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            if len(df) < 5:
                continue
            start_idx, calc_count = _get_need_calc_rows(df)
            if calc_count == 0:
                continue
            file_data_map[file_path] = df
            files_to_write.append(file_path)
        except Exception as e:
            print_red(f"读取文件失败 {file_path}: {e}")
    if not files_to_write:
        print_green("所有文件无需更新技术指标")
        return
    print_green(f"需要计算的文件: {len(files_to_write)} 个")
    for file_path in tqdm(files_to_write, desc="计算指标", unit="个"):
        try:
            df = file_data_map[file_path]
            df = _calculate_indicators_incremental(df)
            df = _reorder_columns(df)
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print_red(f"计算指标失败 {file_path}: {e}")
    print_green("技术指标计算完成")


def _update_all_data():
    _update_etf_list()
    _update_code_list()
    etf_updated, etf_stats = _update_daily_data(get_etf_list_file(), ETF_DATA_DIR, "更新 ETF 日线数据", ak.fund_etf_hist_em)
    stock_updated, stock_stats = _update_daily_data(get_stock_list_file(), STOCK_DATA_DIR, "更新 股票 日线数据", ak.stock_zh_a_hist)
    _fill_fundamentals_aspect(max_workers=THREAD_POOL_IO, stock_files=stock_updated, etf_files=etf_updated)
    _fill_technical_aspect(files=stock_updated + etf_updated)


if __name__ == "__main__":
    _update_all_data()
    from benchmarks_load_base_data import _validate_data_completeness, _validate_all_indicators
    print()
    _validate_data_completeness()
    print()
    _validate_all_indicators()
