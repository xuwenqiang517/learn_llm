import akshare as ak
import sys
from pathlib import Path
import pandas as pd
from datetime import date
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil, print_green, print_red
from utils.file_util import FileUtil

logger = LogUtil.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
ETF_DATA_DIR = DATA_DIR / "etf_data"
STOCK_DATA_DIR = DATA_DIR / "stock_data"

FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ETF_DATA_DIR, STOCK_DATA_DIR)

ETF_LIST_FILE = BASE_DATA_DIR / "etf_list.csv"
STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.csv"

_spot_cache = {}
_lrb_cache = {}

# ==============基础数据==============

def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df["trade_date"] <= date.today()].tail(days)["trade_date"].tolist()
    print_green(f"获取交易日列表成功: {len(trading_days)} 条")
    return [str(day).replace("-", "") for day in trading_days]


def _update_etf_list() -> None:
    try:
        print_green("开始更新 ETF 列表...")
        df = ak.fund_etf_spot_em()
        if "总市值" in df.columns:
            df["总市值"] = (df["总市值"] / 100000000).round(2).astype(str) + "亿"
        df[["代码", "名称", "总市值", "换手率", "量比"]].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
        print_green(f"ETF 列表已保存到: {ETF_LIST_FILE}")
    except Exception as e:
        print_red(f"更新 ETF 列表失败: {e}")


def _update_code_list() -> None:
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
    except Exception as e:
        logger.error(f"更新股票列表失败: {e}")
        raise


def _has_missing_days(cache_file: Path, trading_days: list) -> bool:
    if not cache_file.exists():
        return True
    cached_df = pd.read_csv(cache_file, encoding="utf-8-sig", parse_dates=["日期"])
    cached_dates = set(cached_df["日期"].dt.strftime("%Y%m%d").tolist())
    missing_days = [day for day in trading_days if day not in cached_dates]
    return len(missing_days) > 0


def _update_daily_data(
    list_file: Path,
    data_dir: Path,
    desc: str,
    fetch_func,
    max_workers: int = 50
) -> None:
    trading_days = _get_trading_days(40)
    list_df = pd.read_csv(list_file, encoding="utf-8-sig", dtype={"代码": str})

    def _update_single(code_name: tuple) -> None:
        code, name = code_name
        cache_file = data_dir / f"{code}.csv"
        if not _has_missing_days(cache_file, trading_days):
            return

        try:
            hist_df = fetch_func(symbol=code, start_date=trading_days[0], end_date=trading_days[-1])
            if not hist_df.empty:
                hist_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
            else:
                print_red(f"{desc.replace('更新 ', '').replace(' 日线数据', '')} 日线数据为空: {code}_{name}")
        except Exception as e:
            print_red(f"{desc.replace('更新 ', '').replace(' 日线数据', '')} 更新失败 {code}_{name}: {e}")

    code_names = list_df[["代码", "名称"]].values.tolist()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_update_single, code_names), desc=desc, unit="条", total=len(code_names)))
    print_green(f"{desc}完成")


# ==============基本面数据==============

def _fetch_stock_spot() -> pd.DataFrame:
    """获取A股实时行情"""
    if "spot" in _spot_cache:
        return _spot_cache["spot"]

    try:
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值", "涨跌幅", "换手率"]]
        df.columns = ["代码", "PE", "PB", "总市值", "流通市值", "涨跌幅", "换手率"]
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        _spot_cache["spot"] = df
        print_green(f"获取A股实时行情成功: {len(df)} 条")
        return df
    except Exception as e:
        print_red(f"获取A股实时行情失败: {e}")
        return pd.DataFrame()


def _fetch_etf_spot() -> pd.DataFrame:
    """获取ETF实时行情"""
    if "etf" in _spot_cache:
        return _spot_cache["etf"]

    try:
        df = ak.fund_etf_spot_em()
        df = df[["代码", "名称", "涨跌幅", "换手率", "IOPV实时估值", "总市值", "流通市值"]]
        df.columns = ["代码", "名称", "涨跌幅", "换手率", "净值", "总市值", "流通市值"]
        _spot_cache["etf"] = df
        print_green(f"获取ETF实时行情成功: {len(df)} 条")
        return df
    except Exception as e:
        print_red(f"获取ETF实时行情失败: {e}")
        return pd.DataFrame()


def _fetch_lrb_data() -> pd.DataFrame:
    """获取A股利润表数据（净利润同比、营收同比）"""
    if "lrb" in _lrb_cache:
        return _lrb_cache["lrb"]

    try:
        df = ak.stock_lrb_em(date="20240930")
        df = df[["股票代码", "净利润同比", "营业总收入同比"]]
        df.columns = ["代码", "净利润同比", "营收同比"]
        df["代码"] = df["代码"].astype(str).str.zfill(6)
        _lrb_cache["lrb"] = df
        print_green(f"获取利润表数据成功: {len(df)} 条")
        return df
    except Exception as e:
        print_red(f"获取利润表数据失败: {e}")
        return pd.DataFrame()


def _update_stock_fundamentals(file_path: Path, spot_df: pd.DataFrame, lrb_df: pd.DataFrame) -> None:
    """更新单个股票文件的基本面数据"""
    try:
        code = file_path.stem
        if code not in spot_df["代码"].values:
            return

        spot_data = spot_df[spot_df["代码"] == code].iloc[0]

        lrb_data = None
        if code in lrb_df["代码"].values:
            lrb_data = lrb_df[lrb_df["代码"] == code].iloc[0]

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty:
            return

        df["PE"] = round(spot_data["PE"], 2) if pd.notna(spot_data["PE"]) else None
        df["PB"] = round(spot_data["PB"], 2) if pd.notna(spot_data["PB"]) else None
        df["总市值"] = f"{round(spot_data['总市值'] / 1e8, 2)}亿" if pd.notna(spot_data["总市值"]) else None
        df["流通市值"] = f"{round(spot_data['流通市值'] / 1e8, 2)}亿" if pd.notna(spot_data["流通市值"]) else None

        if lrb_data is not None:
            df["净利润同比"] = f"{round(lrb_data['净利润同比'], 2)}%" if pd.notna(lrb_data["净利润同比"]) else None
            df["营收同比"] = f"{round(lrb_data['营收同比'], 2)}%" if pd.notna(lrb_data["营收同比"]) else None
        else:
            df["净利润同比"] = None
            df["营收同比"] = None

        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print_red(f"更新股票基本面失败 {file_path}: {e}")


def _update_etf_fundamentals(file_path: Path, etf_df: pd.DataFrame) -> None:
    """更新单个ETF文件的基本面数据"""
    try:
        code = file_path.stem
        if code not in etf_df["代码"].values:
            return

        fund_data = etf_df[etf_df["代码"] == code].iloc[0]

        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if df.empty:
            return

        df["净值"] = fund_data["净值"]
        df["PE"] = None
        df["PB"] = None
        df["净利润同比"] = None
        df["营收同比"] = None
        df["总市值"] = f"{round(fund_data['总市值'] / 1e8, 2)}亿" if pd.notna(fund_data.get("总市值")) else None
        df["流通市值"] = f"{round(fund_data['流通市值'] / 1e8, 2)}亿" if pd.notna(fund_data.get("流通市值")) else None

        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print_red(f"更新ETF基本面失败 {file_path}: {e}")


def _fill_fundamentals_aspect(max_workers: int = 20) -> None:
    """计算所有股票和ETF的基本面数据"""
    print_green("开始获取基本面数据...")

    stock_spot = _fetch_stock_spot()
    etf_spot = _fetch_etf_spot()
    lrb_data = _fetch_lrb_data()

    stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
    etf_files = list(ETF_DATA_DIR.glob("*.csv"))

    print_green(f"开始更新股票基本面，共 {len(stock_files)} 个文件")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda f: _update_stock_fundamentals(f, stock_spot, lrb_data), stock_files),
                  desc="更新股票基本面", unit="个", total=len(stock_files)))
    print_green("股票基本面更新完成")

    print_green(f"开始更新ETF基本面，共 {len(etf_files)} 个文件")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda f: _update_etf_fundamentals(f, etf_spot), etf_files),
                  desc="更新ETF基本面", unit="个", total=len(etf_files)))
    print_green("ETF基本面更新完成")


# ==============技术指标==============

def _calculate_ema(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return df[col].ewm(span=period, adjust=False).mean()


def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算MACD指标"""
    df = df.copy()
    ema12 = _calculate_ema(df, "收盘", 12)
    ema26 = _calculate_ema(df, "收盘", 26)
    df["DIF"] = (ema12 - ema26).round(2)
    df["DEA"] = _calculate_ema(df, "DIF", 9).round(2)
    df["MACD"] = ((df["DIF"] - df["DEA"]) * 2).round(2)
    return df


def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算RSI指标"""
    df = df.copy()
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = (100 - 100 / (1 + rs)).round(2)
    return df


def _calculate_boll(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算布林带"""
    df = df.copy()
    df["BOLL_MID"] = df["收盘"].rolling(window=period).mean().round(2)
    std = df["收盘"].rolling(window=period).std()
    df["BOLL_UP"] = (df["BOLL_MID"] + 2 * std).round(2)
    df["BOLL_LOW"] = (df["BOLL_MID"] - 2 * std).round(2)
    return df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算真实波幅"""
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
    """计算所有技术指标"""
    df = df.copy()

    df["MA5"] = df["收盘"].rolling(window=5).mean().round(2)
    df["MA10"] = df["收盘"].rolling(window=10).mean().round(2)
    df["MA20"] = df["收盘"].rolling(window=20).mean().round(2)

    if "成交量" in df.columns:
        df["VOL_MA5"] = df["成交量"].rolling(window=5).mean().round(2)
        df["VOL_MA10"] = df["成交量"].rolling(window=10).mean().round(2)
        df["VOL_MA20"] = df["成交量"].rolling(window=20).mean().round(2)

    df["连涨天数"] = (df["涨跌幅"] > 0).astype(int)[::-1].cumprod()[::-1]
    df["3日涨幅"] = df["涨跌幅"].rolling(window=3).sum().round(2)
    df["5日涨幅"] = df["涨跌幅"].rolling(window=5).sum().round(2)

    df = _calculate_macd(df)
    df = _calculate_rsi(df)
    df = _calculate_boll(df)
    df = _calculate_atr(df)

    return df


def _process_file(file_path: Path) -> None:
    """处理单个文件"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if len(df) < 5:
            return

        df = _calculate_indicators(df)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print_red(f"处理文件失败 {file_path}: {e}")


def _fill_technical_aspect(max_workers: int = 20) -> None:
    """计算所有股票和ETF的技术指标"""
    etf_files = list(ETF_DATA_DIR.glob("*.csv"))
    stock_files = list(STOCK_DATA_DIR.glob("*.csv"))

    all_files = etf_files + stock_files

    print_green(f"开始计算技术指标，共 {len(all_files)} 个文件")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_process_file, all_files), desc="计算技术指标", unit="个", total=len(all_files)))

    print_green("技术指标计算完成")


# ==============更新所有数据==============

def _update_all_data():
    _update_etf_list()
    _update_code_list()

    _update_daily_data(ETF_LIST_FILE, ETF_DATA_DIR, "更新 ETF 日线数据", ak.fund_etf_hist_em)
    _update_daily_data(STOCK_LIST_FILE, STOCK_DATA_DIR, "更新 股票 日线数据", ak.stock_zh_a_hist)

    _fill_fundamentals_aspect(max_workers=20)
    _fill_technical_aspect(max_workers=20)


if __name__ == "__main__":
    _update_all_data()
