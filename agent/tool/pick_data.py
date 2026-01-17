import akshare as ak
import sys
from pathlib import Path
from datetime import date
from typing import List, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.file_util import FileUtil
from utils.data_path_util import (
    get_stock_list_file, get_etf_list_file, get_stock_data_dir, get_etf_data_dir, get_pick_dir
)


class FilterPlugin:
    name: str
    desc: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class 多头排列过滤器(FilterPlugin):
    def __init__(self, days: int = 3):
        self.days = days
        self.name = f"多头排列_{days}日"
        self.desc = f"近{days}天多头排列（MA5 > MA10 > MA20）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        df = df.head(self.days).copy()
        ma_cols = ["MA5", "MA10", "MA20"]

        if not all(col in df.columns for col in ma_cols):
            return df.head(0)

        mask = (df["MA5"] > df["MA10"]) & (df["MA10"] > df["MA20"])
        if mask.all():
            return df
        return df.head(0)


class 量能多头过滤器(FilterPlugin):
    def __init__(self, days: int = 3):
        self.days = days
        self.name = f"量能多头_{days}日"
        self.desc = f"近{days}天量能多头（VOL_MA5 > VOL_MA10 > VOL_MA20）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        df = df.head(self.days).copy()
        vol_cols = ["VOL_MA5", "VOL_MA10", "VOL_MA20"]

        if not all(col in df.columns for col in vol_cols):
            return df.head(0)

        mask = (df["VOL_MA5"] > df["VOL_MA10"]) & (df["VOL_MA10"] > df["VOL_MA20"])
        if mask.all():
            return df
        return df.head(0)


class 涨幅过滤器(FilterPlugin):
    def __init__(self, days: int = 3, threshold: float = 5.0):
        self.days = days
        self.threshold = threshold
        self.name = f"涨幅_{days}日_{threshold}%"
        self.desc = f"近{days}天涨幅超过{threshold}%"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        recent = df.head(self.days)
        total_gain = (recent["涨跌幅"].sum())
        return df if total_gain > self.threshold else df.head(0)


class 过滤创业板(FilterPlugin):
    name = "过滤创业板"
    desc = "过滤创业板股票（300/301开头）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        return df.head(0) if code and (code.startswith("300") or code.startswith("301")) else df


class 过滤科创板(FilterPlugin):
    name = "过滤科创板"
    desc = "过滤科创板股票（688开头）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        return df.head(0) if code and code.startswith("688") else df


class 过滤北交所(FilterPlugin):
    name = "过滤北交所"
    desc = "过滤北交所股票（8开头或92开头）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        return df.head(0) if code and (code.startswith("8") or code.startswith("92")) else df


class 过滤ST(FilterPlugin):
    name = "过滤ST"
    desc = "过滤ST/*ST股票"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if not code or not stock_names:
            return df
        name = stock_names.get(code, "")
        return df.head(0) if name and ("ST" in name or "*ST" in name) else df


class 过滤数据不完整(FilterPlugin):
    name = "过滤数据不完整"
    desc = "过滤近N天数据不完整的股票（上市不足N天或存在交易日缺失）"
    _trading_calendar: pd.DataFrame = None

    def __init__(self, days: int = 15, min_trading_days: int = 12):
        self.days = days
        self.min_trading_days = min_trading_days
        self.name = f"过滤数据不完整_{days}日"
        self.desc = f"过滤近{days}天数据不完整的股票"

    @classmethod
    def _get_trading_calendar(cls) -> pd.DataFrame:
        if cls._trading_calendar is None:
            trading_df = ak.tool_trade_date_hist_sina()
            trading_df["trade_date"] = pd.to_datetime(trading_df["trade_date"])
            cls._trading_calendar = trading_df
        return cls._trading_calendar

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        df = df.copy()
        df["日期_dt"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期_dt", ascending=False).head(self.days)

        latest_date = df.iloc[0]["日期_dt"]
        df["日期_str"] = df["日期_dt"].apply(lambda x: x.strftime("%Y%m%d"))
        data_dates = set(df["日期_str"].tolist())

        trading_df = self._get_trading_calendar()
        trading_days = trading_df[trading_df["trade_date"] < latest_date].tail(self.days)["trade_date"].tolist()
        trading_set = set([d.strftime("%Y%m%d") for d in trading_days])

        missing_days = trading_set - data_dates
        if len(data_dates) < self.min_trading_days or len(missing_days) > (self.days - self.min_trading_days):
            return df.head(0)

        return df




def _load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    data = {}
    files = list(data_dir.glob("*.csv"))
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if not df.empty and "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期", ascending=False)
                data[f.stem] = df
        except Exception:
            continue
    return data


STOCK_LIST_FILE = get_stock_list_file()
ETF_LIST_FILE = get_etf_list_file()
STOCK_DATA_DIR = get_stock_data_dir()
ETF_DATA_DIR = get_etf_data_dir()
PICK_DIR = get_pick_dir()
FileUtil.ensure_dirs(PICK_DIR)


def _load_stock_names() -> Dict[str, str]:
    names = {}
    if STOCK_LIST_FILE.exists():
        df = pd.read_csv(STOCK_LIST_FILE, encoding="utf-8-sig", dtype={"代码": str})
        for _, row in df.iterrows():
            names[row["代码"]] = row["名称"]
    return names


def _load_etf_names() -> Dict[str, str]:
    names = {}
    if ETF_LIST_FILE.exists():
        df = pd.read_csv(ETF_LIST_FILE, encoding="utf-8-sig", dtype={"代码": str})
        for _, row in df.iterrows():
            names[row["代码"]] = row["名称"]
    return names


def _apply_filters(df: pd.DataFrame, code: str, filters: List[FilterPlugin], stock_names: Dict[str, str] = None) -> bool:
    for f in filters:
        if len(f.apply(df, code, stock_names)) == 0:
            return False
    return True


def pick_etf(filters: List[FilterPlugin]) -> pd.DataFrame:
    if not filters:
        return pd.DataFrame()
    data = _load_data(ETF_DATA_DIR)
    etf_names = _load_etf_names()
    results = []

    for code, df in data.items():
        if _apply_filters(df, code, filters):
            latest = df.iloc[0].to_dict()
            latest["代码"] = code
            latest["名称"] = etf_names.get(code, "")
            results.append(latest)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(axis=1, how="all")
    cols = ["代码", "名称"] + [c for c in result_df.columns if c not in ["代码", "名称", "日期", "股票代码"]]
    result_df = result_df[cols]
    return result_df.sort_values("涨跌幅", ascending=False)


def pick_stock(filters: List[FilterPlugin]) -> pd.DataFrame:
    if not filters:
        return pd.DataFrame()
    data = _load_data(STOCK_DATA_DIR)
    stock_names = _load_stock_names()
    results = []

    for code, df in data.items():
        if _apply_filters(df, code, filters, stock_names):
            latest = df.iloc[0].to_dict()
            latest["代码"] = code
            latest["名称"] = stock_names.get(code, "")
            results.append(latest)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(axis=1, how="all")
    cols = ["代码", "名称"] + [c for c in result_df.columns if c not in ["代码", "名称", "日期", "股票代码"]]
    result_df = result_df[cols]
    return result_df.sort_values("涨跌幅", ascending=False)


def _apply_filters_with_log(
    data: Dict[str, pd.DataFrame],
    filters: List[FilterPlugin],
    names: Dict[str, str] = None
) -> pd.DataFrame:
    if not filters:
        return pd.DataFrame()

    remaining_codes = set(data.keys())

    for f in filters:
        filtered_codes = set()
        for code in remaining_codes:
            if code in data:
                df = data[code]
                if len(f.apply(df, code, names)) == 0:
                    filtered_codes.add(code)

        before_count = len(remaining_codes)
        remaining_codes = remaining_codes - filtered_codes
        after_count = len(remaining_codes)
        filtered_count = before_count - after_count

        if filtered_codes:
            sample = list(filtered_codes)[:5]
            sample_with_name = [f"{c}({names.get(c, '')})" if names else c for c in sample]
            print(f"  {f.desc}: {before_count}个->{after_count}个={filtered_count}个, 样例: {', '.join(sample_with_name)}")
        else:
            print(f"  {f.desc}: {before_count}个->{after_count}个={filtered_count}个")

    results = []
    for code in remaining_codes:
        if code in data:
            latest = data[code].iloc[0].to_dict()
            latest["代码"] = code
            latest["名称"] = names.get(code, "") if names else ""
            results.append(latest)

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.dropna(axis=1, how="all")
    cols = ["代码", "名称"] + [c for c in result_df.columns if c not in ["代码", "名称", "日期", "股票代码"]]
    result_df = result_df[cols]
    return result_df.sort_values("涨跌幅", ascending=False)


def run_pick(
    etf_filters: List[FilterPlugin] = None,
    stock_filters: List[FilterPlugin] = None
) -> tuple:
    today = date.today().strftime("%Y%m%d")
    etf_df = pd.DataFrame()
    stock_df = pd.DataFrame()

    if etf_filters:
        print("\n=== ETF 筛选 ===")
        etf_data = _load_data(ETF_DATA_DIR)
        etf_names = _load_etf_names()
        print(f"  ETF初始数量: {len(etf_data)}")
        etf_df = _apply_filters_with_log(etf_data, etf_filters, etf_names)

        if not etf_df.empty:
            etf_file = PICK_DIR / f"etf_{today}.csv"
            etf_df.to_csv(etf_file, index=False, encoding="utf-8-sig")
            print(f"  ETF最终结果: {len(etf_df)} 只, 已保存到: {etf_file}")
        else:
            print("  无符合条件的ETF")

    if stock_filters:
        print("\n=== 股票 筛选 ===")
        stock_data = _load_data(STOCK_DATA_DIR)
        stock_names = _load_stock_names()
        print(f"  股票初始数量: {len(stock_data)}")
        stock_df = _apply_filters_with_log(stock_data, stock_filters, stock_names)

        if not stock_df.empty:
            stock_file = PICK_DIR / f"stock_{today}.csv"
            stock_df.to_csv(stock_file, index=False, encoding="utf-8-sig")
            print(f"  股票最终结果: {len(stock_df)} 只, 已保存到: {stock_file}")
        else:
            print("  无符合条件的股票")

    return etf_df, stock_df

def _run_pick():
    etf_result, stock_result = run_pick(
        etf_filters=[
            多头排列过滤器(days=3),
            涨幅过滤器(days=3, threshold=5.0),
            涨幅过滤器(days=7, threshold=10.0),
        ],
        stock_filters=[
            过滤创业板(),
            过滤科创板(),
            过滤北交所(),
            过滤ST(),
            过滤数据不完整(days=15),
            多头排列过滤器(days=3),
            量能多头过滤器(days=3),
            涨幅过滤器(days=1, threshold=3),
            涨幅过滤器(days=3, threshold=10.0),
            涨幅过滤器(days=5, threshold=20.0),
        ]
    )



if __name__ == "__main__":
    _run_pick()
