import akshare as ak
import sys
from pathlib import Path
from datetime import date
from typing import List, Callable, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.data_path_util import (
    get_stock_list_file, get_etf_list_file, get_stock_data_dir, get_etf_data_dir, get_pick_dir
)
from utils.file_util import FileUtil


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

        df = df.head(self.days)
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

        df = df.head(self.days)
        vol_cols = ["VOL_MA5", "VOL_MA10", "VOL_MA20"]

        if not all(col in df.columns for col in vol_cols):
            return df.head(0)

        mask = (df["VOL_MA5"] > df["VOL_MA10"]) & (df["VOL_MA10"] > df["VOL_MA20"])
        if mask.all():
            return df
        return df.head(0)


class 涨幅过滤器(FilterPlugin):
    def __init__(self, days: int = 3, threshold: float = 5.0, mode: str = "min"):
        self.days = days
        self.threshold = threshold
        self.mode = mode
        if mode == "min":
            self.name = f"涨幅_{days}日_{threshold}%"
            self.desc = f"近{days}天涨幅超过{threshold}%"
        else:
            self.name = f"涨幅_{days}日<{threshold}%"
            self.desc = f"近{days}天涨幅低于{threshold}%"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        recent = df.head(self.days)
        total_gain = (recent["涨跌幅"].sum())
        if self.mode == "min":
            return df if total_gain > self.threshold else df.head(0)
        else:
            return df if total_gain < self.threshold else df.head(0)


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


class RSI过滤器(FilterPlugin):
    def __init__(self, mode: str = "超卖", threshold: float = 30):
        self.mode = mode
        self.threshold = threshold
        self.name = f"RSI_{mode}_{threshold}"
        if mode == "超卖":
            self.desc = f"RSI < {threshold}（超卖）"
        elif mode == "超买":
            self.desc = f"RSI > {threshold}（超买）"
        else:
            self.desc = f"RSI {mode}"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "RSI" not in df.columns or df.empty:
            return df.head(0)

        latest_rsi = df.iloc[0]["RSI"]
        if pd.isna(latest_rsi):
            return df.head(0)

        if self.mode == "超卖" and latest_rsi < self.threshold:
            return df
        elif self.mode == "超买" and latest_rsi > self.threshold:
            return df
        return df.head(0)


class MACD过滤器(FilterPlugin):
    def __init__(self, mode: str = "金叉"):
        self.mode = mode
        self.name = f"MACD_{mode}"
        if mode == "金叉":
            self.desc = "MACD金叉（DIF > DEA）"
        elif mode == "红柱":
            self.desc = "MACD红柱（MACD > 0）"
        else:
            self.desc = f"MACD {mode}"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "DIF" not in df.columns or "DEA" not in df.columns or df.empty:
            return df.head(0)

        latest_dif = df.iloc[0]["DIF"]
        latest_dea = df.iloc[0]["DEA"]
        latest_macd = df.iloc[0].get("MACD", 0)

        if pd.isna(latest_dif) or pd.isna(latest_dea):
            return df.head(0)

        if self.mode == "金叉" and latest_dif > latest_dea:
            return df
        elif self.mode == "红柱" and latest_macd > 0:
            return df
        return df.head(0)


class 涨停过滤器(FilterPlugin):
    def __init__(self, days: int = 5):
        self.days = days
        self.name = f"近{days}日涨停"
        self.desc = f"近{days}日有涨停（涨跌幅 >= 9.9%）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "涨跌幅" not in df.columns or df.empty:
            return df.head(0)

        df = df.head(self.days)
        limit_up = df[df["涨跌幅"] >= 9.9]
        if len(limit_up) > 0:
            return df
        return df.head(0)


class 市值过滤器(FilterPlugin):
    def __init__(self, min_cap: float = None, max_cap: float = None, unit: str = "亿"):
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.unit = unit
        self.name = f"市值过滤器"
        if min_cap and max_cap:
            self.desc = f"市值 {min_cap}-{max_cap}{unit}"
        elif min_cap:
            self.desc = f"市值 > {min_cap}{unit}"
        elif max_cap:
            self.desc = f"市值 < {max_cap}{unit}"
        else:
            self.desc = "市值过滤"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "总市值" not in df.columns or df.empty:
            return df

        market_cap_str = df.iloc[0]["总市值"]
        if not market_cap_str or pd.isna(market_cap_str):
            return df.head(0)

        try:
            market_cap_str = str(market_cap_str).replace("亿", "").replace(",", "")
            market_cap = float(market_cap_str)
        except:
            return df.head(0)

        if self.min_cap and market_cap < self.min_cap:
            return df.head(0)
        if self.max_cap and market_cap > self.max_cap:
            return df.head(0)
        return df


class 换手率过滤器(FilterPlugin):
    def __init__(self, threshold: float = 20.0):
        self.threshold = threshold
        self.name = f"换手率>{threshold}%"
        self.desc = f"换手率 > {threshold}%"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "换手率" not in df.columns or df.empty:
            return df.head(0)

        latest_turnover = df.iloc[0]["换手率"]
        if pd.isna(latest_turnover):
            return df.head(0)

        if latest_turnover > self.threshold:
            return df
        return df.head(0)


class PE过滤器(FilterPlugin):
    def __init__(self, min_pe: float = 0):
        self.min_pe = min_pe
        self.name = f"PE>{min_pe}"
        self.desc = f"PE > {min_pe}"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "PE" not in df.columns or df.empty:
            return df.head(0)

        latest_pe = df.iloc[0]["PE"]
        if pd.isna(latest_pe):
            return df.head(0)

        if latest_pe > self.min_pe:
            return df
        return df.head(0)


class 市值下限过滤器(FilterPlugin):
    def __init__(self, min_cap: float = 100.0, unit: str = "亿"):
        self.min_cap = min_cap
        self.unit = unit
        self.name = f"市值>{min_cap}{unit}"
        self.desc = f"市值 > {min_cap}{unit}"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "总市值" not in df.columns or df.empty:
            return df.head(0)

        market_cap_str = df.iloc[0]["总市值"]
        if not market_cap_str or pd.isna(market_cap_str):
            return df.head(0)

        try:
            market_cap_str = str(market_cap_str).replace("亿", "").replace(",", "")
            market_cap = float(market_cap_str)
        except:
            return df.head(0)

        if market_cap > self.min_cap:
            return df
        return df.head(0)


class 涨跌幅范围过滤器(FilterPlugin):
    def __init__(self, min_pct: float = 3.0, max_pct: float = 5.0):
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.name = f"涨跌幅{min_pct}%-{max_pct}%"
        self.desc = f"涨跌幅在{min_pct}%~{max_pct}%之间"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "涨跌幅" not in df.columns or df.empty:
            return df.head(0)

        latest_change = df.iloc[0]["涨跌幅"]
        if pd.isna(latest_change):
            return df.head(0)

        try:
            change_str = str(latest_change).replace("%", "")
            change = float(change_str)
        except:
            return df.head(0)

        if self.min_pct <= change <= self.max_pct:
            return df
        return df.head(0)


class 过滤涨停(FilterPlugin):
    name = "过滤涨停"
    desc = "过滤当天涨停的股票（涨跌幅 >= 9.9%）"

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if "涨跌幅" not in df.columns or df.empty:
            return df

        latest_change = df.iloc[0]["涨跌幅"]
        if pd.isna(latest_change):
            return df

        if latest_change >= 9.9:
            return df.head(0)

        return df


class 过滤数据不完整(FilterPlugin):
    name = "过滤数据不完整"
    desc = "过滤近N天数据不完整的股票（上市不足N天或存在交易日缺失）"
    _trading_calendar: Set[str] = None
    _trading_calendar_loaded = False
    _trading_calendar_list: List[str] = None

    def __init__(self, days: int = 15, min_trading_days: int = 12):
        self.days = days
        self.min_trading_days = min_trading_days
        self.name = f"过滤数据不完整_{days}日"
        self.desc = f"过滤近{days}天数据不完整的股票"

    @classmethod
    def _get_trading_calendar(cls) -> Set[str]:
        if cls._trading_calendar_loaded:
            return cls._trading_calendar
        
        import akshare as ak
        trading_df = ak.tool_trade_date_hist_sina()
        cls._trading_calendar = set()
        cls._trading_calendar_list = []
        for dt in trading_df["trade_date"]:
            if isinstance(dt, str):
                date_str = dt.replace("-", "")[:8]
            else:
                date_str = dt.strftime("%Y%m%d")
            if date_str not in cls._trading_calendar:
                cls._trading_calendar.add(date_str)
                cls._trading_calendar_list.append(date_str)
        cls._trading_calendar_loaded = True
        return cls._trading_calendar

    def apply(self, df: pd.DataFrame, code: str = None, stock_names: Dict[str, str] = None) -> pd.DataFrame:
        if len(df) < self.days:
            return df.head(0)

        df_head = df.head(self.days + 10)
        dates_list = df_head["日期"].tolist()
        
        if not dates_list:
            return df.head(0)
        
        if isinstance(dates_list[0], str):
            date_strs = [d[:10].replace("-", "") for d in dates_list]
        else:
            date_strs = [d.strftime("%Y%m%d") for d in dates_list]
        
        latest_date_str = date_strs[0]
        data_dates = set(date_strs[:self.days])

        trading_list = self._get_trading_calendar_list()
        
        needed_dates = set()
        for d in reversed(trading_list):
            if d <= latest_date_str:
                needed_dates.add(d)
                if len(needed_dates) >= self.days:
                    break
        
        missing_days = needed_dates - data_dates
        if len(data_dates) < self.min_trading_days or len(missing_days) > (self.days - self.min_trading_days):
            return df.head(0)

        return df

    @classmethod
    def _get_trading_calendar_list(cls) -> List[str]:
        if cls._trading_calendar_list is None:
            cls._get_trading_calendar()
        return cls._trading_calendar_list




def _load_data(data_dir: Path, target_date: str = None) -> Dict[str, pd.DataFrame]:
    data = {}
    files = list(data_dir.glob("*.csv"))
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            if not df.empty and "日期" in df.columns:
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期", ascending=False)
                if target_date:
                    target_dt = pd.to_datetime(target_date)
                    df = df[df["日期"] <= target_dt]
                if not df.empty:
                    data[f.stem] = df
        except Exception:
            continue
    return data


def _get_available_trading_dates(data_dir: Path, limit: int = 30) -> List[str]:
    dates = set()
    files = list(data_dir.glob("*.csv"))
    exclude_prefixes = ("300", "301", "688", "8", "92")
    for f in files:
        try:
            code = f.stem
            if code.startswith(exclude_prefixes):
                continue
            df = pd.read_csv(f, encoding="utf-8-sig", usecols=["日期"], nrows=limit)
            if "日期" in df.columns and len(df) > 0:
                for d in df["日期"]:
                    dates.add(str(d)[:10])
        except Exception:
            continue
    return sorted(dates, reverse=True)


def _get_data_for_date(df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
    if target_date is None:
        return df
    target_dt = pd.to_datetime(target_date)
    filtered = df[df["日期"] <= target_dt]
    return filtered


STOCK_FILTERS = [
    过滤创业板(),
    过滤科创板(),
    过滤北交所(),
    过滤ST(),
    过滤数据不完整(days=15),
    多头排列过滤器(days=3),
    量能多头过滤器(days=3),
    换手率过滤器(threshold=5.0),
    PE过滤器(min_pe=0),
    市值下限过滤器(min_cap=50.0),
]

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
        stock_filters=STOCK_FILTERS,
    )


if __name__ == "__main__":
    _run_pick()
