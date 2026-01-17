from pathlib import Path
from datetime import date, datetime
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"

BASE_DATA_DIR = DATA_DIR / "base"
ETF_DATA_DIR = DATA_DIR / "etf_data"
STOCK_DATA_DIR = DATA_DIR / "stock_data"
PICK_DIR = DATA_DIR / "pick"
MSG_DIR = DATA_DIR / "msg"
ANALYZER_DIR = DATA_DIR / "analyzer"
CACHE_DIR = DATA_DIR / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_base_dir() -> Path:
    return BASE_DIR


def get_temp_dir() -> Path:
    return TEMP_DIR


def get_data_dir() -> Path:
    return DATA_DIR


def get_base_data_dir() -> Path:
    return BASE_DATA_DIR


def get_etf_data_dir() -> Path:
    return ETF_DATA_DIR


def get_stock_data_dir() -> Path:
    return STOCK_DATA_DIR


def get_pick_dir() -> Path:
    return PICK_DIR


def get_msg_dir() -> Path:
    return MSG_DIR


def get_analyzer_dir() -> Path:
    return ANALYZER_DIR


def get_stock_spot_cache_file() -> Path:
    return CACHE_DIR / "stock_spot.pkl"


def get_etf_spot_cache_file() -> Path:
    return CACHE_DIR / "etf_spot.pkl"


def get_cache_dir() -> Path:
    return CACHE_DIR


def get_industry_cache_file() -> Path:
    return CACHE_DIR / "industry_mapping.pkl"


def get_lrb_cache_file() -> Path:
    return CACHE_DIR / "lrb_data.pkl"


def get_last_trading_day() -> str:
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df["trade_date"] <= date.today()]["trade_date"].tolist()
    if not trading_days:
        return date.today().strftime("%Y%m%d")
    last_day = max(trading_days)
    return last_day.strftime("%Y%m%d")


def _delete_old_list_files(base_name: str, keep_date: str) -> None:
    pattern = f"{base_name}_*.csv"
    for f in BASE_DATA_DIR.glob(pattern):
        if f.stem.replace(f"{base_name}_", "") != keep_date:
            try:
                f.unlink()
            except Exception:
                pass


def _is_same_day_data(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    today = date.today()
    last_trading = get_last_trading_day()
    return file_mtime.date() == today


def get_etf_list_file() -> Path:
    last_trading = get_last_trading_day()
    return BASE_DATA_DIR / f"etf_list_{last_trading}.csv"


def get_stock_list_file() -> Path:
    last_trading = get_last_trading_day()
    return BASE_DATA_DIR / f"stock_list_{last_trading}.csv"


def get_pick_etf_file(today: str = None) -> Path:
    if today is None:
        today = date.today().strftime("%Y%m%d")
    return PICK_DIR / f"etf_{today}.csv"


def get_pick_stock_file(today: str = None) -> Path:
    if today is None:
        today = date.today().strftime("%Y%m%d")
    return PICK_DIR / f"stock_{today}.csv"


def get_message_file(today: str = None) -> Path:
    if today is None:
        today = date.today().strftime("%Y%m%d")
    return MSG_DIR / f"message_{today}.json"


def get_stock_data_file(code: str) -> Path:
    return STOCK_DATA_DIR / f"{code}.csv"


def get_etf_data_file(code: str) -> Path:
    return ETF_DATA_DIR / f"{code}.csv"


def get_analyzer_file(name: str) -> Path:
    return ANALYZER_DIR / name


def read_csv_with_code(path: Path, code_col: str = "代码") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", dtype={code_col: str})


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_pick_etf(today: str = None) -> pd.DataFrame:
    path = get_pick_etf_file(today)
    return read_csv_with_code(path)


def read_pick_stock(today: str = None) -> pd.DataFrame:
    path = get_pick_stock_file(today)
    return read_csv_with_code(path)


def read_message(today: str = None) -> dict:
    path = get_message_file(today)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def read_etf_list() -> pd.DataFrame:
    return read_csv_with_code(get_etf_list_file())


def read_stock_list() -> pd.DataFrame:
    return read_csv_with_code(get_stock_list_file())


def read_stock_data(code: str) -> pd.DataFrame:
    path = get_stock_data_file(code)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def read_etf_data(code: str) -> pd.DataFrame:
    path = get_etf_data_file(code)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _is_cache_expired(file_path: Path, days: int = 7) -> bool:
    if not file_path.exists():
        return True
    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
    return (datetime.now() - file_mtime).days > days


def load_cache(file_path: Path):
    if not file_path.exists():
        return None
    try:
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cache(data, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"保存缓存失败: {e}")


def get_missing_dates(local_dates: set, remote_days: list) -> list:
    remote_set = set(remote_days)
    return [d for d in remote_days if d not in local_dates]
