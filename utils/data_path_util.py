from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = PROJECT_DIR / ".temp" / "data"

def get_base_data_dir() -> Path:
    return BASE_DATA_DIR

def get_stock_data_dir() -> Path:
    return BASE_DATA_DIR / "stock_data"

def get_etf_data_dir() -> Path:
    return BASE_DATA_DIR / "etf_data"

def get_pick_dir() -> Path:
    return BASE_DATA_DIR / "pick"

def get_stock_list_file() -> Path:
    return BASE_DATA_DIR / "base" / "stock_list_20260116.csv"

def get_etf_list_file() -> Path:
    return BASE_DATA_DIR / "base" / "etf_list_20260116.csv"

def get_industry_cache_file() -> Path:
    return get_base_data_dir() / "industry_cache.csv"

def get_lrb_cache_file() -> Path:
    return get_base_data_dir() / "lrb_cache.csv"

def get_stock_spot_cache_file() -> Path:
    return get_base_data_dir() / "stock_spot_cache.csv"

def get_etf_spot_cache_file() -> Path:
    return get_base_data_dir() / "etf_spot_cache.csv"

def get_last_trading_day() -> str:
    import pandas as pd
    import akshare as ak
    df = ak.tool_trade_date_hist_sina()
    return df.iloc[0]["trade_date"]

def _delete_old_list_files():
    pass

def _is_same_day_data(file_path: Path) -> bool:
    return False

def _is_cache_expired(cache_file: Path, hours: int = 24) -> bool:
    return True

def load_cache(cache_file: Path) -> dict:
    return {}

def save_cache(cache_file: Path, data: dict):
    pass
