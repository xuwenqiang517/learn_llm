import akshare as ak
import sys
from pathlib import Path
import pandas as pd
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil
from utils.file_util import FileUtil

logger = LogUtil.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
DAILY_DATA_DIR = DATA_DIR / "daily"

FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, DAILY_DATA_DIR)

ETF_LIST_FILE = BASE_DATA_DIR / "etf_list.csv"
CACHE_EXPIRY_DAYS = 1


def _is_cache_valid(cache_file: Path) -> bool:
    return cache_file.exists() and FileUtil.is_cache_valid(cache_file, CACHE_EXPIRY_DAYS)


def _update_etf_list():
    if not _is_cache_valid(ETF_LIST_FILE):
        try:
            ak.fund_etf_category_sina(symbol="ETF基金")[['代码', '名称']].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.error(f"更新 ETF 列表失败: {e}")

def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    return df[df['trade_date'] <= date.today()].tail(days)['trade_date'].tolist()

# def _update_daily_etf_data():
#     try:
#         etf_list_df = pd.read_csv(ETF_LIST_FILE, encoding="utf-8-sig")
#         for etf_code, etf_name in etf_list_df[['代码', '名称']].values:
#             etf_daily_df = ak.fund_etf_daily_sina(symbol=etf_code)
#             etf_daily_df.to_csv(DAILY_DATA_DIR / f"{etf_code}_{etf_name}.csv", index=False, encoding="utf-8-sig")
#     except Exception as e:
#         logger.error(f"更新 ETF 数据失败: {e}")

def update_etf_data():
    _update_etf_list()


if __name__ == "__main__":
    print("=" * 70)
    print("ETF 数据更新工具")
    print("=" * 70)
    update_etf_data()
    print("=" * 70)
    print(_get_trading_days())
