import akshare as ak
import sys
from pathlib import Path
from numpy import void
import pandas as pd
from datetime import date
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil,print_green,print_red,print_yellow
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


def _update_etf_list() -> void:
    if not _is_cache_valid(ETF_LIST_FILE):
        try: 
            print_green("开始更新 ETF 列表...")
            df = ak.fund_etf_category_sina(symbol="ETF基金")
            print_green(f"更新 ETF 列表成功: {df.shape[0]} 条")
            df[['代码', '名称']].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
            print_green(f"ETF 列表已保存到: {ETF_LIST_FILE}")
        except Exception as e:
            print_red(f"更新 ETF 列表失败: {e}")
    else:
        print_green("ETF 列表已存在")

def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df['trade_date'] <= date.today()].tail(days)['trade_date'].tolist()
    print_green(f"获取交易日列表成功: {len(trading_days)} 条")
    return [str(day) for day in trading_days]

# 根据_update_etf_list和_get_trading_days更新ETF日线数据，数据用csv保存
def _update_daily_etf_data() -> void:
    etf_list_df = pd.read_csv(ETF_LIST_FILE, encoding="utf-8-sig")
    trading_days = _get_trading_days()
    for etf_code, etf_name in tqdm(etf_list_df[['代码', '名称']].values, desc="更新 ETF 日线数据", unit="条", total=etf_list_df.shape[0]):
        for date in trading_days:
            cache_file = DAILY_DATA_DIR / date / f"{etf_code}.csv"
            print(cache_file)
            if not _is_cache_valid(cache_file):
                try:
                    print_green(f"开始更新 ETF 日线数据: {etf_code}_{etf_name}_{date}")
                    etf_daily_df = ak.fund_etf_fund_daily_em(symbol=etf_code, date=date)
                    etf_daily_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
                    print_green(f"更新 ETF 日线数据成功: {etf_code}_{etf_name}_{date}")
                except Exception as e:
                    print_red(f"更新 ETF 日线数据失败: {e}")


if __name__ == "__main__":
    _update_etf_list()
    _get_trading_days()
    _update_daily_etf_data()