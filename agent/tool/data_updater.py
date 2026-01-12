import akshare as ak
import sys
from pathlib import Path
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
ETF_DATA_DIR = DATA_DIR / "etf_data"
STOCK_DATA_DIR = DATA_DIR / "stock_data"

FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ETF_DATA_DIR, STOCK_DATA_DIR)

ETF_LIST_FILE = BASE_DATA_DIR / "etf_list.csv"
STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.csv"
CACHE_EXPIRY_DAYS = 1


def _is_cache_valid(cache_file: Path) -> bool:
    return cache_file.exists() and FileUtil.is_cache_valid(cache_file, CACHE_EXPIRY_DAYS)

def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df["trade_date"] <= date.today()].tail(days)["trade_date"].tolist()
    print_green(f"获取交易日列表成功: {len(trading_days)} 条")
    return [str(day).replace("-","") for day in trading_days]


def _update_etf_list() -> None:
    if not _is_cache_valid(ETF_LIST_FILE):
        try: 
            print_green("开始更新 ETF 列表...")
            df = ak.fund_etf_spot_em()
            df[["代码", "名称"]].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
            print_green(f"ETF 列表已保存到: {ETF_LIST_FILE}")
        except Exception as e:
            print_red(f"更新 ETF 列表失败: {e}")
    else:
        print_green("ETF 列表已存在")

def _update_daily_etf_data() -> None:
    trading_days = _get_trading_days(15)
    etf_list_df = pd.read_csv(ETF_LIST_FILE, encoding="utf-8-sig", dtype={"代码": str})
    for etf_code, etf_name in tqdm(etf_list_df[["代码", "名称"]].values, desc="更新 ETF 日线数据", unit="条", total=etf_list_df.shape[0]):
        cache_file = ETF_DATA_DIR / f"{etf_code}.csv"
        if _is_cache_valid(cache_file):
            continue
        fund_etf_hist_sina_df = ak.fund_etf_hist_em(symbol=etf_code, start_date=trading_days[0], end_date=trading_days[-1])
        if not fund_etf_hist_sina_df.empty:
            fund_etf_hist_sina_df.to_csv(ETF_DATA_DIR / f"{etf_code}.csv", index=False, encoding="utf-8-sig")
        else:
            print_red(f"ETF 日线数据为空: {etf_code}_{etf_name}")
    print_green("更新 ETF 日线数据完成")
            
            
def _update_code_list() -> None:
    try:
        stock_list = ak.stock_info_a_code_name()
        stock_list.to_csv(STOCK_LIST_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"股票列表已更新（{len(stock_list)}只）")
    except Exception as e:
        logger.error(f"更新股票列表失败: {e}")

def _update_daily_stock_data() -> None:
    trading_days = _get_trading_days(15)
    stock_list_df = pd.read_csv(STOCK_LIST_FILE, encoding="utf-8-sig", dtype={"code": str})
    for stock_code, stock_name in tqdm(stock_list_df[["code", "name"]].values, desc="更新 股票 日线数据", unit="条", total=stock_list_df.shape[0]):
        cache_file = STOCK_DATA_DIR / f"{stock_code}.csv"
        if _is_cache_valid(cache_file):
            continue
        stock_hist_sina_df = ak.stock_zh_a_hist(symbol=stock_code,start_date=trading_days[0], end_date=trading_days[-1])
        if not stock_hist_sina_df.empty:
            stock_hist_sina_df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        else:
            print_yellow(f"股票 日线数据为空: {stock_code}_{stock_name}")
    print_green("更新 股票 日线数据完成")

if __name__ == "__main__":
    _update_etf_list()
    _update_daily_etf_data()
    _update_code_list()
    _update_daily_stock_data()
    
