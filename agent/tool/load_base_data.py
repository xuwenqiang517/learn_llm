import akshare as ak
import sys
from pathlib import Path
import pandas as pd
from datetime import date
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil,print_green,print_red
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



def _get_trading_days(days: int = 15) -> list:
    df = ak.tool_trade_date_hist_sina()
    trading_days = df[df["trade_date"] <= date.today()].tail(days)["trade_date"].tolist()
    print_green(f"获取交易日列表成功: {len(trading_days)} 条")
    return [str(day).replace("-","") for day in trading_days]


def _update_etf_list() -> None:
    try: 
        print_green("开始更新 ETF 列表...")
        df = ak.fund_etf_spot_em()
        # 将总市值从元转换为亿元，并添加"亿"单位，保留2位小数
        if "总市值" in df.columns:
            df["总市值"] = (df["总市值"] / 100000000).round(2).astype(str) + "亿"
        df[["代码", "名称", "总市值", "换手率", "量比"]].to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
        print_green(f"ETF 列表已保存到: {ETF_LIST_FILE}")
    except Exception as e:
        print_red(f"更新 ETF 列表失败: {e}")

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

def _update_code_list() -> None:
    try:
        # 创建空的结果DataFrame
        big_df = pd.DataFrame()
        
        # 获取上海主板A股
        logger.info("获取上海主板A股...")
        stock_sh = ak.stock_info_sh_name_code(symbol="主板A股")
        if not stock_sh.empty:
            stock_sh = stock_sh[["证券代码", "证券简称"]].copy()  # 使用copy避免视图警告
            stock_sh["type"] = "上海主板"
            stock_sh.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_sh], ignore_index=True)
        
        # 获取深圳A股
        logger.info("获取深圳A股...")
        stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
        if not stock_sz.empty:
            stock_sz = stock_sz.copy()  # 使用copy避免视图警告
            stock_sz["A股代码"] = stock_sz["A股代码"].astype(str).str.zfill(6)
            
            # 分离深圳主板和创业板
            # 创业板股票代码以300开头
            # 深圳主板股票代码以000开头
            # 中小板股票代码以002开头
            
            # 深圳主板（000开头）
            stock_sz_main = stock_sz[stock_sz["A股代码"].str.startswith("000")][["A股代码", "A股简称"]].copy()
            if not stock_sz_main.empty:
                stock_sz_main["type"] = "深圳主板"
                stock_sz_main.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_main], ignore_index=True)
            
            # 中小板（002开头）
            stock_sz_sme = stock_sz[stock_sz["A股代码"].str.startswith("002")][["A股代码", "A股简称"]].copy()
            if not stock_sz_sme.empty:
                stock_sz_sme["type"] = "中小板"
                stock_sz_sme.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_sme], ignore_index=True)
            
            # 创业板（300开头）
            stock_sz_gem = stock_sz[stock_sz["A股代码"].str.startswith("300")][["A股代码", "A股简称"]].copy()
            if not stock_sz_gem.empty:
                stock_sz_gem["type"] = "创业板"
                stock_sz_gem.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_gem], ignore_index=True)
            
            # 创业板注册制（301开头）
            stock_sz_gem_reg = stock_sz[stock_sz["A股代码"].str.startswith("301")][["A股代码", "A股简称"]].copy()
            if not stock_sz_gem_reg.empty:
                stock_sz_gem_reg["type"] = "创业板"
                stock_sz_gem_reg.columns = ["代码", "名称", "板块类型"]
                big_df = pd.concat([big_df, stock_sz_gem_reg], ignore_index=True)
        
        # 获取科创板
        logger.info("获取科创板...")
        stock_kcb = ak.stock_info_sh_name_code(symbol="科创板")
        if not stock_kcb.empty:
            stock_kcb = stock_kcb[["证券代码", "证券简称"]].copy()  # 使用copy避免视图警告
            stock_kcb["type"] = "科创板"
            stock_kcb.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_kcb], ignore_index=True)
        
        # 获取北京股
        logger.info("获取北京股...")
        stock_bse = ak.stock_info_bj_name_code()
        if not stock_bse.empty:
            stock_bse = stock_bse[["证券代码", "证券简称"]].copy()  # 使用copy避免视图警告
            stock_bse["type"] = "北京股"
            stock_bse.columns = ["代码", "名称", "板块类型"]
            big_df = pd.concat([big_df, stock_bse], ignore_index=True)
        
        # 保存结果
        big_df.to_csv(BASE_DATA_DIR / "stock_list.csv", index=False, encoding="utf-8-sig")
        logger.info(f"股票列表已更新（{len(big_df)}只）")
        logger.info(f"各类股票数量:")
        logger.info(big_df["板块类型"].value_counts())
    except Exception as e:
        logger.error(f"更新股票列表失败: {e}")
        raise

def _update_all_data():
    _update_etf_list()
    _update_code_list()

    _update_daily_data(
        ETF_LIST_FILE,
        ETF_DATA_DIR,
        "更新 ETF 日线数据",
        ak.fund_etf_hist_em
    )
    _update_daily_data(
        STOCK_LIST_FILE,
        STOCK_DATA_DIR,
        "更新 股票 日线数据",
        ak.stock_zh_a_hist
    )

if __name__ == "__main__":
    _update_all_data()
