from typing import Any, Dict
import akshare as ak
import pandas as pd
from tqdm import tqdm
from StockDayData import StockDayData
from cache_utils import *
from datetime import datetime
import numpy as np
import time


def get_stock_list() -> pd.DataFrame:
    print("获取股票数据")
    stock_sh = ak.stock_info_sh_name_code(symbol="主板A股")
    stock_sh = stock_sh[["证券代码", "证券简称"]]
    stock_sh.columns = ["代码", "名称"]

    stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
    stock_sz = stock_sz[["A股代码", "A股简称"]]
    stock_sz.columns = ["代码", "名称"]
    
    all = stock_sh._append(stock_sz, ignore_index=True)

    all = all[~all["代码"].str.startswith(("688", "300", "301"))]
    all = all[~all["名称"].str.contains("ST")]
    print(f"获取股票数量：{len(all)}")
    return all


def get_stock_his2(code_name_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    all_data = []
    for row in tqdm(code_name_df.itertuples(), total=len(code_name_df), desc="下载进度"):
        try:
            stock_df = get_with_cache_feather(f"{row.代码}_{start}_{end}.feather", lambda: ak.stock_zh_a_hist(symbol=row.代码, period="daily", start_date=start, end_date=end), sleep_time=1)
            stock_df.insert(2, "名称", row.名称)
            stock_df["日期"] = pd.to_datetime(stock_df["日期"]).dt.strftime("%Y%m%d")
            all_data.append(stock_df)
        except Exception as e:
            print(f"获取股票 {row.代码} 数据失败: {e}")
            break
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        return result_df
    return pd.DataFrame()    



def calc_tech(df: pd.DataFrame) -> pd.DataFrame:
    # ma5 = df["收盘"].rolling(window=5).mean().round(2)
    # ma10 = df["收盘"].rolling(window=10).mean().round(2)
    # ma20 = df["收盘"].rolling(window=20).mean().round(2)
    # vol_ma5 = df["成交量"].rolling(window=5).mean().round(2)
    # vol_ma10 = df["成交量"].rolling(window=10).mean().round(2)
    # vol_ma20 = df["成交量"].rolling(window=20).mean().round(2)
    df["change_3d"] = df["涨跌幅"].rolling(window=3).sum().round(2)
    df["change_5d"] = df["涨跌幅"].rolling(window=5).sum().round(2)
    df["change_10d"] = df["涨跌幅"].rolling(window=10).sum().round(2)
    
    df["consecutive_up_days"] = calc_up_days(df["收盘"].values)
    df["vol_rank"] = df["成交量"].rank(ascending=False, method="min").astype(int)
    return df
    
    
    
def calc_up_days(close_prices: np.ndarray) -> np.ndarray:
    consecutive_up = []
    count = 0
    for i in range(len(close_prices)):
        if i == 0:
            consecutive_up.append(0)
            count = 0
        else:
            if close_prices[i] > close_prices[i - 1]:
                count += 1
            else:
                count = 0
            consecutive_up.append(count)
    return consecutive_up


def calc_basic_indicators(dict: dict):
    stock_zh_a_spot_em_df = get_with_cache("stock_zh_a_spot_em.pkl", ak.stock_zh_a_spot_em)
    for row in stock_zh_a_spot_em_df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值"]].values:
        code = row[0]
        if code not in dict:
            continue
        df = dict[code]
        df["pe"] = row[1]
        df["pb"] = row[2]
        df["market_cap"] = int(row[3] / 100000000)
        df["circulating_market_cap"] = int(row[4] / 100000000)
    return dict



def build_index(df: pd.DataFrame) -> Dict[str, Dict[str, StockDayData]]:
    rs: Dict[str, Dict[str, StockDayData]] = {}
    for code, group in df.groupby("股票代码"):
        records = group.to_dict('records')
        rs[code] = {r['日期']: StockDayData(
            date=r['日期'], code=r['股票代码'], name=r['名称'],
            open=r['开盘'], close=r['收盘'], high=r['最高'], low=r['最低'],
            change_pct=r['涨跌幅'], turnover_rate=r['换手率'],
            change_3d=r['change_3d'], change_5d=r['change_5d'],
            change_10d=r['change_10d'], consecutive_up_days=r['consecutive_up_days'],
            vol_rank=r['vol_rank']
        ) for r in records}
    return rs



def get_all_data_feather(start_date: str = None, end_date: str = None) -> Dict[str, Dict[str, StockDayData]]:

    stock_list_df=get_with_cache_feather(f"get_stock_list_{day}.pkl", get_stock_list)
    # stock_his_df=get_with_cache_feather(f"get_stock_his_{start_date}_{end_date}.pkl", lambda: get_stock_his2(stock_list_df, start=start_date, end=end_date))
    # stock_his_df=get_stock_his2(stock_list_df, start=start_date, end=end_date)
    # full_df=get_with_cache_feather(f"get_stock_rich_{start_date}_{end_date}.pkl", lambda: calc_tech(stock_his_df))
    full_df=pd.DataFrame()
    for year in range(2015, 2026):
        stock_his_df=get_cache_feather(f"stock_his_{year}.feather")
        full_df=pd.concat([full_df, stock_his_df], ignore_index=True)    

    full_df=calc_tech(full_df)
    return build_index(full_df)

if __name__ == "__main__":
    # strt_time=datetime.now()
    # day=datetime.now().strftime("%Y%m%d")
    # dict=get_all_data_feather(start_date="20150101", end_date="20251231")                                                                                                                                                                                                                                                                                                                                                                              
    # print(dict["600082"]["20151231"])
    # print(f"总耗时: {datetime.now()-strt_time}")
    

    # df=get_cache_feather(f"get_stock_his_20150101_20251231.pkl")

    for year, group in df.groupby(df["日期"].str[:4]):
        set_cache_feather(group,f"stock_his_{year}.feather")
        # group.to_feather(f"get_stock_his_{year}.feather")

    print(df.head())
    
