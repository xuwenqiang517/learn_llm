from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
import akshare as ak
import pandas as pd
from tqdm import tqdm
from StockDayData import StockDayData
from cache_utils import get_with_cache

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

def get_stock_his(df: pd.DataFrame, start_date: str = "20100101", end_date: str = None):
    rs = {}
    stock_list = df[["代码", "名称"]].values
    print(f"获取股票历史数据: {start_date} - {end_date}")
    
    def fetch_stock(code_name):
        code, name = code_name
        try:
            stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date)
            stock_df.insert(2, "名称", name)
            day_int = pd.to_datetime(stock_df["日期"]).dt.strftime("%Y%m%d")
            stock_df["日期"] = day_int
            return code, stock_df
        except Exception as e:
            return code, None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_stock, code_name): code_name for code_name in stock_list}
        
        for future in tqdm(as_completed(futures), total=len(stock_list), desc="下载进度"):
            code, stock_df = future.result()
            if stock_df is not None:
                rs[code] = stock_df
    
    return rs

def calc_tech_indicators(dict: dict):
    for code, df in dict.items():
        ma5 = df["收盘"].rolling(window=5).mean().round(2)
        ma10 = df["收盘"].rolling(window=10).mean().round(2)
        ma20 = df["收盘"].rolling(window=20).mean().round(2)
        vol_ma5 = df["成交量"].rolling(window=5).mean().round(2)
        vol_ma10 = df["成交量"].rolling(window=10).mean().round(2)
        vol_ma20 = df["成交量"].rolling(window=20).mean().round(2)
        df["change_3d"] = df["涨跌幅"].rolling(window=3).sum().round(2)
        df["change_5d"] = df["涨跌幅"].rolling(window=5).sum().round(2)
        df["change_10d"] = df["涨跌幅"].rolling(window=10).sum().round(2)
        df["volume_ratio"] = (df["成交量"] / vol_ma20.replace(0, 1)).round(2)
        df["volume_trend"] = (ma5 > ma10) & (ma10 > ma20) & (vol_ma5 > vol_ma10) & (vol_ma10 > vol_ma20)
        
        close_prices = df["收盘"].values
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
        df["consecutive_up_days"] = consecutive_up
    
    return dict

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

def cslc_index(dict: dict):
    dict2: Dict[str, Dict[str, StockDayData]] = {}
    for code, df in dict.items():
        dict2[code] = {}
        for row in df.itertuples(index=False):
            dict2[code][row[0]] = StockDayData(
                date=row[0],
                code=row[1],
                name=row[2],
                open=row[3],
                close=row[4],
                high=row[5],
                low=row[6],
                volume=row[7],
                amount=row[8],
                amplitude=row[9],
                change_pct=row[10],
                change=row[11],
                turnover_rate=row[12],
                pe=row[13],
                pb=row[14],
                market_cap=row[15],
                circulating_market_cap=row[16],
                change_3d=row[17],
                change_5d=row[18],
                change_10d=row[19],
                volume_ratio=row[20],
                volume_trend=row[21],
                consecutive_up_days=row[22]
            )
    return dict2

def get_all_data(start_date: str = None, end_date: str = None) -> Dict[str, Dict[str, StockDayData]]:
    df = get_with_cache("get_stock_list.pkl", get_stock_list)
    dict = get_with_cache(f"get_stock_his_{start_date}_{end_date}.pkl", lambda: get_stock_his(df, start_date=start_date, end_date=end_date))
    
    for code, df in dict.items():
        if df is None:
            continue
        df = df[(df["日期"] >= start_date) & (df["日期"] <= end_date)]
        dict[code] = df
    
    calc_basic_indicators(dict)
    dict = calc_tech_indicators(dict)
    dict = cslc_index(dict)

    return dict
