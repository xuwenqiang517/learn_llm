import datetime
import akshare as ak
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from cache_utils import *
from datetime import datetime, date

today = date.today().strftime("%Y%m%d")

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

def get_stock_data_tx(code: str) -> pd.DataFrame:
    print(f"获取股票数据 {code}")
    market = "sh" if code.startswith("6") else "sz"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,1300,qfq,1"
    headers = {"Referer": "http://finance.qq.com/mp/"}
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    if data.get("code") == 0 and data.get("data") and data["data"].get(f"{market}{code}"):
        items = data["data"][f"{market}{code}"].get("qfqday", [])
        print(f"获取股票数据 {code} 成功，共 {len(items)} 条数据")
        items = [row[:6] for row in items]
        df = pd.DataFrame(items, columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])
        df["股票代码"] = code
        df["日期"] = df["日期"].str.replace("-", "")
        df["开盘"] = df["开盘"].astype(float)
        df["收盘"] = df["收盘"].astype(float)
        df["最高"] = df["最高"].astype(float)
        df["最低"] = df["最低"].astype(float)
        df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce").fillna(0).astype(int)
        df["涨跌幅"] = (df["收盘"] - df["收盘"].shift(1)) / df["收盘"].shift(1) * 100
        df["涨跌幅"] = df["涨跌幅"].round(2).fillna(0)
        df = df[["股票代码", "日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]
        df = calc_tech(df)
        return df
    else:
        print(f"获取股票数据 {code} 失败")
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

def get_data()->pd.DataFrame:
    
    stock_list=get_with_cache_feather(f"stock_list_{today}.feather", lambda: get_stock_list())
    print(f"获取股票数量：{len(stock_list)}")

    #遍历stock_list 获取股票数据，优先从缓存取，并判断缓存结果中 最大的交易日 和当前最新交易日，如果最大交易日 小于 当前最新交易日，则从tx获取数据
    stock_data = pd.DataFrame()
    for code in tqdm(stock_list["代码"]):
        # df = get_with_cache_feather(f"stock_data_{code}_{today}.feather", lambda: get_stock_data_tx(code))
        df = get_stock_data_tx(code)
        if not df.empty:
            stock_data = pd.concat([stock_data, df], ignore_index=True)
        else:
            print(f"获取股票数据 {code} 失败")

    #计算技术指标
    stock_data = calc_tech(stock_data)
    return stock_data

def get_full_data()->pd.DataFrame:
    df=get_with_cache_feather(f"stock_full_{today}.feather", lambda: get_data())
    df.set_index(["股票代码", "日期"], inplace=True)
    return df


if __name__ == '__main__':
    df=get_full_data()
    #df 建索引 股票代码+日期
    
    
    # print(df.loc[("302132", "20260130")])
    print(df.loc[("302132")])

