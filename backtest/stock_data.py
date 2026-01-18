from datetime import date
import imp
from typing import OrderedDict, Dict, Any
import akshare as ak
import pandas as pd
import pickle
import sys
from pathlib import Path
from dataclasses import dataclass


@dataclass
class StockDayData:
    date: int
    code: str
    name: str
    open: float
    close: float
    high: float
    low: float
    volume: int
    amount: float
    amplitude: float
    change_pct: float
    change: float
    turnover_rate: float
    change_3d: float
    change_5d: float
    change_10d: float
    volume_ratio: float
    volume_trend: int
    pe: float
    pb: float
    market_cap: float
    circulating_market_cap: float
    def print(self):
        print(f"日期{self.date} 股票{self.code} {self.name} 开盘{self.open} 收盘{self.close} 最高{self.high} 最低{self.low} 成交量{self.volume} 成交额{self.amount} 振幅{self.amplitude} 涨跌幅{self.change_pct} 涨跌{self.change} 换手率{self.turnover_rate} 3日涨跌{self.change_3d} 5日涨跌{self.change_5d} 10日涨跌{self.change_10d} 量比{self.volume_ratio} 量价多头{self.volume_trend} PE{self.pe} PB{self.pb} 总市值{self.market_cap} 流通市值{self.circulating_market_cap}")


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cache_url= Path(__file__).resolve().parent.parent / ".temp/data/cache" 
# 获取沪深股票列表
def get_stock_list() -> pd.DataFrame:
    print("获取股票数据")
    stock_sh = ak.stock_info_sh_name_code(symbol="主板A股")
    stock_sh=stock_sh[["证券代码", "证券简称"]]
    stock_sh.columns = ["代码", "名称"]

    print("上海主板A股数量：", len(stock_sh))
    stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
    stock_sz=stock_sz[["A股代码", "A股简称"]]
    stock_sz.columns = ["代码", "名称"]
    print("深圳A股列表数量：", len(stock_sz))
    all = stock_sh._append(stock_sz, ignore_index=True)
    print("合并后总股票数量：", len(all))

    # 过滤掉科创/创业股票
    all = all[~all["代码"].str.startswith(("688", "300", "301"))]
    print("过滤科创/创业股票后数量：", len(all))

    # 过滤掉ST股票
    all = all[~all["名称"].str.contains("ST")]
    print("过滤ST股票后数量：", len(all))

    return all


# 获取股票的所有信息,返回 代码，名称，叠加上获取到的历史数据
def get_stock_his(df: pd.DataFrame):
    from tqdm import tqdm
    rs={}
    stock_list = df[["代码", "名称"]].values
    for code, name in tqdm(stock_list, desc="获取股票数据"):
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20250101", end_date=date.today().strftime("%Y%m%d"))
        stock_df.insert(2, "名称", name)
        day_int=pd.to_datetime(stock_df["日期"]).dt.strftime("%Y%m%d").astype(int)
        stock_df["日期"] = day_int
        rs[code] = stock_df
    return rs

def get_stock_basic_info(dict: dict):
    df = ak.stock_zh_a_spot_em()
    df = df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值", "涨跌幅", "换手率"]]
    df.columns = ["代码", "PE", "PB", "总市值", "流通市值", "涨跌幅", "换手率"]
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    print(df)
        

import msgpack
from dataclasses import asdict, fields


def _dataclass_to_dict(obj):
    if isinstance(obj, StockDayData):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    return obj


def read_cache(file_name):
    try:
        with open(cache_url/file_name, "rb") as f:
            data = f.read()
            try:
                data = msgpack.unpackb(data, raw=False)
            except:
                data = pickle.loads(data)
            
            def restore_dataframes(obj):
                if isinstance(obj, dict):
                    return {k: restore_dataframes(v) for k, v in obj.items()}
                elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                    cols = list(obj[0].keys()) if obj else []
                    if cols:
                        return pd.DataFrame(obj, columns=cols)
                return obj
            
            return restore_dataframes(data)
    except:
        return None


def write_cache(file_name, data):
    packed = msgpack.packb(data, default=_dataclass_to_dict, use_bin_type=True)
    with open(cache_url/file_name, "wb") as f:
        f.write(packed)

def get_with_cache(file_name, func):
    rs = read_cache(file_name)
    if rs is None:
        rs = func()
        write_cache(file_name, rs)
        print(f"写入缓存 {file_name}")
    else:
        print(f"从缓存中读取 {file_name}")
    return rs


#计算技术指标
def calc_tech_indicators(dict: dict):
    for code, df in dict.items():
        ma5 = df["收盘"].rolling(window=5).mean().round(2)
        ma10 = df["收盘"].rolling(window=10).mean().round(2)
        ma20 = df["收盘"].rolling(window=20).mean().round(2)
        # 计算成交量指标
        vol_ma5 = df["成交量"].rolling(window=5).mean().round(2)
        vol_ma10 = df["成交量"].rolling(window=10).mean().round(2)
        vol_ma20 = df["成交量"].rolling(window=20).mean().round(2)
        ma5 = df["收盘"].rolling(window=5).mean().round(2)
        ma10 = df["收盘"].rolling(window=10).mean().round(2)
        ma20 = df["收盘"].rolling(window=20).mean().round(2)
        # 计算成交量指标
        vol_ma5 = df["成交量"].rolling(window=5).mean().round(2)
        vol_ma10 = df["成交量"].rolling(window=10).mean().round(2)
        vol_ma20 = df["成交量"].rolling(window=20).mean().round(2)
        df["3日涨幅"] = df["涨跌幅"].rolling(window=3).sum().round(2)
        df["5日涨幅"] = df["涨跌幅"].rolling(window=5).sum().round(2)
        df["10日涨幅"] = df["涨跌幅"].rolling(window=10).sum().round(2)
        df["量比"] = df["成交量"] / vol_ma20
        df["量比"] = df["量比"].round(2)
        # 是否量价多头 用ma5和ma10比较和ma20比较 如果ma5和ma10都大于ma20 则认为是量价多头
        # 是否量价多头 用vol_ma5和vol_ma10比较和vol_ma20比较 如果vol_ma5和vol_ma10都大于vol_ma20 则认为是量价多头
        df["量价多头"] = ((ma5 > ma20) & (ma10 > ma20)&(vol_ma5 > vol_ma20)&(vol_ma10 > vol_ma20))
    return dict
def calc_basic_indicators(dict: dict):
    stock_zh_a_spot_em_df=get_with_cache("stock_zh_a_spot_em.pkl", ak.stock_zh_a_spot_em)
    # 遍历stock_zh_a_spot_em_df，根据代码，把PE,PB,总市值,流通市值,涨跌幅,换手率 合并到dict中
    for row in stock_zh_a_spot_em_df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值"]].values:
        code=row[0]
        if code not in dict:
            continue
        df=dict[code]
        df["pe"]=row[1]
        df["pb"]=row[2]
        # 单位亿
        df["market_cap"]=int(row[3]/100000000)
        df["circulating_market_cap"]=int(row[4]/100000000)
    return dict

def cslc_index(dict: dict):
    # 转成二级dict 第一层key用code，第二层key用日期，转成StockDayData对象
    dict2: Dict[str, Dict[int, StockDayData]] = {}
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
                change_3d=row[13],
                change_5d=row[14],
                change_10d=row[15],
                volume_ratio=row[16],
                volume_trend=row[17],
                pe=row[18],
                pb=row[19],
                market_cap=row[20],
                circulating_market_cap=row[21]
            )
    return dict2

def get_all_data() -> Dict[str, Dict[int, StockDayData]]:
    # 取股票列表
    df=get_with_cache("get_stock_list.pkl", get_stock_list)
    # 取股票历史数据
    dict=get_with_cache("get_stock_his.pkl", lambda: get_stock_his(df))
    # 计算技术指标
    dict=get_with_cache("calc_tech_indicators.pkl", lambda: calc_tech_indicators(dict))
    # 基本面指标
    dict=get_with_cache("calc_basic_indicators.pkl", lambda: calc_basic_indicators(dict))
    # 转换成二级索引
    dict=get_with_cache("cslc_index.pkl", lambda: cslc_index(dict))

    print(f"股票数量: {len(dict)}")
    return dict


def pick_stock(dict: Dict[str, Dict[int, StockDayData]], day: int, filters: list) -> list:
    rs_list=[]
    for code, data in dict.items():
        info=data.get(day)
        if info is None:
            continue
        # 过滤逻辑
        flag=True
        for f in filters:
            if f=="量价多头":
                if info.volume_trend==False:
                    flag=False
                    break
                if info.pe<=0:
                    flag=False
                    break
                if info.market_cap<=100:
                    flag=False
                    break
                if info.change_pct >= 5 or info.change_pct <= -3 or info.turnover_rate<=5:
                    flag=False
                    break
                if info.change_3d<=1:
                    flag=False
                    break
                if info.change_5d<=3:
                    flag=False
                    break
            # PE合理范围 (5-80)
                if info.pe < 5 or info.pe > 80:
                    flag=False
                    break
            # PB不过高 (< 10)
                if info.pb >= 10:
                    flag=False
                    break
            # 量能充沛 (量比 > 1.5)
                if info.volume_ratio <= 1.5:
                    flag=False
                    break
            # 近期涨幅适中 (10日涨幅 < 30%)
                if info.change_10d >= 30:
                    flag=False
                    break
            # 波动适中 (振幅 < 10%)
                if info.amplitude >= 10:
                    flag=False
                    break
            # 收盘价 > 10元 (避免仙股)
                if info.close <= 10:
                    flag=False
                    break
            # 换手率适中 (3% < 换手率 < 30%)
                if info.turnover_rate <= 3 or info.turnover_rate >= 30:
                    flag=False
                    break
            # 成长股筛选 (PE 20-60, 量比 > 1.5)
                if info.pe < 20 or info.pe > 60:
                    flag=False
                    break
                if info.volume_ratio <= 1.5:
                    flag=False
                    break
                if info.change_10d >= 20:
                    flag=False
                    break
            # 价值股筛选 (PE 5-25, PB < 3, 中大市值)
                if info.pe < 5 or info.pe > 25:
                    flag=False
                    break
                if info.pb >= 3:
                    flag=False
                    break
                if info.market_cap <= 200:
                    flag=False
                    break
        
        if flag:
            rs_list.append(info)
    for f in rs_list:
        f.print()
    print(f"日期{day} 选股数量{len(rs_list)}")
    return rs_list

if __name__ == "__main__":
    dict=get_all_data()
    stock_list=pick_stock(dict, 20260116,["量价多头"])
    # print(stock_list)
