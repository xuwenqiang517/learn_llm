import array
from datetime import date, datetime
from typing import OrderedDict, Dict, Any
import akshare as ak
import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass, fields
from tqdm import tqdm
import msgpack

@dataclass
class StockDayData:
    date: str
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
    def toString(self):
        # print(f"日期{self.date} 股票{self.code} {self.name} 开盘{self.open} 收盘{self.close} 最高{self.high} 最低{self.low} 成交量{self.volume} 成交额{self.amount} 振幅{self.amplitude} 涨跌幅{self.change_pct} 涨跌{self.change} 换手率{self.turnover_rate} 3日涨跌{self.change_3d} 5日涨跌{self.change_5d} 10日涨跌{self.change_10d} 量比{self.volume_ratio} 量价多头{self.volume_trend} PE{self.pe} PB{self.pb} 总市值{self.market_cap} 流通市值{self.circulating_market_cap}")
        return f"日期:{self.date} {self.code} {self.name} 开盘{self.open} 收盘{self.close} 成交量{self.volume} 涨跌幅{self.change_pct} 涨跌{self.change} 换手率{self.turnover_rate} 量比{self.volume_ratio} 量价多头{self.volume_trend} PE{self.pe} PB{self.pb} 总市值"

    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, StockDayData):
            return False
        return self.code == other.code

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

def _dict_to_dataclass(obj):
    if isinstance(obj, dict):
        if "date" in obj and "code" in obj and "name" in obj:
            return StockDayData(**obj)
        return {k: _dict_to_dataclass(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dict_to_dataclass(item) for item in obj]
    return obj

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

cache_url= Path(__file__).resolve().parent.parent / ".temp/data/cache" 

def get_with_cache(file_name, func):
    try:
        with open(cache_url/file_name, "rb") as f:
            data = f.read()
            data = msgpack.unpackb(data, raw=False)
            def restore_dataframes(obj):
                if isinstance(obj, dict):
                    return {k: restore_dataframes(v) for k, v in obj.items()}
                elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
                    cols = list(obj[0].keys()) if obj else []
                    if cols:
                        return pd.DataFrame(obj, columns=cols)
                return obj
            
            print(f"从缓存中读取 {file_name}")
            return _dict_to_dataclass(restore_dataframes(data))
    except Exception as e:
        print(f"读缓存{file_name}失败 原因：{e}")
    
    rs = func()
    packed = msgpack.packb(rs, default=_dataclass_to_dict, use_bin_type=True)
    with open(cache_url/file_name, "wb") as f:
        f.write(packed)
    print(f"写入缓存 {file_name}")
    return rs

class TradingCalendar:
    day_dict = None
    today = None
    def __init__(self) -> None:
        self.today=date.today().strftime("%Y%m%d")
        self.day_dict = get_with_cache("trading_days.pkl",self.load)
        

    def load(self):
        df = ak.tool_trade_date_hist_sina()
        rs = {}
        dates = []
        include_today = datetime.now().hour>15
        
        for d in df["trade_date"]:
            dt = d.strftime("%Y%m%d")
            if dt <= self.today:
                if not include_today and dt==self.today:
                    continue
                dates.append(dt)
        
        for i, date in enumerate(dates):
            prev_day = dates[i-1] if i > 0 else "000000"
            next_day = dates[i+1] if i < len(dates)-1 else "000000"
            rs[date] = [prev_day, next_day]
        
        return rs


    def last(self):
        return list(self.day_dict.keys())[-1]

    # 返回当前日期的前一个交易日
    def pre(self, cur_day: str = None):
        if cur_day is None:
            cur_day = self.today
        arr = self.day_dict.get(cur_day)
        if arr:
            return arr[0]
        max_day = "000000"
        for day in self.day_dict.keys():
            if day < cur_day and day > max_day:
                max_day = day
        return max_day
        

    #返回当前日期的后一个交易日
    def next(self, cur_day: str = None):
        if cur_day is None:
            cur_day = self.today
        arr = self.day_dict.get(cur_day)
        if arr:
            return arr[1]
        min_day = "999999"
        for day in self.day_dict.keys():
            if day > cur_day and day < min_day:
                min_day = day
        return min_day


calendar=TradingCalendar()


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

    # return all.head(20)
    return all


# 获取股票的所有信息,返回 代码，名称，叠加上获取到的历史数据
def get_stock_his(df: pd.DataFrame):
    rs={}
    stock_list = df[["代码", "名称"]].values
    last_day=calendar.last()
    for code, name in tqdm(stock_list, desc="获取股票数据"):
        stock_df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20250101", end_date=last_day)
        stock_df.insert(2, "名称", name)
        day_int=pd.to_datetime(stock_df["日期"]).dt.strftime("%Y%m%d")
        stock_df["日期"] = day_int
        rs[code] = stock_df
    return rs

def get_stock_basic_info(dict: dict):
    df = ak.stock_zh_a_spot_em()
    df = df[["代码", "市盈率-动态", "市净率", "总市值", "流通市值", "涨跌幅", "换手率"]]
    df.columns = ["代码", "PE", "PB", "总市值", "流通市值", "涨跌幅", "换手率"]
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    print(df)
    

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

def get_all_data() -> Dict[str, Dict[str, StockDayData]]:
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


def pick_stock(dict: Dict[str, Dict[str, StockDayData]], day: str, strategy: str) -> list:
    rs_list=[]
    # print(f"dict数量:{len(dict.keys())}")
    # print(f"过滤策略:{strategy}")
    for code, data in dict.items():
        info=data.get(day)
        if info is None:
            # print(f"股票{code} 日期{day},没有信息，跳过")
            continue
        # 过滤逻辑
        flag=True
        if strategy=="量价多头":
            flag = flag and info.pe>0 and info.market_cap>100 and info.change_pct<5 and info.change_pct>2 and info.turnover_rate>5 
            flag = flag and info.change_3d>3 and info.change_5d>10 and info.pe>5 and info.pe<80 and info.pb<10
        
        if flag:
            rs_list.append(info)
        if len(rs_list)>=5:
            break
    # for f in rs_list:
    #     print(f)
    # print(f"日期{day} 选股数量{len(rs_list)}")
    return rs_list



@dataclass
class BuySellInfo:
    code:str
    name:str
    # 日期
    buy_day:str
    # 买入价格
    buy_price:float
    # 卖出日期
    sell_day:str = ""
    # 卖出价格
    sell_price:float = 0.0
    # 买入数量
    count: int = 0
    #每日收盘
    close:float = 0.0
    

    # 买入金额
    def buy_value(self):
        return round(self.count*self.buy_price, 2)
    # 卖出金额
    def sell_value(self):
        return round(self.count*self.sell_price, 2)
    # 盈亏率
    def profit_rate(self):
        return round(self.sell_value()/self.buy_value()-1, 2)
    # 盈亏金额
    def profit_amount(self):
        return round(self.sell_value()-self.buy_value(), 2)
    # 持仓金额
    def hold_value(self):
        return round(self.close*self.count, 2)      

    
    def buy_string(self):
        print(f"股票{self.code} {self.name} 买入日期{self.buy_day} 买入价格{self.buy_price}  股数:{self.count}  买入金额{self.buy_value()}")

    def sell_string(self):
        print(f"股票{self.code} {self.name} 买入日期{self.buy_day} 买入价格{self.buy_price} 卖出日期{self.sell_day} 卖出价格{self.sell_price} 股数:{self.count} 累计涨跌幅:{self.profit_rate()}  盈亏:{self.profit_amount()}")



def get_data(data_dict:Dict[str, Dict[str, StockDayData]]=None,code:str=None,day:str=None)->StockDayData:
    return data_dict.get(code).get(day) if data_dict.get(code) is not None else None

def back_test(start:str="20260101",end:str="20260115",data_dict:Dict[str, Dict[str, StockDayData]]=None,base_total_amount:float=100000.00,buy_count:int=10000):
    # 昨天的信息
    calendar
    pre_day=calendar.pre(start)
    cur_day=calendar.next(pre_day)
    stock_list=[]
    hold_set=set()
    sell_list=[]
    buy_list=[]
    dict_info={}
    #空闲金额
    total_amount=base_total_amount
    
    while cur_day<end:
        print("="*60+f"今天:{cur_day}"+ "="*60)
        # 根据上一个交易日选出来的票
        #选票
        print(f"{pre_day} 选票:=========")
        stock_list=pick_stock(data_dict,pre_day,"量价多头")
        for stock in stock_list:
            print(stock.toString())

        # 判断卖出
        print(f"{cur_day} 卖出:=========")
        sell_list=[]
        for stock in hold_set:
            stock_info_pre=get_data(data_dict,stock.code,pre_day)
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            if stock_info_pre is None or stock_info_cur is None:
                print(f"股票{stock.code} 日期{cur_day},没有信息，跳过")
                continue
            if(stock_info_pre.change_pct<-5):
                sell_list.append(stock)
                trade_info=dict_info[stock.code]
                trade_info.sell_price=stock_info_cur.open
                trade_info.sell_day=cur_day
                # 金额处理
                total_amount+=trade_info.sell_value()


        for stock in sell_list:
            hold_set.remove(stock)
            trade=dict_info[stock.code]
            trade.sell_string()
            dict_info.pop(stock.code)

        # 判断买入
        print(f"{cur_day} 买入:=========")
        buy_list=[]
        for stock in stock_list:
            if(len(hold_set)<5) and stock not in hold_set:
                stock_info_cur=get_data(data_dict,stock.code,cur_day)
                if stock_info_cur is None:
                    print(f"股票{stock.code} 日期{cur_day},没有信息，跳过")
                    continue
                hold_set.add(stock)
                buy_list.append(stock)
                trade=BuySellInfo(code=stock.code,name=stock.name,buy_day=cur_day,buy_price=stock_info_cur.open,sell_price=0.0,count=(buy_count/stock_info_cur.open)//100*100)
                dict_info[stock.code]=trade
                trade.buy_string()
                # 金额处理
                total_amount-=trade.buy_value()


        print(f"{cur_day} 持仓:=========")
        hold_value=0
        for stock in hold_set:
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            print(stock_info_cur.toString())
            code=stock.code
            trade_info=dict_info[code]
            
            # 更新今天的收盘价
            trade_info.close=stock_info_cur.close
            # 更新持仓金额
            hold_value+=trade_info.hold_value()
        
        print(f"{cur_day} 空闲金额:{total_amount}")
        print(f"{cur_day} 持仓金额:{hold_value}")
        print(f"{cur_day} 总金额:{total_amount+hold_value}")

        pre_day=cur_day
        cur_day=calendar.next(cur_day)

    print( f"总收益率:{round((total_amount+hold_value-base_total_amount)/base_total_amount*100,2)}%" )


if __name__ == "__main__":
    data_dict=get_all_data()
    # stock_list=pick_stock(data_dict, "20260116","量价多头")
    back_test(data_dict=data_dict,start="20250101",end="20260115")

