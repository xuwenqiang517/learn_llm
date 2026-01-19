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


def print_green(msg,enable=True):
    if enable:
        print(f"\033[92m{msg}\033[0m")

def print_red(msg,enable=True):
    if enable:
        print(f"\033[91m{msg}\033[0m")

def print_yellow(msg,enable=True):
    if enable:
        print(f"\033[93m{msg}\033[0m")

def print_blue(msg,enable=True):
    if enable:
        print(f"\033[94m{msg}\033[0m")

def print_white(msg,enable=True):
    if enable:
        print(f"\033[97m{msg}\033[0m")

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
    pe: float
    pb: float
    market_cap: float
    circulating_market_cap: float
    change_3d: float
    change_5d: float
    change_10d: float
    volume_ratio: float
    volume_trend: bool
    consecutive_up_days: int

    def toString(self):
        return f"开盘{self.open} 收盘{self.close} 成交量{self.volume} 涨跌幅{self.change_pct} 涨跌{self.change} 换手率{self.turnover_rate} 量比{self.volume_ratio} PE{self.pe} PB{self.pb} 量价多头{self.volume_trend} 连涨天数{self.consecutive_up_days} 3日涨跌{self.change_3d} 5日涨跌{self.change_5d}"
    def full_info(self):
        return f"日期:{self.date} {self.code} {self.name} 开盘{self.open} 收盘{self.close} 成交量{self.volume} 涨跌幅{self.change_pct} 涨跌{self.change} 换手率{self.turnover_rate} PE{self.pe} PB{self.pb} 量比{self.volume_ratio} 量价多头{self.volume_trend} 连涨天数{self.consecutive_up_days} 3日涨跌{self.change_3d} 5日涨跌{self.change_5d}"

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
    start_time=datetime.now()
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
            
            # print(f"从缓存中读取 {file_name} 耗时：{datetime.now()-start_time}ms")
            return _dict_to_dataclass(restore_dataframes(data))
    except Exception as e:
        print(f"读缓存{file_name}失败 原因：{e}")
    
    rs = func()
    packed = msgpack.packb(rs, default=_dataclass_to_dict, use_bin_type=True)
    with open(cache_url/file_name, "wb") as f:
        f.write(packed)
    print(f"写入缓存 {file_name} 耗时：{datetime.now()-start_time}ms")
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

    # print("上海主板A股数量：", len(stock_sh))
    stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
    stock_sz=stock_sz[["A股代码", "A股简称"]]
    stock_sz.columns = ["代码", "名称"]
    # print("深圳A股列表数量：", len(stock_sz))
    all = stock_sh._append(stock_sz, ignore_index=True)
    # print("合并后总股票数量：", len(all))

    # 过滤掉科创/创业股票
    all = all[~all["代码"].str.startswith(("688", "300", "301"))]
    # print("过滤科创/创业股票后数量：", len(all))

    # 过滤掉ST股票
    all = all[~all["名称"].str.contains("ST")]
    # print("过滤ST股票后数量：", len(all))
    print(f"获取股票数量：{len(all)}")
    # return all.head(20)
    return all


# 获取股票的所有信息,返回 代码，名称，叠加上获取到的历史数据
def get_stock_his(df: pd.DataFrame, start_date: str = "20100101", end_date: str = None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
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
            # print(f"获取 {code} 失败: {e}")
            return code, None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch_stock, code_name): code_name for code_name in stock_list}
        
        for future in tqdm(as_completed(futures), total=len(stock_list), desc="下载进度"):
            code, stock_df = future.result()
            if stock_df is not None:
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
        vol_ma5 = df["成交量"].rolling(window=5).mean().round(2)
        vol_ma10 = df["成交量"].rolling(window=10).mean().round(2)
        vol_ma20 = df["成交量"].rolling(window=20).mean().round(2)
        # df["ma5"] = ma5
        # df["ma10"] = ma10
        # df["ma20"] = ma20
        # df["vol_ma5"] = vol_ma5
        # df["vol_ma10"] = vol_ma10
        # df["vol_ma20"] = vol_ma20
        df["change_3d"] = df["涨跌幅"].rolling(window=3).sum().round(2)
        df["change_5d"] = df["涨跌幅"].rolling(window=5).sum().round(2)
        df["change_10d"] = df["涨跌幅"].rolling(window=10).sum().round(2)
        df["volume_ratio"] = (df["成交量"] / vol_ma20).round(2)
        df["volume_trend"] = (ma5 > ma10) & (ma10 > ma20) & (vol_ma5 > vol_ma10) & (vol_ma10 > vol_ma20)
        
        # 计算连涨天数（收盘价连续上涨的天数）
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

                # 基本面指标
                pe=row[13],
                pb=row[14],
                market_cap=row[15],
                circulating_market_cap=row[16],
                # 技术指标
                change_3d=row[17],
                change_5d=row[18],
                change_10d=row[19],
                volume_ratio=row[20],
                volume_trend=row[21],
                consecutive_up_days=row[22]
                
            )
    return dict2

def get_all_data(start_date:str=None, end_date:str=None) -> Dict[str, Dict[str, StockDayData]]:
    # 取股票列表
    df=get_with_cache("get_stock_list.pkl", get_stock_list)
    # 取股票历史数据
    # his_start="20100101"
    # his_end=calendar.last()
    dict=get_with_cache(f"get_stock_his_{start_date}_{end_date}.pkl", lambda: get_stock_his(df, start_date=start_date, end_date=end_date))
    # dict=get_with_cache(f"get_stock_his_{his_start}_{his_end}.pkl", lambda: get_stock_his(df, start_date=his_start, end_date=his_end))
    
    for code, df in dict.items():
        if df is None:
            continue
        # 只保留日期>=start_date and 日期<=end_date
        df=df[(df["日期"]>=start_date) & (df["日期"]<=end_date)]
        dict[code]=df
    

    # 基本面指标
    # dict=get_with_cache(f"calc_basic_indicators.pkl", lambda: calc_basic_indicators(dict))
    calc_basic_indicators(dict)
    # 计算技术指标
    # dict=get_with_cache(f"calc_tech_indicators_{start_date}_{end_date}.pkl", lambda: calc_tech_indicators(dict))
    dict=calc_tech_indicators(dict)
    # 转换成二级索引
    # dict=get_with_cache(f"cslc_index_{start_date}_{end_date}.pkl", lambda: cslc_index(dict))
    dict=cslc_index(dict)

    # print(f"股票数量: {len(dict)}")
    return dict


def pick_stock(dict: Dict[str, Dict[str, StockDayData]], day: str, strategy: str) -> list:
    rs_list=[]
    for code, data in dict.items():
        info=data.get(day)
        if info is None:
            continue
        # 过滤逻辑
        # print(info.full_info())
        flag=True
        #pickrule
        # flag = flag and info.volume_trend==True
        flag = flag and info.consecutive_up_days>=3
        # flag = flag and info.pe>0
        flag = flag and info.market_cap>100
        # flag = flag and info.change < 5 and info.change > 1
        flag = flag and info.change_pct < 0.98
        flag = flag and info.volume_ratio>1.5
        flag = flag and info.change_3d>3 and info.change_5d>10
        if flag:
            rs_list.append(info)
    
    # 按涨幅最小排序
    rs_list.sort(key=lambda x: x.change, reverse=True)
    rs=rs_list[:5]
    # print(f"选中{rs}")
    return rs



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
    # 持仓天数(包括买入天)
    hold_day:int = 0
    # 卖出原因
    sell_reason:str = ""

    # 买入金额
    def buy_value(self):
        return round(self.count*self.buy_price, 2)
    # 卖出金额
    def sell_value(self):
        return round(self.count*self.sell_price, 2)
    # 盈亏率
    def profit_rate(self):
        return round((self.sell_value()/self.buy_value()-1)*100, 2)
    # 盈亏金额 #持仓时用 hold_value()，卖出时用 sell_value()
    def profit_amount(self):
        if self.sell_price > 0:
            return round(self.sell_value() - self.buy_value(), 2)
        else:
            return round(self.hold_value() - self.buy_value(), 2)
    # 持仓金额
    def hold_value(self):
        return round(self.close*self.count, 2)      
    # 持有收益率
    def hold_profit_rate(self):
        return round((self.hold_value()/self.buy_value()-1)*100, 2)

        
    def buy_string(self):
        return f"股票{self.code} {self.name} 买入:{self.buy_day} 买入价：{self.buy_price} 股数:{self.count} 金额{self.buy_value()}"

    def sell_string(self):
        return f"股票{self.code} {self.name} 时间:{self.buy_day}-{self.sell_day}（{self.hold_day}天） 价格:{self.buy_price} -> {self.sell_price} 涨跌幅:{self.profit_rate()}%  盈亏:{self.profit_amount()}  卖出原因:{self.sell_reason}"
    
    def hold_string(self):
        return f"股票{self.code} {self.name} 买入:{self.buy_day} 买入价:{self.buy_price} 收盘价:{self.close} 持仓:{self.hold_day}天  持仓收益率{self.hold_profit_rate()}% 盈亏:{self.profit_amount()}"



def get_data(data_dict:Dict[str, Dict[str, StockDayData]]=None,code:str=None,day:str=None)->StockDayData:
    return data_dict.get(code).get(day) if data_dict.get(code) is not None else None

def back_test(start:str="20260101",end:str="20260119",data_dict:Dict[str, Dict[str, StockDayData]]=None,base_total_amount:float=100000.00,buy_count:int=10000,max_hold_count:int=10,print_detail:bool=False,print_fail_reason:bool=False)->tuple[float,float]:
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
    #最高收益率
    # 最低收益率
    max_profit_rate=float("-inf")
    min_profit_rate=float("inf")
    # 卖出价最高的票信息
    max_sell_price=0
    max_sell_price_stock=None
    # 胜率
    win_count=0
    # 总交易次数
    trade_count=0
    while cur_day<end:
        print_white("="*80+f"今天:{cur_day}"+ "="*80,print_detail)
        #选票
        print_white(f"选票:=========",print_detail)
        stock_list=pick_stock(data_dict,pre_day,"量价多头")
        for stock in stock_list:
            print_blue(stock.full_info(),print_detail)
        # 判断卖出
        print_white(f"{cur_day} 卖出:=========",print_detail)
        sell_list=[]
        for stock in hold_set:
            stock_info_pre=get_data(data_dict,stock.code,pre_day)
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            if stock_info_pre is None or stock_info_cur is None:
                print_white(f"股票{stock.code} 日期{cur_day},没有信息，跳过",print_detail)
                continue
            trade_info=dict_info[stock.code]
            sell_flag=False
            #sellrule
            if(stock_info_pre.change<-5):
                sell_flag=True
                trade_info.sell_reason="[1]涨跌幅<-5%"
            elif trade_info.hold_day>=3 and trade_info.hold_profit_rate()<3:
                sell_flag=True
                trade_info.sell_reason=f"[2]持仓{trade_info.hold_day}天,收益率:"+f"{trade_info.hold_profit_rate()}%,<5%"
            elif trade_info.hold_day>=5 and trade_info.hold_profit_rate()<10:
                sell_flag=True
                trade_info.sell_reason=f"[3]持仓{trade_info.hold_day}天,收益率:"+f"{trade_info.hold_profit_rate()}%,<10%"
            # elif trade_info.hold_day>=10:
            #     sell_flag=True
            #     trade_info.sell_reason=f"[4]持仓{trade_info.hold_day}天,收益率:"+f"{trade_info.hold_profit_rate()}%"
            # elif trade_info.hold_profit_rate()>=40:
            #     sell_flag=True
            #     trade_info.sell_reason=f"[5]持仓{trade_info.hold_day}天,收益率:"+f"{trade_info.hold_profit_rate()}%,>=40%"
            
            # 确认要卖
            if sell_flag:
                sell_list.append(stock)
                trade_info.sell_price=stock_info_cur.open
                trade_info.sell_day=cur_day
                # 金额处理
                total_amount+=trade_info.sell_value()
                if trade_info.profit_amount()>max_sell_price:
                    max_sell_price=trade_info.profit_amount()
                    max_sell_price_stock=trade_info
                # 记录交易次数
                trade_count+=1
                # 记录胜率
                if trade_info.profit_amount()>0:
                    win_count+=1
                else:
                    print_yellow(f"{trade_info.sell_string()} || {stock_info_cur.toString()}",print_fail_reason)


        for stock in sell_list:
            hold_set.remove(stock)
            trade=dict_info[stock.code]
            print_red(trade.sell_string(),print_detail)
            dict_info.pop(stock.code)

        # 判断买入
        print_white(f"{cur_day} 买入:=========",print_detail)
        buy_list=[]
        for stock in stock_list:
            if(len(hold_set)<max_hold_count) and stock not in hold_set:
                stock_info_cur=get_data(data_dict,stock.code,cur_day)
                if stock_info_cur is None:
                    print_white(f"股票{stock.code} 日期{cur_day},没有信息，跳过",print_detail)
                    continue
                count = int(buy_count / stock_info_cur.open) // 100 * 100
                if count == 0:
                    print_white(f"股票{stock.code}{stock.name} 股价{stock_info_cur.open}过高，无法买入，跳过",print_detail)
                    continue
                if total_amount<count*stock_info_cur.open:
                    print_white(f"股票{stock.code}{stock.name} 资金不足，无法买入，跳过",print_detail)
                    continue
                hold_set.add(stock)
                buy_list.append(stock)
                trade=BuySellInfo(code=stock.code,name=stock.name,buy_day=cur_day,buy_price=stock_info_cur.open,sell_price=0.0,count=count)
                dict_info[stock.code]=trade
                print_green(f"{trade.buy_string()} || {stock_info_cur.toString()}",print_detail)
                # 金额处理
                total_amount-=trade.buy_value()


        print_yellow(f"{cur_day} 持仓:=========",print_detail)
        hold_value=0
        for stock in hold_set:
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            if stock_info_cur is None:
                print_white(f"股票{stock.code} 日期{cur_day},没有信息，跳过",print_detail)
                continue
            trade_info=dict_info[stock.code]
            # 更新今天的收盘价
            trade_info.close=stock_info_cur.close
            # 更新持仓金额
            hold_value+=trade_info.hold_value()
            trade_info.hold_day+=1
            print_yellow(f"{trade_info.hold_string()} ||{stock_info_cur.toString()}",print_detail)
            
            
        print_white(f"{cur_day} 空闲金额:{total_amount}",print_detail)
        print_white(f"{cur_day} 持仓金额:{hold_value}",print_detail)
        print_white(f"{cur_day} 总金额:{total_amount+hold_value}",print_detail)
        # 更新最高收益率
        max_profit_rate=max(max_profit_rate, round((total_amount+hold_value)/base_total_amount*100,2))
        # 更新最低收益率
        min_profit_rate=min(min_profit_rate, round((total_amount+hold_value)/base_total_amount*100,2))



        pre_day=cur_day
        cur_day=calendar.next(cur_day)

    total_return_rate=round((total_amount+hold_value-base_total_amount)/base_total_amount*100,2)
    win_rate=round(win_count/trade_count*100,2) if trade_count>0 else 0
    rs_str=f"{start},{end} 资金{base_total_amount}->{total_amount+hold_value} 总收益率:{total_return_rate}% 最高:{max_profit_rate}% 最低:{min_profit_rate}% 胜率:{win_rate}% 总交易次数:{trade_count}"
    print_red( rs_str,True)
    # print_yellow(f" 收益最高的票{max_sell_price_stock.sell_string()}",True)
    return total_return_rate,win_rate

def clearn_cache():
    #清空cache目录:cache_url
    import shutil
    shutil.rmtree(cache_url)
    
    

if __name__ == "__main__":
    # start_date="20100101"
    #
    # start_date="20260101"
    # data_dict=get_all_data(start_date=start_date)
    # stock_list=pick_stock(dict=data_dict, day="20260119", strategy="量价多头")
    # back_test(data_dict=data_dict,start=start_date,end="20260119",max_hold_count=5,buy_count=15000,base_total_amount=100000.00)
    # print(stock_list)
# ["20100101","20101231"],["20110101","20111231"],["20120101","20121231"],["20130101","20131231"],["20140101","20141231"],
    arr=[["20150101","20151231"],["20160101","20161231"],["20170101","20171231"],["20180101","20181231"],["20190101","20191231"],["20200101","20201231"],["20210101","20211231"],["20220101","20221231"],["20230101","20231231"],["20240101","20241231"],["20250101","20251231"]]


    arr.reverse()

    avg_return_rate=0
    avg_win_rate=0
    for pair in arr:
        start_date=pair[0]
        end_date=pair[1]
        data_dict=get_all_data(start_date=start_date, end_date=end_date)
        total_return_rate,win_rate=back_test(data_dict=data_dict,start=start_date,end=end_date,max_hold_count=3,buy_count=20000,base_total_amount=100000.00,print_fail_reason=False)
        avg_return_rate+=total_return_rate
        avg_win_rate+=win_rate
    print(f"平均收益率:{round(avg_return_rate/len(arr)*100,2)}%")
    print(f"平均胜率:{round(avg_win_rate/len(arr),2)}%")


    # pair=["20200101","20201231"]
    # start_date=pair[0]
    # end_date=pair[1]
    # data_dict=get_all_data(start_date=start_date, end_date=end_date)
    # back_test(data_dict=data_dict,start=start_date,end=end_date,max_hold_count=5,buy_count=20000,base_total_amount=100000.00,print_detail=False)



# pick_stock
# back_test