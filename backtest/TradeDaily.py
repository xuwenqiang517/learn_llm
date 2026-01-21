from dataclasses import dataclass, field
from BuySellInfo import BuySellInfo
from TradingCalendar import TradingCalendar
from typing import OrderedDict, Dict, Any
from StockDayData import StockDayData

class TradeDaily:
    # 最大持仓数量
    max_hold_count:int
    # 最大持仓金额
    max_hold_amount:float
    # 初始资金
    base_amount:float
    # 上一个交易日
    pre_day:str=""
    # 当前日期
    curr_day:str=""
    # 数据字典
    data_dict:dict={}

    enable_log:bool=False
    
    # 原始股票列表
    original_stock_list: list = []
    # 截断股票列表
    cat_stock_list: list = []
    # 持仓字典
    hold_dict:dict={}
    # 卖出原因
    sell_reason_dict:dict={}
    # 可用金额
    total_free_amount:float=0.0
    # 卖出次数
    sell_count:int=0
    # 胜利次数
    win_count:int=0
    # 最大收益率
    max_profit_rate:float=0.0
    # 最小收益率
    min_profit_rate:float=100000.0

    pick_count=0
    pick_stock_count=0

    def __init__(self,max_hold_count:int,max_hold_amount:float,base_amount:float,pre_day:str="",curr_day:str="",data_dict:dict={},enable_log:bool=False) -> None:
        self.max_hold_count=max_hold_count
        self.max_hold_amount=max_hold_amount
        self.base_amount=base_amount
        self.total_free_amount=base_amount
        self.pre_day=pre_day
        self.curr_day=curr_day
        self.data_dict=data_dict
        self.enable_log=enable_log

    def do_pick(self,func:callable,max_hold_count:int)->list:
        self.original_stock_list=func()
        self.cat_stock_list=self.original_stock_list[:max_hold_count]
        self.pick_stock_count+=len(self.original_stock_list)
        self.pick_count+=1
        # for stock in self.cat_stock_list:
        #     self.print_blue(f"pick: {len(self.cat_stock_list)}/{len(self.original_stock_list)} " + stock.full_info(),self.enable_log)
        return self.cat_stock_list

    def do_buy(self):
        self.buy_list=[]
        for stock in self.cat_stock_list:
            if len(self.hold_dict)==self.max_hold_count:
                break
            cur_stock=self.get_data(self.data_dict,stock.code,self.curr_day)
            if cur_stock is None :
                continue
            buy_price=cur_stock.open
            buy_count=int(self.max_hold_amount/buy_price)//100*100
            buy_amount=buy_price*buy_count
            if buy_count == 0:
                continue
            if stock.code in self.hold_dict.keys():
                continue
                # self.hold_dict[stock.code]=stock
            if self.total_free_amount<buy_amount:
                continue
            # 购买股票
            self.total_free_amount-=buy_amount
            buy_sell_info=BuySellInfo(code=stock.code,name=stock.name,buy_day=self.curr_day,buy_price=buy_price,sell_price=0.0,count=buy_count)
            self.hold_dict[stock.code]=buy_sell_info

            self.print_green(f"buy: {buy_sell_info.buy_string()} || {stock.toString()}",self.enable_log)


    def do_sell(self):
        sell_list=[]
        for code,buy_sell_info in self.hold_dict.items():
            pre_stock=self.get_data(self.data_dict,code,self.pre_day)
            cur_stock=self.get_data(self.data_dict,code,self.curr_day)
            if pre_stock is None or cur_stock is None:
                continue
            buy_sell_info=self.hold_dict[code]
            sell_reason=self.sell_rule(pre_stock,cur_stock,buy_sell_info)
            if len(sell_reason)>0:
                self.sell_reason_dict[sell_reason]=self.sell_reason_dict.get(sell_reason,0)+1
                sell_list.append(buy_sell_info)
                # 金额
                self.total_free_amount+=buy_sell_info.sell_amount()
                # 卖出次数
                self.sell_count+=1
                if buy_sell_info.sell_amount()>buy_sell_info.buy_amount():
                    self.win_count+=1
        for buy_sell_info in sell_list:
            self.hold_dict.pop(buy_sell_info.code)
            self.print_red(f"sell: {buy_sell_info.sell_string()}",self.enable_log)
    
    def do_hold(self):
        total_hold_amount=0
        for code,buy_sell_info in self.hold_dict.items():
            # pre_stock=self.get_data(self.data_dict,code,self.pre_day)
            cur_stock=self.get_data(self.data_dict,code,self.curr_day)
            if cur_stock is None:
                continue
            buy_sell_info=self.hold_dict[code]
            # 更新今天的收盘价
            buy_sell_info.close=cur_stock.close
            buy_sell_info.hold_day+=1
            # 更新持仓金额
            total_hold_amount+=buy_sell_info.hold_amount()
            self.print_yellow(f"hold: {buy_sell_info.hold_string()} {cur_stock.toString()}",self.enable_log)

        total_amount=total_hold_amount+self.total_free_amount
        self.print_white(f"{self.curr_day} 可用:{self.total_free_amount} 持仓: {total_hold_amount} 总资金: {total_amount}",self.enable_log)
        # 更新最大收益率
        self.max_profit_rate=max(self.max_profit_rate,round((total_amount/self.base_amount)*100,2))
        # 更新最小收益率
        self.min_profit_rate=min(self.min_profit_rate,round((total_amount/self.base_amount)*100,2))

    def end(self)->tuple[float,float]:
        win_rate=round(self.win_count/self.sell_count*100,2) if self.sell_count>0 else 0
        total_hold_amount=sum([buy_sell_info.hold_amount() for buy_sell_info in self.hold_dict.values()])
        rate=round(((self.total_free_amount+total_hold_amount)/self.base_amount)*100,2)
        self.print_white(f"{self.curr_day} 总收益率: {rate}% 最大收益率: {self.max_profit_rate}% 最小收益率: {self.min_profit_rate}% 胜率: {win_rate}% 卖出统计: {sorted(self.sell_reason_dict.items(),key=lambda x:x[0])} 平均选票数量: {self.pick_stock_count/self.pick_count}" )
        return rate,win_rate

    def update_day(self,day:str):
        self.pre_day=self.curr_day
        self.curr_day=day
    

    def print_green(self,msg,enable=True):
        if enable:
            print(f"\033[92m{msg}\033[0m")

    def print_red(self,msg,enable=True):
        if enable:
            print(f"\033[91m{msg}\033[0m")

    def print_yellow(self,msg,enable=True):
        if enable:
            print(f"\033[93m{msg}\033[0m")

    def print_blue(self,msg,enable=True):
        if enable:
            print(f"\033[94m{msg}\033[0m")

    def print_white(self,msg,enable=True):
        if enable:
            print(f"\033[97m{msg}\033[0m")
    def get_data(self,data_dict: Dict[str, Dict[str, StockDayData]] = None, code: str = None, day: str = None) -> StockDayData:
        return data_dict.get(code).get(day) if data_dict.get(code) is not None else None


    def sell_rule(self,pre: StockDayData, cur: StockDayData, buy_sell_info: BuySellInfo) -> int:
        sell_flag = ""
        sell_price = cur.close
        # 止损价
        stop_price=buy_sell_info.stop_price()
        # 持仓收益率
        current_rate = buy_sell_info.profit_rate(sell_price)
        if cur.open < stop_price:
            buy_sell_info.sell_reason = f"[1]开盘{cur.open}低于止损价{stop_price}，直接卖出"
            sell_flag = "止损价"
            sell_price = cur.open
        # elif cur.low < stop_price:
        #     buy_sell_info.sell_reason = f"[2]最低价{cur.low}低于止损价{stop_price}，直接卖出"
        #     sell_flag = True
        #     sell_price = stop_price
        elif pre.change_pct < -5:
            buy_sell_info.sell_reason = f"[3]昨日跌{pre.change_pct}%,直接卖出"
            sell_flag = "昨日跌5%"
            sell_price = cur.open
        # elif buy_sell_info.hold_day > 2:
        #     if current_rate < 3:
        #         buy_sell_info.sell_reason = f"[4]持仓{buy_sell_info.hold_day}天,收益率:{current_rate}%,<3%"
        #         sell_flag = "持有2天收益<3%"
        # elif buy_sell_info.hold_day > 3:
        #     if current_rate < 10:
        #         buy_sell_info.sell_reason = f"[5]持仓{buy_sell_info.hold_day}天,收益率:{current_rate}%,<10%"
        #         sell_flag = True
        elif buy_sell_info.hold_day > 6:
            if current_rate < 20:
                buy_sell_info.sell_reason = f"[6]持仓{buy_sell_info.hold_day}天,收益率:{current_rate}%,<20%"
                sell_flag = "持有6天收益<20%"
        # elif current_rate > 40:
        #     buy_sell_info.sell_reason = f"[7]持仓{buy_sell_info.hold_day}天,收益率:{current_rate}%,>40%"
        #     sell_flag = True

        if len(sell_flag)>0:
            buy_sell_info.sell_price = sell_price
            buy_sell_info.sell_day = cur.date

        return sell_flag
    