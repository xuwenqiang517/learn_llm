from typing import OrderedDict, Dict, Any
from StockDayData import StockDayData
from TradingCalendar import TradingCalendar
from data_fetcher import get_all_data
from BuySellInfo import BuySellInfo


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

calendar = TradingCalendar()


def sell_rule(pre: StockDayData, cur: StockDayData, trade_info: BuySellInfo) -> bool:
    sell_flag = False
    sell_price = cur.close
    # 止损价
    stop_price=trade_info.stop_price()
    # 持仓收益率
    current_rate = trade_info.profit_rate(sell_price)
    if cur.open < stop_price:
        trade_info.sell_reason = f"[1]开盘{cur.open}低于止损价{stop_price}，直接卖出"
        sell_flag = True
        sell_price = cur.open
    # elif cur.low < stop_price:
    #     trade_info.sell_reason = f"[2]最低价{cur.low}低于止损价{stop_price}，直接卖出"
    #     sell_flag = True
    #     sell_price = stop_price
    elif pre.change_pct < -3:
        trade_info.sell_reason = f"[3]昨日跌{pre.change_pct}%,直接卖出"
        sell_flag = True
        sell_price = cur.open
    elif trade_info.hold_day > 2:
        if current_rate < 3:
            trade_info.sell_reason = f"[4]持仓{trade_info.hold_day}天,收益率:{current_rate}%,<3%"
            sell_flag = True
    elif trade_info.hold_day > 3:
        if current_rate < 10:
            trade_info.sell_reason = f"[5]持仓{trade_info.hold_day}天,收益率:{current_rate}%,<10%"
            sell_flag = True
    elif trade_info.hold_day > 5:
        if current_rate < 15:
            trade_info.sell_reason = f"[6]持仓{trade_info.hold_day}天,收益率:{current_rate}%,<15%"
            sell_flag = True
    elif current_rate > 25:
        trade_info.sell_reason = f"[7]持仓{trade_info.hold_day}天,收益率:{current_rate}%,>25%"
        sell_flag = True

    if sell_flag:
        trade_info.sell_price = sell_price
        trade_info.sell_day = cur.date

    return sell_flag

def pick_rule(info: StockDayData) -> bool:
        # if info.volume_trend==True and info.consecutive_up_days>=2 and info.change_pct < 9.8 and info.change_3d>5 and info.change_5d>10:
        #     return True 
        # if info.consecutive_up_days==3 and info.change_pct<6 and info.turnover_rate>5 and info.turnover_rate<15 and info.pe>0:
        #     return True #连涨
        # if info.volume_trend==True and info.change_pct<5 and info.change_pct>1:
        #     return True # 量价多头
        # if info.volume_ratio>1.5 and info.ma5_gt_ma10==1 and info.vma5_gt_vma10==1 and info.consecutive_up_days>=3 and info.change_3d>5:
        #     return True # 量价多头
        if info.consecutive_up_days>=3 and info.change_3d>5 and info.change_5d>10 and info.change_pct<5:
            return True # 量价多头
        return False

def pick_stock(dict: Dict[str, Dict[str, StockDayData]], day: str) -> list:
    rs_list=[]
    for code, data in dict.items():
        info=data.get(day)
        if info is None:
            continue
        # 过滤逻辑
        if pick_rule(info):
            rs_list.append(info)
    # 按涨幅最小排序
    rs_list.sort(key=lambda x: x.vol_rank, reverse=True)
    return rs_list

def get_data(data_dict: Dict[str, Dict[str, StockDayData]] = None, code: str = None, day: str = None) -> StockDayData:
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
    # 空闲金额
    total_free_amount=base_total_amount
    # 最高收益率
    # 最低收益率
    max_profit_rate=float("-inf")
    min_profit_rate=float("inf")
    # 胜率
    win_count=0
    # 总交易次数
    trade_count=0
    while cur_day<end:
        print_white("="*80+f"今天:{cur_day}"+ "="*80,print_detail)
        #选票
        stock_list=pick_stock(data_dict,pre_day)
        stock_len=len(stock_list)
        stock_list=stock_list[:max_hold_count]
        for stock in stock_list:
            print_blue(f"pick: //{stock_len} " + stock.full_info(),print_detail)
        # 判断买入
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
                if total_free_amount<count*stock_info_cur.open:
                    print_white(f"股票{stock.code}{stock.name} 资金不足，无法买入，跳过",print_detail)
                    continue
                hold_set.add(stock)
                buy_list.append(stock)
                trade=BuySellInfo(code=stock.code,name=stock.name,buy_day=cur_day,buy_price=round(stock_info_cur.open,2),sell_price=0.0,count=count)
                dict_info[stock.code]=trade
                print_green(f"buy:{trade.buy_string()} || {stock_info_cur.toString()}",print_detail)
                # 金额处理
                total_free_amount-=trade.buy_amount()

        # 判断卖出
        sell_list=[]
        for stock in hold_set:
            stock_info_pre=get_data(data_dict,stock.code,pre_day)
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            if stock_info_pre is None or stock_info_cur is None:
                print_white(f"股票{stock.code} 日期{cur_day},没有信息，跳过",print_detail)
                continue
            trade_info=dict_info[stock.code]
            # 确认要卖
            if sell_rule(stock_info_pre,stock_info_cur,trade_info):
                sell_list.append(stock)
                # 金额处理
                total_free_amount+=trade_info.sell_amount()
                # 记录交易次数
                trade_count+=1
                # 记录胜率
                if trade_info.sell_amount()>trade_info.buy_amount():
                    win_count+=1
                else:
                    print_yellow(f"{trade_info.sell_string()} || {stock_info_cur.toString()}",print_fail_reason)
        for stock in sell_list:
            hold_set.remove(stock)
            trade=dict_info[stock.code]
            print_red("sell:" + trade.sell_string(),print_detail)
            dict_info.pop(stock.code)
        # 持仓
        total_hold_amount=0
        for stock in hold_set:
            stock_info_cur=get_data(data_dict,stock.code,cur_day)
            if stock_info_cur is None:
                print_white(f"股票{stock.code} 日期{cur_day},没有信息，跳过",print_detail)
                continue
            trade_info=dict_info[stock.code]
            # 更新今天的收盘价
            trade_info.close=stock_info_cur.close
            # 更新持仓金额
            total_hold_amount+=trade_info.hold_amount()
            trade_info.hold_day+=1
            print_yellow(f"hold:{trade_info.hold_string()} ||{stock_info_cur.toString()}",print_detail)
        print_white(f"{cur_day} 空闲金额:{round(total_free_amount,2)} 持仓金额:{round(total_hold_amount,2)} 总金额:{round((total_free_amount+total_hold_amount),2)}",print_detail)
        # 更新最高收益率
        max_profit_rate=max(max_profit_rate, round((total_free_amount+total_hold_amount)/base_total_amount*100,2))
        # 更新最低收益率
        min_profit_rate=min(min_profit_rate, round((total_free_amount+total_hold_amount)/base_total_amount*100,2))
        pre_day=cur_day
        cur_day=calendar.next(cur_day)

    total_amount=total_free_amount+total_hold_amount
    total_return_rate=round((total_amount-base_total_amount)/base_total_amount*100,2)
    win_rate=round(win_count/trade_count*100,2) if trade_count>0 else 0
    rs_str=f"{start},{end} 资金{base_total_amount}->{total_amount} 总收益率:{total_return_rate}% 最高:{max_profit_rate}% 最低:{min_profit_rate}% 胜率:{win_rate}% 总交易次数:{trade_count}"
    print_red( rs_str,True)
    return total_return_rate,win_rate



def test_cur():
    data_dict=get_all_data(start_date="20260101", end_date=calendar.last())
    stock_list=pick_stock(dict=data_dict, day="20260119")
    for stock in stock_list:
        print(stock.full_info())

def test_10(arr=[["20150101","20151231"],["20160101","20161231"],["20170101","20171231"],["20180101","20181231"],["20190101","20191231"],["20200101","20201231"],["20210101","20211231"],["20220101","20221231"],["20230101","20231231"],["20240101","20241231"],["20250101","20251231"]]):
    arr.reverse()
    avg_return_rate=0
    avg_win_rate=0
    for pair in arr:
        start_date=pair[0]
        end_date=pair[1]
        data_dict=get_all_data(start_date=start_date, end_date=end_date)
        total_return_rate,win_rate=back_test(data_dict=data_dict,start=start_date,end=end_date
        ,max_hold_count=5,buy_count=20000,base_total_amount=100000.00
        ,print_fail_reason=False)
        avg_return_rate+=total_return_rate
        avg_win_rate+=win_rate
    print(f"平均收益率:{round(avg_return_rate/len(arr),2)}%")
    print(f"平均胜率:{round(avg_win_rate/len(arr),2)}%")

def test_1():
    # pair=["20200101","20201231"]
    pair=["20250601","20251231"]
    # pair=["20251201","20251231"]
    start_date=pair[0]
    end_date=pair[1]
    data_dict=get_all_data(start_date=start_date, end_date=end_date)
    # print_white(data_dict["600519"])
    # print_white(data_dict["000001"]["20201231"])
    back_test(data_dict=data_dict,start=start_date,end=end_date
    ,max_hold_count=5,buy_count=20000,base_total_amount=100000.00
    ,print_detail=True)

if __name__ == "__main__":
    test_cur()
    # test_10([["20200101","20201231"],["20210101","20211231"],["20220101","20221231"],["20230101","20231231"],["20240101","20241231"],["20250101","20251231"]])
    # test_10()

    # test_1()