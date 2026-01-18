# 维护交易日信息 用dict存，key为交易日转日期后的int，value为一个一位数组，[0]是上一个交易日，[1]是下一个交易日

from typing import OrderedDict
import akshare as ak
from datetime import date, datetime

trade_days = {}

# 到今天前到有效交易日 包含今天
def get_trading_days(days: int = None) -> OrderedDict:
    today = int(date.today().strftime("%Y%m%d"))
    df = ak.tool_trade_date_hist_sina()
    rs = OrderedDict()
    pre = 0
    for d in df["trade_date"]:
        dt = int(d.strftime("%Y%m%d"))
        if dt <= today:
            rs[dt]=pre
            pre=dt
    return rs


