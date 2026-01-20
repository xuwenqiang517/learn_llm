from datetime import date, datetime
import akshare as ak
from cache_utils import get_with_cache

class TradingCalendar:
    day_dict = None
    today = None
    
    def __init__(self) -> None:
        self.today = date.today().strftime("%Y%m%d")
        self.day_dict = get_with_cache("trading_days.pkl", self.load)
        
    def load(self):
        df = ak.tool_trade_date_hist_sina()
        rs = {}
        dates = []
        include_today = datetime.now().hour > 15
        
        for d in df["trade_date"]:
            dt = d.strftime("%Y%m%d")
            if dt <= self.today:
                if not include_today and dt == self.today:
                    continue
                dates.append(dt)
        
        for i, date in enumerate(dates):
            prev_day = dates[i-1] if i > 0 else "000000"
            next_day = dates[i+1] if i < len(dates)-1 else "000000"
            rs[date] = [prev_day, next_day]
        
        return rs

    def last(self):
        return list(self.day_dict.keys())[-1]

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
