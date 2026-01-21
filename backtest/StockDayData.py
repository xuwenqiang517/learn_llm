from dataclasses import dataclass

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
    consecutive_up_days: int
    volume_ratio: float
    ma5_gt_ma10: int
    ma10_gt_ma20: int
    vma5_gt_vma10: int
    vma10_gt_vma20: int
    volume_trend: bool
    vol_rank: int
    

    def toString(self):
        return f"[stock_daily] {self.date} 开{self.open} 收{self.close} 涨跌幅:{self.change_pct}% 换手率:{self.turnover_rate} 量比:{self.volume_ratio} PE:{self.pe} PB:{self.pb} 多头:{self.volume_trend} 连涨:{self.consecutive_up_days}天 3日:{self.change_3d} 5日:{self.change_5d} 量排名:{self.vol_rank}"
    def full_info(self):
        return f"[stock_daily_full] {self.date} {self.code} {self.name} {self.open}->{self.close} 涨跌幅:{self.change_pct} 换手率:{self.turnover_rate} PE:{self.pe} PB:{self.pb} 量比:{self.volume_ratio} 多头:{self.volume_trend} 连涨:{self.consecutive_up_days}天 3日:{self.change_3d} 5日:{self.change_5d} 10日:{self.change_10d} 量排名:{self.vol_rank}"

    def __hash__(self):
        return hash(self.code)
    
    def __eq__(self, other):
        if not isinstance(other, StockDayData):
            return False
        return self.code == other.code
