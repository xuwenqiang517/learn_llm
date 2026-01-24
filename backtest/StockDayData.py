from dataclasses import dataclass

@dataclass(slots=True, frozen=False)
class StockDayData:
    date: str = None
    code: str = None
    name: str = None
    open: float = None
    close: float = None
    high: float = None
    low: float = None
    volume: int = None
    amount: float = None
    amplitude: float = None
    change_pct: float = None
    change: float = None
    turnover_rate: float = None
    pe: float = None
    pb: float = None
    market_cap: float = None
    circulating_market_cap: float = None
    change_3d: float = None
    change_5d: float = None
    change_10d: float = None
    consecutive_up_days: int = None
    volume_ratio: float = None
    ma5_gt_ma10: int = None
    ma10_gt_ma20: int = None
    vma5_gt_vma10: int = None
    vma10_gt_vma20: int = None
    volume_trend: bool = None
    vol_rank: int = None
    

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
