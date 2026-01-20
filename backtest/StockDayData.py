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
