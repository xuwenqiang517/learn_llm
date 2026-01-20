from dataclasses import dataclass

@dataclass
class BuySellInfo:
    code: str
    name: str
    buy_day: str
    buy_price: float
    sell_day: str = ""
    sell_price: float = 0.0
    count: int = 0
    close: float = 0.0
    hold_day: int = 0
    sell_reason: str = ""

    def buy_value(self):
        return round(self.count * self.buy_price, 2)

    def sell_value(self):
        return round(self.count * self.sell_price, 2)

    def profit_rate(self):
        return round((self.sell_value() / self.buy_value() - 1) * 100, 2)

    def profit_amount(self):
        if self.sell_price > 0:
            return round(self.sell_value() - self.buy_value(), 2)
        else:
            return round(self.hold_value() - self.buy_value(), 2)

    def hold_value(self):
        return round(self.close * self.count, 2)

    def hold_profit_rate(self):
        if self.buy_value() == 0:
            return 0.0
        return round((self.hold_value() / self.buy_value() - 1) * 100, 2)

    def buy_string(self):
        return f"股票{self.code} {self.name} 买入:{self.buy_day} 买入价：{self.buy_price} 股数:{self.count} 金额{self.buy_value()}"

    def sell_string(self):
        return f"股票{self.code} {self.name} 时间:{self.buy_day}-{self.sell_day}（{self.hold_day}天） 价格:{self.buy_price} -> {self.sell_price} 涨跌幅:{self.profit_rate()}% 盈亏:{self.profit_amount()}  卖出原因:{self.sell_reason}"

    def hold_string(self):
        return f"股票{self.code} {self.name} 买入:{self.buy_day} 买入价:{self.buy_price} 收盘价:{self.close} 持仓:{self.hold_day}天  持仓收益率{self.hold_profit_rate()}% 盈亏:{self.profit_amount()}"
