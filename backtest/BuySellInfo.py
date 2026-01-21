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

    # 买入金额
    def buy_amount(self):
        return round(self.count * self.buy_price, 2)

    # 卖出金额
    def sell_amount(self):
        return round(self.count * self.sell_price, 2)

    # 收益率
    def profit_rate(self, price: float = None):
        if price is not None:
            return round((price * self.count / self.buy_amount() - 1) * 100, 2)
        if self.sell_price > 0:
            return round((self.sell_amount() / self.buy_amount() - 1) * 100, 2)
        else:
            return round((self.hold_amount() / self.buy_amount() - 1) * 100, 2)
    
    
    # 收益金额
    def profit_amount(self):
        if self.sell_price > 0:
            return round(self.sell_amount() - self.buy_amount(), 2)
        else:
            return round(self.hold_amount() - self.buy_amount(), 2)
    # 止损价
    def stop_price(self,rate:float=0.05):
        return round(self.buy_price * (1 - rate),2)
    # 持仓金额
    def hold_amount(self):
        return round(self.close * self.count, 2)


    def buy_string(self):
        return f"[buy_string]{self.name} day:{self.buy_day} 买入：{self.buy_price}*{self.count}={self.buy_amount()} "

    def sell_string(self):
        return f"[sell_string]{self.name} day:{self.buy_day}->{self.sell_day}（{self.hold_day}天 价格:{self.buy_price}->{self.sell_price} 涨跌幅:{self.profit_rate()}% 盈亏:{self.profit_amount()} 卖出价:{self.sell_price}*{self.count}={self.sell_amount()}  卖出原因:{self.sell_reason}"

    def hold_string(self):
        return f"[hold_string]{self.name} 买入:{self.buy_price}*{self.count}={self.buy_amount()} ({self.buy_day}) 持仓:{self.hold_day}天 金额:{self.hold_amount()}*{self.close}={self.hold_amount()} 收益率{self.profit_rate()}% 盈亏:{self.profit_amount()}"
