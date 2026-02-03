"""
高性能回测系统 - 单文件优化版本
结合向量化计算、并行处理和缓存机制
"""

import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from TradingCalendar import TradingCalendar
from BuySellInfo import BuySellInfo
import os
import time
import pickle
import hashlib
from data import get_full_data
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = f"{PROJECT_ROOT}/.temp/data/cache"

# 详细日志相关函数
def init_detailed_log(log_path: str):
    """初始化详细日志文件"""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"详细日志文件已创建: {log_path}")
    return log_file

def write_detailed_log(log_file, msg: str, print_to_console: bool = True):
    """写入详细日志"""
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()
    if print_to_console:
        print(msg)

def close_detailed_log(log_file):
    """关闭详细日志文件"""
    if log_file:
        log_file.close()


class StrategyConfig:
    """策略配置类"""
    
    def __init__(self, 
                 name: str = "默认策略",
                 consecutive_up_days_min: int = 3,
                 change_3d_min: float = 5.0,
                 change_5d_min: float = 10.0,
                 daily_change_max: float = 5.0,
                 stop_loss_rate: float = 0.05,
                 prev_day_drop_threshold: float = -5.0,
                 max_hold_days: int = 6,
                 min_profit_rate: float = 20.0,
                 buy_amount_per_stock: int = 20000,
                 max_hold_count: int = 5):
        
        self.name = name
        self.consecutive_up_days_min = consecutive_up_days_min
        self.change_3d_min = change_3d_min
        self.change_5d_min = change_5d_min
        self.daily_change_max = daily_change_max
        self.stop_loss_rate = stop_loss_rate
        self.prev_day_drop_threshold = prev_day_drop_threshold
        self.max_hold_days = max_hold_days
        self.min_profit_rate = min_profit_rate
        self.buy_amount_per_stock = buy_amount_per_stock
        self.max_hold_count = max_hold_count
    
    def get_hash(self) -> str:
        """生成策略哈希值用于缓存"""
        params_str = f"{self.consecutive_up_days_min}_{self.change_3d_min}_{self.change_5d_min}_{self.daily_change_max}"
        return hashlib.md5(params_str.encode()).hexdigest()[:16]


class HighPerformanceBacktester:
    """高性能回测器 - 单文件优化版本"""
    
    def __init__(self, use_cache: bool = True):
        self.calendar = TradingCalendar()
        self.use_cache = use_cache
        self.date_dict = self._load_and_prepare_data()
        self._precompute_indices()
    
    def _load_and_prepare_data(self) -> Dict[str, Dict[str, Dict]]:
        """加载并预处理数据 - 优化版本"""
        cache_file = f"{CACHE_DIR}/stock_data_cache.pkl"
        
        if self.use_cache and os.path.exists(cache_file):
            print("从缓存加载数据...")
            start_time = time.time()
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time
            print(f"缓存加载耗时: {load_time:.4f}秒")
            return data
        
        print("加载原始数据...")
        start_time = time.time()
        df = get_full_data()
        
        # 优化：使用更高效的数据转换方法
        date_dict = {}
        # 使用groupby加速数据分组
        grouped = df.groupby(level=1)  # 按日期分组
        
        for date_str, group_df in grouped:
            date_dict[date_str] = group_df.to_dict('index')
        
        # 预计算技术指标
        date_dict = self._precompute_indicators(date_dict)
        
        # 保存缓存
        if self.use_cache:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(date_dict, f)
        
        total_time = time.time() - start_time
        print(f"数据加载和预处理耗时: {total_time:.4f}秒")
        
        return date_dict
    
    def _precompute_indicators(self, date_dict: Dict) -> Dict:
        """预处理技术指标 - 修复NaN值并确保数据完整性"""
        print("预处理技术指标...")
        
        # 直接使用原始数据中的指标，修复NaN值
        for date_str, stocks in date_dict.items():
            for code, data in stocks.items():
                # 修复NaN值
                if pd.isna(data.get("change_3d")):
                    data["change_3d"] = 0.0
                if pd.isna(data.get("change_5d")):
                    data["change_5d"] = 0.0
                if pd.isna(data.get("consecutive_up_days")):
                    data["consecutive_up_days"] = 0
                if pd.isna(data.get("vol_rank")):
                    data["vol_rank"] = 0
                if pd.isna(data.get("涨跌幅")):
                    data["涨跌幅"] = 0.0
        
        return date_dict
    
    def _precompute_indices(self):
        """预计算索引加速访问"""
        self.all_dates = sorted(self.date_dict.keys())
        
        # 预计算每个日期的合格股票（按成交量排名）
        self.date_qualified_stocks = {}
        
        # 预计算技术指标缓存（性能优化）
        self.technical_indicators_cache = {}
        
        for date_str in self.all_dates:
            stocks = self.date_dict.get(date_str, {})
            if stocks:
                # 计算真正的成交量排名（按成交量降序）
                stocks_with_volume = [(code, data.get("成交量", 0)) for code, data in stocks.items()]
                stocks_with_volume.sort(key=lambda x: x[1], reverse=True)
                
                # 为每只股票分配排名
                for rank, (code, volume) in enumerate(stocks_with_volume, 1):
                    if code in stocks:
                        stocks[code]["vol_rank"] = rank
                
                # 按成交量排名排序
                sorted_stocks = sorted(
                    [(code, data.get("vol_rank", 0)) for code, data in stocks.items()],
                    key=lambda x: x[1]
                )
                self.date_qualified_stocks[date_str] = [code for code, _ in sorted_stocks]
                
                # 预计算技术指标
                self.technical_indicators_cache[date_str] = {}
                for code, data in stocks.items():
                    self.technical_indicators_cache[date_str][code] = {
                        "consecutive_up_days": data.get("consecutive_up_days", 0),
                        "change_3d": data.get("change_3d", 0),
                        "change_5d": data.get("change_5d", 0),
                        "涨跌幅": data.get("涨跌幅", 0),
                        "vol_rank": data.get("vol_rank", 0)
                    }
    
    def fast_backtest(self, start: str, end: str, config: StrategyConfig) -> Dict[str, Any]:
        """快速回测 - 优化版本"""
        
        pre_day = self.calendar.pre(start)
        cur_day = self.calendar.next(pre_day)
        
        # 初始化状态
        hold_set = set()
        dict_info = {}
        total_free_amount = 100000.00
        
        # 性能统计
        stats = {
            'max_profit_rate': float("-inf"),
            'max_drawdown': 0.0,
            'max_total_amount': total_free_amount,
            'win_count': 0,
            'sell_count': 0,
            'sell_reason_dict': {}
        }
        
        # 主回测循环
        while cur_day <= end:
            # 确保当前日期有数据
            if cur_day not in self.date_dict:
                break
            
            # 1. 快速选股
            selected_codes, _ = self._fast_pick_stock(pre_day, config)
            
            # 2. 买入逻辑
            for code in selected_codes:
                if len(hold_set) >= config.max_hold_count or code in hold_set:
                    continue
                
                cur_data = self._get_stock_data_fast(code, cur_day)
                if not cur_data:
                    continue
                
                buy_amount = self.calculate_dynamic_buy_amount(
                    free_amount=total_free_amount,
                    hold_count=len(hold_set),
                    max_hold=config.max_hold_count,
                    selected_count=len(selected_codes),
                    base_amount=config.buy_amount_per_stock
                )
                count = self._calculate_buy_count(cur_data["open"], buy_amount)
                if count == 0 or total_free_amount < count * cur_data["open"]:
                    continue
                
                # 执行买入
                hold_set.add(code)
                buy_price = round(cur_data["open"], 2)
                trade = BuySellInfo(code=code, name="", buy_day=cur_day, 
                                  buy_price=buy_price, sell_price=0.0, count=count)
                dict_info[code] = trade
                total_free_amount -= trade.buy_amount()
            
            # 3. 卖出逻辑
            sell_list = []
            for code in list(hold_set):
                pre_data = self._get_stock_data_fast(code, pre_day)
                cur_data = self._get_stock_data_fast(code, cur_day)
                
                if not pre_data or not cur_data:
                    continue
                
                trade_info = dict_info[code]
                sell_flag = self._apply_sell_rule(pre_data, cur_data, trade_info, cur_day, config)
                
                if sell_flag:
                    stats['sell_reason_dict'][sell_flag] = stats['sell_reason_dict'].get(sell_flag, 0) + 1
                    sell_list.append(code)
                    total_free_amount += trade_info.sell_amount()
                    stats['sell_count'] += 1
                    
                    if trade_info.sell_amount() > trade_info.buy_amount():
                        stats['win_count'] += 1
            
            # 移除已卖出股票
            for code in sell_list:
                hold_set.remove(code)
                dict_info.pop(code)
            
            # 4. 更新持仓和统计
            total_hold_amount = 0
            for code in hold_set:
                cur_data = self._get_stock_data_fast(code, cur_day)
                if not cur_data:
                    continue
                
                trade_info = dict_info[code]
                trade_info.close = cur_data["close"]
                total_hold_amount += trade_info.hold_amount()
                trade_info.hold_day += 1
            
            # 5. 更新统计指标
            current_total_amount = total_free_amount + total_hold_amount
            stats['max_total_amount'] = max(stats['max_total_amount'], current_total_amount)
            current_drawdown = (stats['max_total_amount'] - current_total_amount) / stats['max_total_amount'] * 100
            stats['max_drawdown'] = max(stats['max_drawdown'], current_drawdown)
            stats['max_profit_rate'] = max(stats['max_profit_rate'], 
                                         round(current_total_amount / 100000.00 * 100, 2))
            
            # 移动到下一天
            pre_day = cur_day
            cur_day = self.calendar.next(cur_day)
        
        # 编译最终结果
        total_hold_amount = sum(trade.hold_amount() for trade in dict_info.values())
        total_amount = total_free_amount + total_hold_amount
        total_return_rate = round((total_amount - 100000.00) / 100000.00 * 100, 2)
        win_rate = round(stats['win_count'] / stats['sell_count'] * 100, 2) if stats['sell_count'] > 0 else 0
        
        return {
            "strategy_name": config.name,
            "total_return_rate": total_return_rate,
            "max_profit_rate": stats['max_profit_rate'],
            "max_drawdown": round(stats['max_drawdown'], 2),
            "win_rate": win_rate,
            "total_trades": stats['sell_count'],
            "sell_reasons": stats['sell_reason_dict'],
            "final_amount": total_amount,
            "start_date": start,
            "end_date": end
        }
    
    def _fast_pick_stock(self, date: str, config: StrategyConfig) -> Tuple[List[str], int]:
        """快速选股 - 返回(选中的股票列表, 符合条件总数)"""
        if date not in self.technical_indicators_cache:
            return [], 0
        
        indicators = self.technical_indicators_cache[date]
        
        max_count = config.max_hold_count
        
        daily_change_max = config.daily_change_max
        change_3d_min = config.change_3d_min
        change_5d_min = config.change_5d_min
        consecutive_up_days_min = config.consecutive_up_days_min
        
        qualified_stocks_with_rank = []
        
        for code_tuple, data in indicators.items():
            if (data["涨跌幅"] < daily_change_max and
                data["change_3d"] > change_3d_min and
                data["change_5d"] > change_5d_min and
                data["consecutive_up_days"] >= consecutive_up_days_min):
                
                stock_code = code_tuple[0] if isinstance(code_tuple, tuple) else code_tuple
                qualified_stocks_with_rank.append((stock_code, data["vol_rank"]))
        
        qualified_stocks_with_rank.sort(key=lambda x: x[1], reverse=True)
        
        total_qualified = len(qualified_stocks_with_rank)
        selected = [code for code, _ in qualified_stocks_with_rank[:max_count]]
        
        return selected, total_qualified
    
    def _get_stock_data_fast(self, code: str, date: str) -> Dict:
        """快速获取股票数据 - O(1)时间复杂度"""
        if date not in self.date_dict:
            return {}
        
        day_data = self.date_dict[date]
        
        # O(1)时间复杂度直接访问
        found_data = day_data.get((code, date))
        
        if not found_data:
            return {}
        
        return {
            "open": found_data.get("开盘", 0),
            "close": found_data.get("收盘", 0),
            "high": found_data.get("最高", 0),
            "low": found_data.get("最低", 0),
            "volume": found_data.get("成交量", 0),
            "change_pct": found_data.get("涨跌幅", 0),
            "consecutive_up_days": found_data.get("consecutive_up_days", 0),
            "change_3d": found_data.get("change_3d", 0),
            "change_5d": found_data.get("change_5d", 0),
            "vol_rank": found_data.get("vol_rank", 0)
        }
    
    def _calculate_buy_count(self, price: float, buy_amount: int) -> int:
        """计算买入数量"""
        return int(buy_amount / price) // 100 * 100 if price > 0 else 0
    
    def calculate_dynamic_buy_amount(self, free_amount: int, hold_count: int, 
                                     max_hold: int, selected_count: int,
                                     base_amount: int) -> int:
        """
        动态计算单只股票可买入金额
        
        策略：
        - 剩余资金 / 剩余可用仓位 = 每只可买金额
        - 如果选中股票数少于剩余仓位，按实际选中数分配
        """
        remaining_slots = max_hold - hold_count
        if remaining_slots <= 0:
            return 0
        
        effective_count = min(selected_count, remaining_slots)
        if effective_count <= 0:
            return 0
        
        return int(free_amount / effective_count)
    
    def _apply_sell_rule(self, pre_data: Dict, cur_data: Dict, 
                        trade_info: BuySellInfo, cur_day: str, 
                        config: StrategyConfig) -> str:
        """应用卖出规则"""
        stop_price = trade_info.stop_price(config.stop_loss_rate)
        current_rate = trade_info.profit_rate(cur_data["close"])
        
        if cur_data["open"] < stop_price:
            trade_info.sell_reason = f"止损价({config.stop_loss_rate*100}%)"
            trade_info.sell_price = cur_data["open"]
            trade_info.sell_day = cur_day
            return f"止损价({config.stop_loss_rate*100}%)"
        elif pre_data["change_pct"] < config.prev_day_drop_threshold:
            trade_info.sell_reason = f"昨日跌{abs(config.prev_day_drop_threshold)}%"
            trade_info.sell_price = cur_data["open"]
            trade_info.sell_day = cur_day
            return f"昨日跌{abs(config.prev_day_drop_threshold)}%"
        elif trade_info.hold_day > config.max_hold_days and current_rate < config.min_profit_rate:
            trade_info.sell_reason = f"持有超期({config.max_hold_days}天,收益<{config.min_profit_rate}%)"
            trade_info.sell_price = cur_data["close"]
            trade_info.sell_day = cur_day
            return f"持有超期({config.max_hold_days}天,收益<{config.min_profit_rate}%)"
        
        return ""


def detailed_backtest_with_logging(start: str, end: str, config: StrategyConfig, 
                                  log_file_path: str = None,
                                  print_detail: bool = True,
                                  print_fail_reason: bool = True) -> Dict[str, Any]:
    """
    详细日志回测 - 输出每日选股买卖持仓信息
    与 backtest_df_optimized.py 的详细日志格式完全对齐
    """
    import time
    
    log_file = init_detailed_log(log_file_path) if log_file_path else None
    
    def strip_ansi(msg: str) -> str:
        """去除ANSI颜色控制符"""
        import re
        return re.sub(r'\033\[[0-9;]*m', '', msg)
    
    def write_log(msg: str, print_to_console: bool = True):
        """写入日志"""
        if log_file:
            log_file.write(strip_ansi(msg) + "\n")
            log_file.flush()
        if print_to_console:
            print(msg)
    
    def write_green(msg: str):
        write_log(f"\033[92m{msg}\033[0m", print_detail)
    
    def write_red(msg: str):
        write_log(f"\033[91m{msg}\033[0m", print_detail)
    
    def write_yellow(msg: str, print_to_console: bool = True):
        write_log(f"\033[93m{msg}\033[0m", print_to_console)
    
    def write_blue(msg: str):
        write_log(f"\033[94m{msg}\033[0m", print_detail)
    
    def write_white(msg: str):
        write_log(f"\033[97m{msg}\033[0m", print_detail)
    
    write_white("=" * 80)
    write_white(f"=== 详细回测开始 ===")
    write_white(f"策略: {config.name}")
    write_white(f"回测期间: {start} 到 {end}")
    write_white(f"初始资金: 100000.00")
    write_white(f"选股条件: 连涨≥{config.consecutive_up_days_min}天 | 3日涨>{config.change_3d_min}% | 5日涨>{config.change_5d_min}% | 今日涨幅<{config.daily_change_max}%")
    write_white(f"卖出条件: 止损{config.stop_loss_rate*100}% | 昨跌{int(abs(config.prev_day_drop_threshold))}% | 持仓>{config.max_hold_days}天收益<{config.min_profit_rate}%")
    write_white(f"资金管理: 动态分配 | 基准每只{config.buy_amount_per_stock}元 | 最多{config.max_hold_count}只")
    write_white("=" * 80)
    
    backtester = HighPerformanceBacktester(use_cache=True)
    
    pre_day = backtester.calendar.pre(start)
    cur_day = backtester.calendar.next(pre_day)
    
    hold_set = set()
    dict_info = {}
    total_free_amount = 100000.00
    
    stats = {
        'max_profit_rate': float("-inf"),
        'max_drawdown': 0.0,
        'max_total_amount': total_free_amount,
        'win_count': 0,
        'sell_count': 0,
        'sell_reason_dict': {}
    }
    
    day_count = 0
    
    while cur_day <= end:
        day_count += 1
        
        if cur_day not in backtester.date_dict:
            break
        
        next_day = backtester.calendar.next(cur_day)
        if next_day > end or next_day not in backtester.date_dict:
            if next_day not in backtester.date_dict:
                write_white(f"日期{next_day}没有数据，结束回测")
            else:
                write_white(f"下个交易日{next_day}超过回测结束日期{end}，结束回测")
            break
        
        if print_detail and (day_count % 10 == 0):
            write_white(f"进度: 第{day_count}天 - 当前日期: {cur_day}")
        else:
            write_white("=" * 80 + f"{cur_day}盘后选股 → {next_day}交易" + "=" * 80)
        
        selected_codes, total_qualified = backtester._fast_pick_stock(cur_day, config)
        selected_codes = selected_codes[:config.max_hold_count]
        
        write_white(f"pick: 候选{total_qualified}只 | 选中{len(selected_codes)}只")
        
        for i, code in enumerate(selected_codes):
            cur_data = backtester._get_stock_data_fast(code, cur_day)
            
            if cur_data:
                write_blue(f"pick: //{len(selected_codes)} {code} {cur_day} 开{cur_data['open']:.2f} 收{cur_data['close']:.2f} 涨{cur_data['change_pct']:.2f}% 连涨{cur_data.get('consecutive_up_days', 0)}天 量比{cur_data.get('vol_rank', 0):.1f} 3日{cur_data.get('change_3d', 0):.2f} 5日{cur_data.get('change_5d', 0):.2f}")
        
        # 2. 买入逻辑：用明天的数据买入今天选出的票
        for code in selected_codes:
            if len(hold_set) >= config.max_hold_count or code in hold_set:
                continue
            
            next_data = backtester._get_stock_data_fast(code, next_day)
            if not next_data:
                write_white(f"股票{code} 日期{next_day}没有信息，跳过")
                continue
            
            buy_amount = backtester.calculate_dynamic_buy_amount(
                free_amount=total_free_amount,
                hold_count=len(hold_set),
                max_hold=config.max_hold_count,
                selected_count=len(selected_codes),
                base_amount=config.buy_amount_per_stock
            )
            count = backtester._calculate_buy_count(next_data["open"], buy_amount)
            if count == 0:
                write_white(f"股票{code} 股价{next_data['open']:.2f}过高，无法买入，跳过")
                continue
            if total_free_amount < count * next_data["open"]:
                write_white(f"股票{code} 资金不足(需要{count * next_data['open']:.2f}，剩余{total_free_amount:.2f})，跳过")
                continue
            
            # 执行买入
            hold_set.add(code)
            buy_price = round(next_data["open"], 2)
            trade = BuySellInfo(code=code, name="", buy_day=next_day, 
                              buy_price=buy_price, sell_price=0.0, count=count)
            dict_info[code] = trade
            total_free_amount -= trade.buy_amount()
            
            day_profit = round((next_data["close"] - buy_price) / buy_price * 100, 2)
            write_green(f"buy:{trade.buy_string()} (动态{buy_amount}) || 开{next_data['open']:.2f} 收{next_data['close']:.2f} 当日收益:{day_profit}%")
        
        # 3. 卖出逻辑：用明天的数据检查是否卖出
        sell_list = []
        for code in list(hold_set):
            pre_data = backtester._get_stock_data_fast(code, cur_day)
            next_data = backtester._get_stock_data_fast(code, next_day)
            
            if not pre_data or not next_data:
                write_white(f"股票{code} 日期{next_day}没有信息，跳过")
                continue
            
            trade_info = dict_info[code]
            sell_flag = backtester._apply_sell_rule(pre_data, next_data, trade_info, next_day, config)
            
            if sell_flag:
                stats['sell_reason_dict'][sell_flag] = stats['sell_reason_dict'].get(sell_flag, 0) + 1
                sell_list.append(code)
                total_free_amount += trade_info.sell_amount()
                stats['sell_count'] += 1
                
                if trade_info.sell_amount() > trade_info.buy_amount():
                    stats['win_count'] += 1
                    write_red("sell:" + trade_info.sell_string())
                else:
                    write_yellow(f"{trade_info.sell_string()} || 开{next_data['open']:.2f} 收{next_data['close']:.2f} 涨{next_data['change_pct']:.2f}%")
        
        for code in sell_list:
            hold_set.remove(code)
            dict_info.pop(code)
        
        # 4. 更新持仓
        total_hold_amount = 0
        for code in hold_set:
            next_data = backtester._get_stock_data_fast(code, next_day)
            if not next_data:
                write_white(f"股票{code} 日期{next_day}没有信息，跳过")
                continue
            
            trade_info = dict_info[code]
            trade_info.close = next_data["close"]
            total_hold_amount += trade_info.hold_amount()
            trade_info.hold_day += 1
            
            write_yellow(f"hold:{trade_info.hold_string()} || 开{next_data['open']:.2f} 收{next_data['close']:.2f} 涨{next_data['change_pct']:.2f}%")
        
        # 5. 资金统计
        current_total_amount = total_free_amount + total_hold_amount
        stats['max_total_amount'] = max(stats['max_total_amount'], current_total_amount)
        current_drawdown = (stats['max_total_amount'] - current_total_amount) / stats['max_total_amount'] * 100
        stats['max_drawdown'] = max(stats['max_drawdown'], current_drawdown)
        stats['max_profit_rate'] = max(stats['max_profit_rate'], 
                                     round(current_total_amount / 100000.00 * 100, 2))
        
        write_white(f"{next_day} 空闲金额:{round(total_free_amount, 2)} 持仓金额:{round(total_hold_amount, 2)} 总金额:{round((total_free_amount + total_hold_amount), 2)}")
        
        cur_day = next_day
    
    final_day = backtester.calendar.pre(cur_day) if cur_day > end else cur_day
    
    if final_day != start:
        write_white("=" * 80)
        write_white(f"【最后选股结果】{final_day}收盘后选出，可用于下一个交易日")
        
        final_selected, final_qualified = backtester._fast_pick_stock(final_day, config)
        final_selected = final_selected[:config.max_hold_count]
        
        write_white(f"pick: 候选{final_qualified}只 | 选中{len(final_selected)}只")
        
        for code in final_selected:
            cur_data = backtester._get_stock_data_fast(code, final_day)
            if cur_data:
                write_blue(f"pick: //{len(final_selected)} {code} {final_day} 开{cur_data['open']:.2f} 收{cur_data['close']:.2f} 涨{cur_data['change_pct']:.2f}% 连涨{cur_data.get('consecutive_up_days', 0)}天 量比{cur_data.get('vol_rank', 0):.1f} 3日{cur_data.get('change_3d', 0):.2f} 5日{cur_data.get('change_5d', 0):.2f}")
        
        next_trading_day = backtester.calendar.next(final_day)
        write_white(f"下个交易日: {next_trading_day}")
        write_white("=" * 80)
    
    if cur_day == end or backtester.calendar.pre(cur_day) == end:
        write_white(f"回测结束，最终日期: {cur_day}")
    elif cur_day > end:
        write_white(f"回测结束，最终日期: {backtester.calendar.pre(cur_day)}")
    
    total_hold_amount = sum(trade.hold_amount() for trade in dict_info.values())
    total_amount = total_free_amount + total_hold_amount
    total_return_rate = round((total_amount - 100000.00) / 100000.00 * 100, 2)
    win_rate = round(stats['win_count'] / stats['sell_count'] * 100, 2) if stats['sell_count'] > 0 else 0
    
    write_white("=" * 80)
    write_white("=== 回测结果汇总 ===")
    write_white(f"策略名称: {config.name}")
    write_white(f"回测期间: {start} - {end}")
    write_white(f"初始资金: 100000.00")
    write_white(f"最终资产: {round(total_amount, 2)}")
    write_white(f"总收益率: {total_return_rate}%")
    write_white(f"最高收益率: {stats['max_profit_rate']}%")
    write_white(f"最大回撤: {stats['max_drawdown']:.2f}%")
    write_white(f"胜率: {win_rate}%")
    write_white(f"总交易次数: {stats['sell_count']}")
    write_white(f"盈利次数: {stats['win_count']}")
    write_white(f"亏损次数: {stats['sell_count'] - stats['win_count']}")
    write_white(f"卖出原因统计:")
    for reason, count in sorted(stats['sell_reason_dict'].items(), key=lambda x: x[1], reverse=True):
        write_white(f"  - {reason}: {count}次")
    write_white("=" * 80)
    
    if log_file:
        close_detailed_log(log_file)
    
    return {
        "strategy_name": config.name,
        "total_return_rate": total_return_rate,
        "max_profit_rate": stats['max_profit_rate'],
        "max_drawdown": round(stats['max_drawdown'], 2),
        "win_rate": win_rate,
        "total_trades": stats['sell_count'],
        "sell_reasons": stats['sell_reason_dict'],
        "final_amount": total_amount,
        "start_date": start,
        "end_date": end
    }


def generate_strategies(total_strategies: int = None) -> List[StrategyConfig]:
    """生成策略参数组合 - 与 backtest_df_optimized.py 完全对齐"""
    import itertools
    
    strategies = []
    
    # 与 backtest_df_optimized.py 完全相同的参数范围
    param_ranges = {
        # # 买入条件参数
        # 'consecutive_up_days': [2, 3],
        # 'change_3d': [3.0, 5.0],
        # 'change_5d': [10.0,15.0],
        # 'daily_change_max': [ 5, 8],
        
        # # 卖出条件参数
        # 'max_hold_days': [2, 3, 4],
        # 'min_profit_rate': [15, 18],
        # 'stop_loss_rate': [ 0.05, 0.08],
        # 'prev_day_drop_threshold': [-5.0, -8.0],
        
        # # 资金管理参数
        # 'max_hold_count': [2,3]
        # 买入条件参数
        'consecutive_up_days': list(range(2, 4,1)),
        'change_3d': list(range(2, 8,2)),
        'change_5d': list(range(5, 25, 2)),
        'daily_change_max': list(range(5, 9,1)),
        
        # 卖出条件参数
        'max_hold_days': list(range(2, 8,1)),
        'min_profit_rate': list(range(15, 26,5)),
        'stop_loss_rate': [x/100 for x in list(range(5, 11,2))],
        'prev_day_drop_threshold': list(range(-3, -8,-1)),
        
        # 资金管理参数
        'max_hold_count': list(range(1, 6,1))
    }
    
    param_combinations = list(itertools.product(
        param_ranges['consecutive_up_days'],
        param_ranges['change_3d'],
        param_ranges['change_5d'],
        param_ranges['daily_change_max'],
        param_ranges['max_hold_days'],
        param_ranges['stop_loss_rate'],
        param_ranges['prev_day_drop_threshold'],
        param_ranges['min_profit_rate'],
        param_ranges['max_hold_count']
    ))
    
    total_possible = len(param_combinations)
    print(f"可生成策略总数: {total_possible}")
    
    if total_strategies is None:
        total_strategies = total_possible
    else:
        total_strategies = min(total_strategies, total_possible)
    
    for i, params in enumerate(param_combinations):
        if i >= total_strategies:
            break
        
        config = StrategyConfig(
            name=f"L{params[0]}-D{params[1]}-W{params[2]}-Z{params[3]}-C{params[4]}-S{int(params[5]*100)}-Y{abs(int(params[6]))}-E{params[7]}-H{params[8]}",
            consecutive_up_days_min=params[0],
            change_3d_min=params[1],
            change_5d_min=params[2],
            daily_change_max=params[3],
            max_hold_days=params[4],
            stop_loss_rate=params[5],
            prev_day_drop_threshold=params[6],
            min_profit_rate=params[7],
            max_hold_count=params[8]
        )
        strategies.append(config)
    
    print(f"策略生成完成，总计 {len(strategies)} 个策略")
    return strategies


def process_strategies_batch(args: tuple) -> List[Dict[str, Any]]:
    """处理批量策略 - 每个进程处理一批策略"""
    config_dicts, start, end, process_id = args
    
    # 在每个进程中重新创建回测器（但使用缓存避免重复计算）
    backtester = HighPerformanceBacktester(use_cache=True)
    
    results = []
    
    # 为每个进程创建独立的进度条
    for i, config_dict in enumerate(tqdm(config_dicts, desc=f"进程 {process_id}", position=process_id)):
        try:
            # 从字典重建配置对象（包含所有9个参数）
            config = StrategyConfig(
                name=config_dict['name'],
                consecutive_up_days_min=config_dict['consecutive_up_days_min'],
                change_3d_min=config_dict['change_3d_min'],
                change_5d_min=config_dict['change_5d_min'],
                daily_change_max=config_dict['daily_change_max'],
                max_hold_days=config_dict['max_hold_days'],
                stop_loss_rate=config_dict['stop_loss_rate'],
                prev_day_drop_threshold=config_dict['prev_day_drop_threshold'],
                min_profit_rate=config_dict['min_profit_rate'],
                max_hold_count=config_dict['max_hold_count']
            )
            
            result = backtester.fast_backtest(start, end, config)
            results.append(result)
        except Exception as e:
            results.append({
                "strategy_name": config_dict.get('name', '未知'),
                "error": str(e),
                "total_return_rate": -100.0
            })
    
    return results


def calculate_multi_period_average(period_results: list, strategy_name: str) -> dict:
    """计算多时间周期的平均结果"""
    
    return_rates = [r['total_return_rate'] for r in period_results]
    win_rates = [r['win_rate'] for r in period_results]
    drawdowns = [r['max_drawdown'] for r in period_results]
    trades = [r['total_trades'] for r in period_results]
    max_profit_rates = [r['max_profit_rate'] for r in period_results]
    
    avg_return = round(sum(return_rates) / len(return_rates), 2)
    avg_win_rate = round(sum(win_rates) / len(win_rates), 2)
    avg_drawdown = round(sum(drawdowns) / len(drawdowns), 2)
    avg_trades = round(sum(trades) / len(trades), 2)
    avg_max_profit = round(sum(max_profit_rates) / len(max_profit_rates), 2)
    
    return_std = round(np.std(return_rates), 2) if len(return_rates) > 1 else 0
    stability_score = round(100 - return_std, 2)
    
    positive_periods = sum(1 for r in return_rates if r > 0)
    win_period_ratio = round(positive_periods / len(return_rates) * 100, 2)
    
    simplified_details = []
    for r in period_results:
        sell_reasons_str = " | ".join([f"{k.split('(')[0]}:{v}" for k, v in r.get('sell_reasons', {}).items()])
        simplified_details.append({
            "周期": r['period'],
            "收益率": f"{r['total_return_rate']:.2f}%",
            "最大收益": f"{r['max_profit_rate']:.2f}%",
            "回撤": f"{r['max_drawdown']:.2f}%",
            "胜率": f"{r['win_rate']:.2f}%",
            "交易数": r['total_trades'],
            "卖出原因": sell_reasons_str if sell_reasons_str else "无"
        })
    
    result = {
        'strategy_name': strategy_name,
        'total_return_rate_avg': avg_return,
        'win_rate_avg': avg_win_rate,
        'max_drawdown_avg': avg_drawdown,
        'max_profit_avg': avg_max_profit,
        'total_trades_avg': avg_trades,
        'stability_score': stability_score,
        'win_period_ratio': win_period_ratio,
        'return_std': return_std,
        'test_periods': len(period_results),
        'period_details': simplified_details
    }
    
    return result


def optimized_single_process_backtest(total_strategies: int = None, 
                                       start_date: str = "20250101",
                                       end_date: str = "20251231",
                                       multi_period: bool = True) -> pd.DataFrame:
    """
    优化单进程回测主函数 - 支持多周期回测
    
    Args:
        total_strategies: 总策略数量，None表示自动计算
        start_date: 回测开始日期
        end_date: 回测结束日期
        multi_period: 是否进行多时间周期测试
    """
    
    print("=== 高性能回测系统（单进程优化版） ===")
    
    # 生成所有策略
    all_strategies = generate_strategies(total_strategies)
    actual_total = len(all_strategies)
    
    if actual_total == 0:
        print("没有可测试的策略")
        return pd.DataFrame()
    
    print(f"开始测试 {actual_total} 个策略")
    
    # 定义多时间周期（每3个月一个周期）
    if multi_period:
        # 从2023年7月到2025年12月的季度周期
        periods = [
            ("20230701", "20230930"),  # 2023年Q3（7-9月）
            ("20231001", "20231231"),  # 2023年Q4（10-12月）
            ("20240101", "20240331"),  # 2024年Q1（1-3月）
            ("20240401", "20240630"),  # 2024年Q2（4-6月）
            ("20240701", "20240930"),  # 2024年Q3（7-9月）
            ("20241001", "20241231"),  # 2024年Q4（10-12月）
            ("20250101", "20250331"),  # 2025年Q1（1-3月）
            ("20250401", "20250630"),  # 2025年Q2（4-6月）
            ("20250701", "20250930"),  # 2025年Q3（7-9月）
            ("20251001", "20251231"),  # 2025年Q4（10-12月）
        ]
        print(f"测试 {len(periods)} 个时间段")
    else:
        periods = [(start_date, end_date)]
    
    # 初始化回测器
    backtester = HighPerformanceBacktester(use_cache=True)
    
    all_results = []
    start_time = time.time()
    
    # 分批处理，避免内存溢出
    batch_size = min(1000, actual_total)
    total_batches = (actual_total + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, actual_total)
        batch_strategies = all_strategies[start_idx:end_idx]
        
        print(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_strategies)} 个策略)")
        
        batch_start_time = time.time()
        
        for i, config in enumerate(tqdm(batch_strategies, desc=f"批次 {batch_idx + 1}")):
            try:
                period_results = []
                
                # 多周期测试
                for j, (period_start, period_end) in enumerate(periods):
                    result = backtester.fast_backtest(period_start, period_end, config)
                    result['period'] = f"{period_start}-{period_end}"
                    period_results.append(result)
                    
                if multi_period:
                    # 多周期测试：计算平均结果
                    avg_result = calculate_multi_period_average(period_results, config.name)
                    all_results.append(avg_result)
                else:
                    # 单周期测试：直接使用第一个周期的结果
                    all_results.append(period_results[0])
                    
            except Exception as e:
                print(f"策略 {config.name} 失败: {e}")
        
        batch_time = time.time() - batch_start_time
        print(f"批次 {batch_idx + 1} 完成，耗时: {batch_time:.2f}秒")
    
    total_time = time.time() - start_time
    print(f"回测完成! 总耗时: {total_time:.2f}秒")
    print(f"平均每个策略耗时: {total_time/actual_total:.4f}秒")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 按收益率排序
    if multi_period:
        df = df.sort_values('total_return_rate_avg', ascending=False)
    else:
        df = df.sort_values('total_return_rate', ascending=False)
    
    return df


def main():
    """主函数 - 运行全量网格策略回测并自动验证最佳策略"""
    
    print("开始全量网格策略回测...")
    
    # 初始化回测器获取真实数据范围
    backtester = HighPerformanceBacktester(use_cache=True)
    all_dates = backtester.all_dates
    
    # 默认开始日期（可调整）
    start_date = "20230701"
    
    # 默认结束日期（可调整）
    end_date = "20260130"
    
    # 自动检测最后一个有数据的日期作为结束日期
    last_data_date = all_dates[-1] if all_dates else end_date
    
    # 如果指定的结束日期晚于最后数据日期，使用最后数据日期
    actual_end_date = min(end_date, last_data_date)
    
    print(f"数据范围: {start_date} 到 {last_data_date}")
    print(f"回测结束日期: {actual_end_date} (自动调整)")
    print(f"交易日数: {len([d for d in all_dates if d <= actual_end_date])}")
    
    # 直接运行大规模回测（使用单进程多周期优化版）
    main_df = optimized_single_process_backtest(total_strategies=None, start_date=start_date, end_date=actual_end_date, multi_period=True)
    
    # 保存结果
    if not main_df.empty:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(f"{PROJECT_ROOT}/.temp/data/results", exist_ok=True)
        
        # 激进型策略 CSV（按平均收益率排序）
        aggressive_df = main_df.sort_values('total_return_rate_avg', ascending=False).head(100)
        aggressive_file = f"{PROJECT_ROOT}/.temp/data/results/aggressive_strategies_{timestamp}.csv"
        aggressive_df.to_csv(aggressive_file, index=False, encoding='utf-8-sig')
        print(f"激进型策略结果已保存至: {aggressive_file}")
        
        # 计算保守型评分
        main_df['conservative_score'] = (
            main_df['total_return_rate_avg'] * 0.30 + 
            main_df['win_period_ratio'] * 0.25 + 
            main_df['max_profit_avg'] * 0.15 - 
            main_df['max_drawdown_avg'] * 0.20 - 
            main_df['return_std'] * 0.10
        )
        
        # 保守型策略 CSV（按综合评分排序，只保留正收益策略）
        conservative_df = main_df[main_df['total_return_rate_avg'] > 0].sort_values('conservative_score', ascending=False).head(100)
        conservative_file = f"{PROJECT_ROOT}/.temp/data/results/conservative_strategies_{timestamp}.csv"
        conservative_df.to_csv(conservative_file, index=False, encoding='utf-8-sig')
        print(f"保守型策略结果已保存至: {conservative_file}")
        
        # 自动识别最佳策略并进行详细验证
        best_strategy_validation(main_df, start_date, actual_end_date, timestamp)
    else:
        print("回测失败，没有有效结果")


def best_strategy_validation(main_df: pd.DataFrame, start_date: str, end_date: str, timestamp: str):
    """最佳策略详细验证 - 输出激进型和保守型两种最优策略"""
    
    if main_df.empty:
        print("没有有效策略结果，跳过详细验证")
        return
    
    print("\n" + "=" * 80)
    print("🎯 策略分析结果")
    print("=" * 80)
    
    aggressive_strategy = main_df.iloc[0]
    print(f"\n🔥 激进型策略 Top 1 (按平均收益率排序):")
    print(f"   策略: {aggressive_strategy['strategy_name']}")
    print(f"   平均收益率: {aggressive_strategy['total_return_rate_avg']}%")
    print(f"   平均胜率: {aggressive_strategy['win_rate_avg']}%")
    print(f"   平均回撤: {aggressive_strategy['max_drawdown_avg']}%")
    print(f"   周期胜率: {aggressive_strategy['win_period_ratio']}%")
    
    conservative_df = main_df[main_df['total_return_rate_avg'] > 0].sort_values('conservative_score', ascending=False)
    conservative_strategy = conservative_df.iloc[0] if not conservative_df.empty else aggressive_strategy
    
    print(f"\n🛡️ 保守型策略 Top 1 (综合评分: 收益30% + 周期胜率25% + 最大收益15% - 回撤20% - 波动10%):")
    print(f"   策略: {conservative_strategy['strategy_name']}")
    print(f"   平均收益率: {conservative_strategy['total_return_rate_avg']}%")
    print(f"   平均胜率: {conservative_strategy['win_rate_avg']}%")
    print(f"   平均回撤: {conservative_strategy['max_drawdown_avg']}%")
    print(f"   周期胜率: {conservative_strategy['win_period_ratio']}%")
    print(f"   综合评分: {conservative_strategy['conservative_score']:.2f}")
    
    print("\n" + "=" * 80)
    print("🔍 开始详细回测验证...")
    print("=" * 80)
    
    for strategy_type, strategy_row in [("激进型", aggressive_strategy), ("保守型", conservative_strategy)]:
        print(f"\n📊 {strategy_type}策略详细回测: {strategy_row['strategy_name']}")
        
        strategy_params = parse_strategy_params(strategy_row['strategy_name'])
        
        config = StrategyConfig(
            name=f"{strategy_type}策略验证_{strategy_row['strategy_name']}",
            consecutive_up_days_min=strategy_params['consecutive_up_days'],
            change_3d_min=strategy_params['change_3d'],
            change_5d_min=strategy_params['change_5d'],
            daily_change_max=strategy_params['daily_change_max'],
            max_hold_days=strategy_params['max_hold_days'],
            stop_loss_rate=strategy_params['stop_loss_rate'],
            prev_day_drop_threshold=strategy_params['prev_day_drop_threshold'],
            min_profit_rate=strategy_params['min_profit_rate'],
            max_hold_count=strategy_params['max_hold_count']
        )
        
        detailed_log_path = f"{PROJECT_ROOT}/.temp/data/log/{strategy_type.lower()}_strategy_detailed_log_{timestamp}.log"
        
        detailed_result = detailed_backtest_with_logging(
            start=start_date, 
            end=end_date, 
            config=config, 
            log_file_path=detailed_log_path
        )
        
        print(f"\n✅ {strategy_type}策略详细验证完成!")
        print(f"📝 详细日志: {detailed_log_path}")
        print(f"📊 验证结果: 收益率{detailed_result['total_return_rate']}% | "
              f"胜率{detailed_result['win_rate']}% | "
              f"最大回撤{detailed_result['max_drawdown']}%")
    
    print("\n" + "=" * 80)


def parse_strategy_params(strategy_name: str) -> Dict[str, float]:
    """从策略名称中解析参数"""
    parts = strategy_name.split('-')
    
    return {
        'consecutive_up_days': int(parts[0][1:]),
        'change_3d': float(parts[1][1:]),
        'change_5d': float(parts[2][1:]),
        'daily_change_max': float(parts[3][1:]),
        'max_hold_days': int(parts[4][1:]),
        'stop_loss_rate': float(parts[5][1:]) / 100,
        'prev_day_drop_threshold': -float(parts[6][1:]),
        'min_profit_rate': float(parts[7][1:]),
        'max_hold_count': int(parts[8][1:])
    }


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    
    main()
    
    total_time = datetime.datetime.now() - start_time
    print(f"总耗时: {total_time}")