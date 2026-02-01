import datetime
from typing import Dict, Any, Callable, Optional
import pandas as pd
import numpy as np
from TradingCalendar import TradingCalendar
from BuySellInfo import BuySellInfo
import os
import time
from data import get_full_data
from tqdm import tqdm

log_file = None
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def init_log_file(log_path: str = None):
    global log_file
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")


def close_log_file():
    global log_file
    if log_file:
        log_file.close()
        log_file = None


def write_log(msg: str):
    global log_file
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def print_green(msg, enable=True):
    if enable:
        print(f"\033[92m{msg}\033[0m")
        write_log(msg)

def print_red(msg, enable=True):
    if enable:
        print(f"\033[91m{msg}\033[0m")
        write_log(msg)

def print_yellow(msg, enable=True):
    if enable:
        print(f"\033[93m{msg}\033[0m")
        write_log(msg)

def print_white(msg, enable=True):
    if enable:
        print(f"\033[97m{msg}\033[0m")
        write_log(msg)

def print_blue(msg, enable=True):
    if enable:
        print(f"\033[94m{msg}\033[0m")
        write_log(msg)

calendar = TradingCalendar()


class StrategyConfig:
    """策略配置类，用于参数化买入卖出条件"""
    
    def __init__(self, 
                 name: str = "默认策略",
                 # 买入条件参数
                 consecutive_up_days_min: int = 3,
                 change_3d_min: float = 5.0,
                 change_5d_min: float = 10.0,
                 daily_change_max: float = 5.0,
                 
                 # 卖出条件参数
                 stop_loss_rate: float = 0.05,
                 prev_day_drop_threshold: float = -5.0,
                 max_hold_days: int = 6,
                 min_profit_rate: float = 20.0,
                 
                 # 资金管理参数
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


def create_pick_rule(config: StrategyConfig) -> Callable[[pd.Series], bool]:
    """创建选股规则函数"""
    def pick_rule(row: pd.Series) -> bool:
        if (row["consecutive_up_days"] >= config.consecutive_up_days_min and 
            row["change_3d"] > config.change_3d_min and 
            row["change_5d"] > config.change_5d_min and 
            row["涨跌幅"] < config.daily_change_max):
            return True
        return False
    
    # 添加配置参数到函数属性，用于向量化优化
    pick_rule.consecutive_up_days_min = config.consecutive_up_days_min
    pick_rule.change_3d_min = config.change_3d_min
    pick_rule.change_5d_min = config.change_5d_min
    pick_rule.daily_change_max = config.daily_change_max
    
    return pick_rule


def create_sell_rule(config: StrategyConfig) -> Callable[[Dict, Dict, BuySellInfo, str], str]:
    """创建卖出规则函数"""
    def sell_rule(pre_data: Dict, cur_data: Dict, trade_info: BuySellInfo, cur_day: str) -> str:
        sell_flag = ""
        sell_price = cur_data["close"]
        stop_price = trade_info.stop_price(config.stop_loss_rate)
        current_rate = trade_info.profit_rate(sell_price)
        
        if cur_data["open"] < stop_price:
            trade_info.sell_reason = f"[1]开盘{cur_data['open']}低于止损价{stop_price}，直接卖出"
            sell_flag = "止损价"
            sell_price = cur_data["open"]
        elif pre_data["change_pct"] < config.prev_day_drop_threshold:
            trade_info.sell_reason = f"[3]昨日跌{pre_data['change_pct']}%,直接卖出"
            sell_flag = "昨日跌5%"
            sell_price = cur_data["open"]
        elif trade_info.hold_day > config.max_hold_days:
            if current_rate < config.min_profit_rate:
                trade_info.sell_reason = f"[6]持仓{trade_info.hold_day}天,收益率:{current_rate}%,<{config.min_profit_rate}%"
                sell_flag = f"持有{config.max_hold_days}天收益<{config.min_profit_rate}%"
        
        if len(sell_flag) > 0:
            trade_info.sell_price = sell_price
            trade_info.sell_day = cur_day
        
        return sell_flag
    return sell_rule


def pick_stock(date_dict: dict, day: str, pick_rule: Callable[[dict], bool]) -> pd.DataFrame:
    """
    选股函数（O(1)时间复杂度优化版）
    
    Args:
        date_dict: 两层字典结构 {日期: {股票代码: 股票数据}}
        day: 选股日期
        pick_rule: 选股规则函数
        
    Returns:
        符合条件的股票DataFrame
    """
    # O(1)时间复杂度直接访问
    if day not in date_dict:
        return pd.DataFrame()
    
    day_stocks = date_dict[day]
    
    # 筛选符合条件的股票
    selected_stocks = []
    for stock_code, stock_data in day_stocks.items():
        # 使用字典数据直接判断条件
        if (stock_data.get("consecutive_up_days", 0) >= getattr(pick_rule, 'consecutive_up_days_min', 3) and
            stock_data.get("change_3d", 0) > getattr(pick_rule, 'change_3d_min', 5.0) and
            stock_data.get("change_5d", 0) > getattr(pick_rule, 'change_5d_min', 10.0) and
            stock_data.get("涨跌幅", 0) < getattr(pick_rule, 'daily_change_max', 5.0)):
            
            # 添加股票代码和日期信息
            stock_data_with_id = stock_data.copy()
            stock_data_with_id["股票代码"] = stock_code
            stock_data_with_id["日期"] = day
            selected_stocks.append(stock_data_with_id)
    
    if not selected_stocks:
        return pd.DataFrame()
    
    # 转换为DataFrame并排序
    result = pd.DataFrame(selected_stocks)
    if "vol_rank" in result.columns:
        result = result.sort_values("vol_rank", ascending=False)
    
    return result


# 性能优化：已移除未使用的缓存和预加载函数，使用O(1)时间复杂度的字典结构


def get_stock_day_from_dict(date_dict: dict, code: str, day: str) -> Optional[Dict]:
    """从字典结构中获取股票日数据（O(1)时间复杂度）"""
    if day not in date_dict:
        return None
    
    day_stocks = date_dict[day]
    if code not in day_stocks:
        return None
    
    stock_data = day_stocks[code]
    return {
        "open": stock_data.get("开盘", 0),
        "close": stock_data.get("收盘", 0),
        "high": stock_data.get("最高", 0),
        "low": stock_data.get("最低", 0),
        "volume": stock_data.get("成交量", 0),
        "change_pct": stock_data.get("涨跌幅", 0)
    }


def back_test_strategy(start: str = "20260101", end: str = "20260119", 
                      date_dict: dict = None, 
                      base_total_amount: float = 100000.00,
                      config: StrategyConfig = None,
                      print_detail: bool = False, print_fail_reason: bool = False) -> Dict[str, Any]:
    """
    策略化回测函数（O(1)时间复杂度优化版）
    
    Args:
        start: 开始日期
        end: 结束日期
        date_dict: 两层字典结构 {日期: {股票代码: 股票数据}}
        base_total_amount: 初始资金
        config: 策略配置
        print_detail: 是否打印详细日志
        print_fail_reason: 是否打印失败原因
        
    Returns:
        回测结果字典
    """
    import time
    
    if config is None:
        config = StrategyConfig()
    
    if date_dict is None:
        raise ValueError("股票数据为空")
    
    pick_rule_func = create_pick_rule(config)
    sell_rule_func = create_sell_rule(config)
    
    pre_day = calendar.pre(start)
    cur_day = calendar.next(pre_day)
    hold_set = set()
    sell_list = []
    buy_list = []
    dict_info = {}
    total_free_amount = base_total_amount
    max_profit_rate = float("-inf")
    max_drawdown = 0.0
    max_total_amount = base_total_amount
    win_count = 0
    sell_count = 0
    sell_reason_dict = {}
    
    # 性能优化：已移除性能统计代码，保持简洁
    
    # 进度跟踪
    day_count = 0
    
    while cur_day < end:
        day_count += 1
        
        # 进度显示（每10天或最后一天显示）
        if print_detail and (day_count % 10 == 0 or cur_day == calendar.pre(end)):
            print_white(f"进度: 第{day_count}天 - 当前日期: {cur_day}", True)
        else:
            print_white("=" * 80 + f"今天:{cur_day}" + "=" * 80, print_detail)
        
        # 选股
        stock_df = pick_stock(date_dict, pre_day, pick_rule_func)
        stock_df = stock_df.head(config.max_hold_count)
        stock_len = len(stock_df)
        
        # 只在详细模式下输出选股信息
        if print_detail:
            for idx, row in stock_df.iterrows():
                code = idx[0]
                day = idx[1]
                print_blue(f"pick: //{stock_len} {code} {day} 开{row['开盘']} 收{row['收盘']} 涨{row['涨跌幅']}% 连涨{row['consecutive_up_days']}天 3日{row['change_3d']} 5日{row['change_5d']} 量排名{row['vol_rank']}", print_detail)
        
        buy_list = []
        for idx, row in stock_df.iterrows():
            code = row['股票代码']  # 直接从行数据获取股票代码
            if len(hold_set) < config.max_hold_count and code not in hold_set:
                # 直接使用字典数据（O(1)时间复杂度）
                cur_data = get_stock_day_from_dict(date_dict, code, cur_day)
                
                if cur_data is None:
                    print_white(f"股票{code} 日期{cur_day},没有信息，跳过", print_detail)
                    continue
                
                count = int(config.buy_amount_per_stock / cur_data["open"]) // 100 * 100
                if count == 0:
                    print_white(f"股票{code} 股价{cur_data['open']}过高，无法买入，跳过", print_detail)
                    continue
                if total_free_amount < count * cur_data["open"]:
                    print_white(f"股票{code} 资金不足，无法买入，跳过", print_detail)
                    continue
                
                hold_set.add(code)
                buy_list.append(code)
                buy_price = round(cur_data["open"], 2)
                trade = BuySellInfo(code=code, name="", buy_day=cur_day, 
                                  buy_price=buy_price, 
                                  sell_price=0.0, count=count)
                dict_info[code] = trade
                day_profit = round((cur_data["close"] - buy_price) / buy_price * 100, 2)
                print_green(f"buy:{trade.buy_string()} || 开{cur_data['open']} 收{cur_data['close']} 当日收益:{day_profit}%", print_detail)
                total_free_amount -= trade.buy_amount()
        
        sell_list = []
        for code in hold_set:
            # 直接使用字典数据（O(1)时间复杂度）
            pre_data = get_stock_day_from_dict(date_dict, code, pre_day)
            cur_data = get_stock_day_from_dict(date_dict, code, cur_day)
            
            if pre_data is None or cur_data is None:
                print_white(f"股票{code} 日期{cur_day},没有信息，跳过", print_detail)
                continue
            
            trade_info = dict_info[code]
            sell_flag = sell_rule_func(pre_data, cur_data, trade_info, cur_day)
            
            if len(sell_flag) > 0:
                sell_reason_dict[sell_flag] = sell_reason_dict.get(sell_flag, 0) + 1
                sell_list.append(code)
                total_free_amount += trade_info.sell_amount()
                sell_count += 1
                
                if trade_info.sell_amount() > trade_info.buy_amount():
                    win_count += 1
                else:
                    print_yellow(f"{trade_info.sell_string()} || 开{cur_data['open']} 收{cur_data['close']} 涨{cur_data['change_pct']}%", print_fail_reason)
        
        for code in sell_list:
            hold_set.remove(code)
            trade = dict_info[code]
            print_red("sell:" + trade.sell_string(), print_detail)
            dict_info.pop(code)
        
        total_hold_amount = 0
        for code in hold_set:
            # 直接使用字典数据（O(1)时间复杂度）
            cur_data = get_stock_day_from_dict(date_dict, code, cur_day)
            
            if cur_data is None:
                print_white(f"股票{code} 日期{cur_day},没有信息，跳过", print_detail)
                continue
            
            trade_info = dict_info[code]
            trade_info.close = cur_data["close"]
            total_hold_amount += trade_info.hold_amount()
            trade_info.hold_day += 1
            print_yellow(f"hold:{trade_info.hold_string()} || 开{cur_data['open']} 收{cur_data['close']} 涨{cur_data['change_pct']}%", print_detail)
        
        print_white(f"{cur_day} 空闲金额:{round(total_free_amount, 2)} 持仓金额:{round(total_hold_amount, 2)} 总金额:{round((total_free_amount + total_hold_amount), 2)}", print_detail)
        
        current_total_amount = total_free_amount + total_hold_amount
        max_total_amount = max(max_total_amount, current_total_amount)
        current_drawdown = (max_total_amount - current_total_amount) / max_total_amount * 100
        max_drawdown = max(max_drawdown, current_drawdown)
        
        max_profit_rate = max(max_profit_rate, round(current_total_amount / base_total_amount * 100, 2))
        
        pre_day = cur_day
        cur_day = calendar.next(cur_day)
    
    total_amount = total_free_amount + total_hold_amount
    total_return_rate = round((total_amount - base_total_amount) / base_total_amount * 100, 2)
    win_rate = round(win_count / sell_count * 100, 2) if sell_count > 0 else 0
    
    result = {
        "strategy_name": config.name,
        "total_return_rate": total_return_rate,
        "max_profit_rate": max_profit_rate,
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": win_rate,
        "total_trades": sell_count,
        "sell_reasons": sell_reason_dict,
        "final_amount": total_amount,
        "start_date": start,
        "end_date": end
    }
    
    rs_str = f"策略:{config.name} {start},{end} 资金{base_total_amount}->{total_amount} 总收益率:{total_return_rate}% 最高:{max_profit_rate}% 最大回撤:{max_drawdown:.2f}% 胜率:{win_rate}% 总交易次数:{sell_count} 卖出原因统计:{sorted(sell_reason_dict.items(), key=lambda x: x[0], reverse=True)}"
    # print_red(rs_str, True)
    
    return result


def compare_strategies(strategies: list[StrategyConfig], 
                      start: str = "20250101", 
                      end: str = "20251231",
                      base_total_amount: float = 100000.00,
                      multi_period: bool = False) -> pd.DataFrame:
    """
    多策略对比回测（优化版）
    
    Args:
        strategies: 策略配置列表
        start: 开始日期
        end: 结束日期
        base_total_amount: 初始资金
        multi_period: 是否进行多时间周期测试
        
    Returns:
        策略对比结果DataFrame
    """
    import time
    from data import get_full_data
    
    print("正在加载股票数据...")
    start_time = time.time()
    df = get_full_data()
    
    # 转换为两层字典结构：{日期: {股票代码: 股票数据}}
    print("正在优化数据结构...")
    date_dict = {}
    for idx, row in df.iterrows():
        stock_code, date_str = idx
        if date_str not in date_dict:
            date_dict[date_str] = {}
        date_dict[date_str][stock_code] = row.to_dict()
    
    data_load_time = time.time() - start_time
    print(f"数据加载和优化完成，耗时: {data_load_time:.2f}秒")
    print(f"数据日期范围: {min(date_dict.keys())} - {max(date_dict.keys())}")
    print(f"总交易日数: {len(date_dict)}")
    
    # 定义多时间周期
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
    else:
        periods = [(start, end)]
    
    print(f"将测试 {len(periods)} 个时间段")
    
    # 串行计算（简化版本）
    results = []
    total_strategies = len(strategies)
    strategy_start_time = time.time()
    
    # 创建异步日志文件（避免频繁打印影响性能）
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = f"{PROJECT_ROOT}/.temp/data/log/backtest_progress_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # 批量写入日志的缓冲区
    log_buffer = []
    buffer_size = 10  # 每10条日志写入一次文件
    
    def write_logs_to_file():
        """批量写入日志到文件"""
        if log_buffer:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                for log_entry in log_buffer:
                    f.write(log_entry + '\n')
            log_buffer.clear()
    
    for i, config in tqdm(enumerate(strategies), total=total_strategies, desc="测试策略进度"):
        strategy_progress = (i + 1) / total_strategies * 100
        
        # 异步写入策略进度
        log_buffer.append(f"[{i+1}/{total_strategies}] 正在测试策略: {config.name} ({strategy_progress:.1f}%)")
        if len(log_buffer) >= buffer_size:
            write_logs_to_file()
        
        period_results = []
        
        for j, (period_start, period_end) in enumerate(periods):
            try:
                period_progress = (j + 1) / len(periods) * 100
                
                period_start_time = time.time()
                
                # 直接使用内存数据
                result = back_test_strategy(
                    start=period_start, 
                    end=period_end, 
                    date_dict=date_dict,
                    base_total_amount=base_total_amount,
                    config=config,
                    print_detail=False,
                    print_fail_reason=False
                )
                period_time = time.time() - period_start_time
                
                result['period'] = f"{period_start}-{period_end}"
                period_results.append(result)
                
                # 异步写入周期结果
                log_buffer.append(f"    周期 {j+1}/{len(periods)}: {period_start}-{period_end} 完成，耗时: {period_time:.2f}秒")
                if len(log_buffer) >= buffer_size:
                    write_logs_to_file()
                
            except Exception as e:
                import traceback
                log_buffer.append(f"    周期 {j+1}/{len(periods)}: {period_start}-{period_end} 测试失败: {e}")
                if len(log_buffer) >= buffer_size:
                    write_logs_to_file()
                continue
        
        if period_results:
            if multi_period:
                # 多周期测试：计算平均结果
                avg_result = calculate_multi_period_average(period_results, config.name)
                results.append(avg_result)
            else:
                # 单周期测试：直接使用第一个周期的结果
                results.append(period_results[0])
    
    # 写入剩余的日志
    write_logs_to_file()
    
    total_time = time.time() - strategy_start_time
    print(f"\n所有策略测试完成，总耗时: {total_time:.2f}秒")
    print(f"平均每个策略耗时: {total_time/total_strategies:.2f}秒")
    print(f"详细进度日志已保存到: {log_file_path}")
    
    # 写入剩余的日志
    write_logs_to_file()
    
    # 转换为DataFrame便于分析
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        if multi_period:
            result_df = result_df.sort_values("total_return_rate_avg", ascending=False)
        else:
            result_df = result_df.sort_values("total_return_rate", ascending=False)
    
    print("\n" + "="*80)
    print("策略对比结果:")
    print("="*80)
    for idx, row in result_df.iterrows():
        if multi_period:
            print(f"{row['strategy_name']}: 平均收益率{row['total_return_rate_avg']}% | 平均胜率{row['win_rate_avg']}% | 平均最大回撤{row['max_drawdown_avg']}% | 稳定性{row['stability_score']}")
        else:
            print(f"{row['strategy_name']}: 收益率{row['total_return_rate']}% | 胜率{row['win_rate']}% | 最大回撤{row['max_drawdown']}% | 交易次数{row['total_trades']}")
    
    return result_df


def calculate_multi_period_average(period_results: list, strategy_name: str) -> dict:
    """计算多时间周期的平均结果"""
    
    # 提取各指标
    return_rates = [r['total_return_rate'] for r in period_results]
    win_rates = [r['win_rate'] for r in period_results]
    drawdowns = [r['max_drawdown'] for r in period_results]
    trades = [r['total_trades'] for r in period_results]
    
    # 计算平均值
    avg_return = round(sum(return_rates) / len(return_rates), 2)
    avg_win_rate = round(sum(win_rates) / len(win_rates), 2)
    avg_drawdown = round(sum(drawdowns) / len(drawdowns), 2)
    avg_trades = round(sum(trades) / len(trades), 2)
    
    # 计算稳定性指标（收益率标准差，越小越稳定）
    return_std = round(np.std(return_rates), 2) if len(return_rates) > 1 else 0
    stability_score = round(100 - return_std, 2)  # 稳定性分数（100-标准差）
    
    # 计算胜率（正收益周期数 / 总周期数）
    positive_periods = sum(1 for r in return_rates if r > 0)
    win_period_ratio = round(positive_periods / len(return_rates) * 100, 2)
    
    result = {
        'strategy_name': strategy_name,
        'total_return_rate_avg': avg_return,
        'win_rate_avg': avg_win_rate,
        'max_drawdown_avg': avg_drawdown,
        'total_trades_avg': avg_trades,
        'stability_score': stability_score,
        'win_period_ratio': win_period_ratio,
        'return_std': return_std,
        'test_periods': len(period_results),
        'period_details': period_results  # 保留详细周期结果
    }
    
    return result




def generate_strategy_combinations():
    """生成所有参数组合的策略"""
    import itertools
    
    # 买入条件参数
    consecutive_up_days_min_arr = [2, 3, 4]
    change_3d_min_arr = [3.0, 5.0, 7.0]
    change_5d_min_arr = [10.0, 15.0, 20.0]
    daily_change_max_arr = [3,5,8]
    
    # 卖出条件参数
    max_hold_days_arr = [2,3,4,5]
    min_profit_rate_arr = [15,18,20,25]
    stop_loss_rate_arr = [0.03, 0.05, 0.08, 0.10]
    prev_day_drop_threshold_arr = [-3.0, -5.0, -8.0]  # 新增昨日跌幅阈值
    
    # 资金管理参数
    max_hold_count_arr = [2,3,4,5]
    
    # 生成所有参数组合
    param_combinations = list(itertools.product(
        consecutive_up_days_min_arr,
        change_3d_min_arr,
        change_5d_min_arr,
        daily_change_max_arr,
        max_hold_days_arr,
        stop_loss_rate_arr,
        prev_day_drop_threshold_arr,  # 新增
        min_profit_rate_arr,
        max_hold_count_arr
    ))
    
    strategies = []
    for i, params in enumerate(param_combinations):
        strategy_name = f"策略{i+1}_连涨{params[0]}天_3日{params[1]}%_5日{params[2]}%_最大涨幅{params[3]}%_止损{int(params[5]*100)}%_昨跌{int(abs(params[6]))}%_持仓{params[4]}天_收益{params[7]}%"
        
        config = StrategyConfig(
            name=strategy_name,
            consecutive_up_days_min=params[0],
            change_3d_min=params[1],
            change_5d_min=params[2],
            daily_change_max=params[3],
            max_hold_days=params[4],
            stop_loss_rate=params[5],
            prev_day_drop_threshold=params[6],  # 新增
            min_profit_rate=params[7],
            max_hold_count=params[8]
        )
        strategies.append(config)
    
    print(f"生成了 {len(strategies)} 个策略组合")
    return strategies





def parameter_grid_search(start: str = "20250101", end: str = "20250228", 
                         max_strategies: int = 50, sample_rate: float = 0.1,
                         multi_period: bool = False):
    """
    参数网格搜索 - 自动测试所有参数组合
    
    Args:
        start: 开始日期
        end: 结束日期
        max_strategies: 最大策略数量（防止组合爆炸）
        sample_rate: 采样率（0-1），用于减少测试数量
        multi_period: 是否进行多时间周期测试
    """
    import random
    
    # 生成所有策略组合
    all_strategies = generate_strategy_combinations()
    
    # 采样策略（防止组合爆炸）
    if len(all_strategies) > max_strategies:
        if sample_rate < 1.0:
            sampled_count = min(max_strategies, int(len(all_strategies) * sample_rate))
            all_strategies = random.sample(all_strategies, sampled_count)
        else:
            all_strategies = all_strategies[:max_strategies]
    
    print(f"测试 {len(all_strategies)} 个策略组合...")
    
    # 运行策略对比
    result_df = compare_strategies(all_strategies, start=start, end=end, multi_period=multi_period)
    
    # 保存详细结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"{PROJECT_ROOT}/.temp/data/results/parameter_grid_results_{timestamp}.csv"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    result_df.to_csv(result_file, index=False, encoding='utf-8-sig')
    
    print(f"结果已保存到: {result_file}")
    
    # 显示最优策略（如果存在有效结果）
    if len(result_df) > 0:
        best_strategy = result_df.iloc[0]
        if multi_period:
            print(f"\n🎯 最优策略: {best_strategy['strategy_name']}")
            print(f"📈 平均收益率: {best_strategy['total_return_rate_avg']}%")
            print(f"✅ 平均胜率: {best_strategy['win_rate_avg']}%")
            print(f"📉 平均最大回撤: {best_strategy['max_drawdown_avg']}%")
            print(f"🔒 稳定性分数: {best_strategy['stability_score']}")
            print(f"🏆 正收益周期比例: {best_strategy['win_period_ratio']}%")
        else:
            print(f"\n🎯 最优策略: {best_strategy['strategy_name']}")
            print(f"📈 总收益率: {best_strategy['total_return_rate']}%")
            print(f"✅ 胜率: {best_strategy['win_rate']}%")
            print(f"📉 最大回撤: {best_strategy['max_drawdown']}%")
            print(f"🔢 总交易次数: {best_strategy['total_trade_count']}")
            print(f"📊 卖出原因统计: {best_strategy['sell_reason_stats']}")
    else:
        print("\n⚠️  没有有效的策略测试结果，所有策略测试失败")
        print("请检查数据加载和策略配置是否正确")
    
    return result_df


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    result_df = parameter_grid_search(sample_rate=1.0, max_strategies=200000, multi_period=True)
    print(f"总耗时: {datetime.datetime.now() - start_time}")