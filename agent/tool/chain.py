# 串联数据更新、分析和发送邮件的完整流程

import sys
from pathlib import Path
from datetime import datetime

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入所需的函数
from agent.tool.data_updater import _get_trading_days, _update_etf_list, _update_daily_etf_data, _update_code_list, _update_daily_stock_data
from agent.tool.analyzer_etf_rising import _analyzer as etf_analyzer
from agent.tool.analyzer_stock_rising import _analyzer as stock_analyzer
from agent.tool.send_msg import send_analyzer_table
from utils.log_util import LogUtil, print_green, print_red

logger = LogUtil.get_logger(__name__)

def is_today_trading_day() -> bool:
    """判断今天是否是交易日"""
    today = datetime.today().strftime("%Y%m%d")
    trading_days = _get_trading_days(5)  # 获取最近5个交易日
    return today in trading_days

def run_full_chain():
    """运行完整的任务链"""
    try:
        print_green("=== 开始运行完整任务链 ===")
        
        # 判断今天是否是交易日
        print_green("1. 判断今天是否是交易日...")
        if not is_today_trading_day():
            print_green("今天不是交易日，无需执行后续任务")
            return
        
        print_green("今天是交易日，开始执行后续任务")
        
        # 1. 数据更新
        print_green("\n2. 开始更新数据...")
        _update_etf_list()              # 更新ETF列表
        _update_daily_etf_data()        # 更新ETF日线数据
        _update_code_list()             # 更新股票代码列表
        _update_daily_stock_data()      # 更新股票日线数据
        
        # 2. ETF连涨分析
        print_green("\n3. 开始分析ETF连涨情况...")
        etf_result = etf_analyzer()
        if etf_result.empty:
            print_green("未找到符合条件的ETF")
        else:
            print_green(f"ETF连涨分析完成，共找到 {len(etf_result)} 个符合条件的ETF")
        
        # 3. 股票连涨分析
        print_green("\n4. 开始分析股票连涨情况...")
        stock_result = stock_analyzer()
        if stock_result.empty:
            print_green("未找到符合条件的股票")
        else:
            print_green(f"股票连涨分析完成，共找到 {len(stock_result)} 个符合条件的股票")
        
        # 4. 发送邮件
        print_green("\n5. 开始发送分析结果邮件...")
        send_analyzer_table()
        
        print_green("\n=== 完整任务链运行完成 ===")
        
    except Exception as e:
        print_red(f"任务链运行失败: {e}")
        logger.error(f"任务链运行失败: {e}", exc_info=True)

if __name__ == "__main__":
    run_full_chain()