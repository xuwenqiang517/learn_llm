# 串联数据更新、分析和发送邮件的完整流程

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入所需的函数
from agent.tool.load_base_data import _get_trading_days, _update_all_data
from agent.tool.send_msg import send_analyzer_table
from agent.remote_model_agent import main as remote_model_agent_main
from utils.log_util import LogUtil, print_green, print_red
from agent.tool.pick_data import _run_pick

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
        
        print_green("2. 检查并更新akshare版本...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'akshare'], 
                          check=True, capture_output=True, text=True)
            print_green("akshare版本更新成功！")
        except Exception as e:
            print_red(f"akshare版本更新失败: {e}")
            print_green("继续使用当前版本...")
        
        # 1.更新基础数据
        _update_all_data()
        _run_pick()
        # # 2. ETF连涨分析
        # print_green("\n3. 开始分析ETF连涨情况...")
        # etf_result = etf_analyzer()
        # if etf_result.empty:
        #     print_green("未找到符合条件的ETF")
        # else:
        #     print_green(f"ETF连涨分析完成，共找到 {len(etf_result)} 个符合条件的ETF")
        
        # # 3. 股票连涨分析
        # print_green("\n4. 开始分析股票连涨情况...")
        # stock_result = stock_analyzer()
        # if stock_result.empty:
        #     print_green("未找到符合条件的股票")
        # else:
        #     print_green(f"股票连涨分析完成，共找到 {len(stock_result)} 个符合条件的股票")
        
        # 4. 发送邮件
        print_green("\n5. 开始发送分析结果邮件...")
        send_analyzer_table()

        # 5. 调用远程模型分析
        print_green("\n6. 开始调用远程模型分析...")
        remote_model_agent_main()
        
        
    except Exception as e:
        logger.error(f"任务链运行失败: {e}", exc_info=True)

if __name__ == "__main__":
    run_full_chain()
