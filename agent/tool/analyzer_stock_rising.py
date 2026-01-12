#分析连涨的股票，最低从3天起,目标生成一个表格，包含股票代码、股票名称、连涨天数、累计涨幅，最近的3天每天的涨幅表头用日期
#股票列表数据在.temp/data/base/stock_list.csv
#股票历史信息在.temp/data/stock_data/


import pandas as pd
from pathlib import Path
import sys
# 将项目根目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.file_util import FileUtil
from utils.log_util import LogUtil

logger = LogUtil.get_logger(__name__)

# ==================== 目录结构定义 ====================
BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
ANALYZER_DIR = DATA_DIR / "analyzer"
FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ANALYZER_DIR)

STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.csv"
DAILY_DATA_DIR = DATA_DIR / "stock_data"



def _analyzer() -> pd.DataFrame:
    # 读取股票列表
    stock_list_df = pd.read_csv(STOCK_LIST_FILE, encoding="utf-8-sig", dtype={"代码": str})
    
    # 过滤掉科创板和创业板股票
    stock_list_df = stock_list_df[~stock_list_df["板块类型"].isin(["科创板", "创业板", "北京股"])]
    
    # 结果列表
    result_data = []
    
    # 遍历每个股票
    for _, stock in stock_list_df.iterrows():
        stock_code = stock["代码"]
        stock_name = stock["名称"]
        plate_type = stock["板块类型"]
        
        # 读取股票历史数据
        stock_file = DAILY_DATA_DIR / f"{stock_code}.csv"
        if not stock_file.exists():
            logger.warning(f"股票数据文件不存在: {stock_file}")
            continue
        
        try:
            # 读取股票历史数据，确保日期列是日期类型
            stock_df = pd.read_csv(stock_file, encoding="utf-8-sig", parse_dates=["日期"])
            
            # 按日期排序（从旧到新）
            stock_df = stock_df.sort_values(by="日期", ascending=True)
            
            # 计算每天的涨幅
            stock_df["涨幅"] = stock_df["收盘"].pct_change() * 100
            
            # 找出连续上涨的天数
            consecutive_rising_days = 0
            cumulative_gain = 0
            recent_3_days = []
            recent_3_dates = []
            
            # 从最新日期开始往前找连续上涨的天数
            # 先按日期排序（从新到旧）
            stock_df_reverse = stock_df.sort_values(by="日期", ascending=False)
            
            for _, row in stock_df_reverse.iterrows():
                if row["涨幅"] > 0:
                    consecutive_rising_days += 1
                    cumulative_gain += row["涨幅"]
                    # 记录最近3天的涨幅和日期
                    if len(recent_3_days) < 3:
                        recent_3_days.append(row["涨幅"])
                        recent_3_dates.append(row["日期"].strftime("%Y-%m-%d"))
                else:
                    break
            
            # 只处理连涨天数≥3天的股票
            if consecutive_rising_days >= 3:
                # 确保有3天的数据
                while len(recent_3_days) < 3:
                    recent_3_days.append(0)
                    recent_3_dates.append("")
                
                # 只保留最近3天的数据
                recent_3_days = recent_3_days[:3]
                recent_3_dates = recent_3_dates[:3]
                
                # 计算最近3日涨幅
                recent_3_gain = sum(recent_3_days)
                
                # 添加到结果列表
                result_row = {
                    "股票代码": stock_code,
                    "股票名称": stock_name,
                    "板块类型": plate_type,
                    "连涨天数": consecutive_rising_days,
                    "累计涨幅": cumulative_gain,
                    "最近3日涨幅": recent_3_gain
                }
                
                # 添加最近3天的日期和涨幅
                for i in range(3):
                    if recent_3_dates[i]:
                        result_row[recent_3_dates[i]] = recent_3_days[i]
                
                result_data.append(result_row)
                
        except Exception as e:
            logger.error(f"分析股票 {stock_code} 时出错: {e}")
            continue
    
    # 转换为DataFrame
    if result_data:
        result_df = pd.DataFrame(result_data)
        
        # 添加过滤条件：最近3天累计涨幅>=10
        result_df = result_df[result_df["最近3日涨幅"] >= 10]
        
        # 如果过滤后没有数据，返回空DataFrame
        if result_df.empty:
            return pd.DataFrame()
        
        # 分离固定列和日期列
        fixed_columns = ["股票代码", "股票名称", "板块类型", "连涨天数", "累计涨幅", "最近3日涨幅"]
        date_columns = [col for col in result_df.columns if col not in fixed_columns]
        
        # 如果有日期列，按日期排序并只保留最近3个
        if date_columns:
            # 转换为日期对象以便排序
            date_columns.sort(key=lambda x: pd.to_datetime(x, errors='coerce'), reverse=True)
            # 只保留最近3个日期列
            selected_date_columns = date_columns[:3]
            
            # 重新组合列
            result_df = result_df[fixed_columns + selected_date_columns]
        
        # 保留小数点1位
        numeric_columns = ["累计涨幅", "最近3日涨幅"] + date_columns
        for col in numeric_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].round(1)
        
        # 按连涨天数降序排序
        result_df = result_df.sort_values(by="连涨天数", ascending=False)
        return result_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    rs=_analyzer()
    rs.to_csv(ANALYZER_DIR / "stock_rising.csv", index=False, encoding="utf-8-sig")
    print(rs)