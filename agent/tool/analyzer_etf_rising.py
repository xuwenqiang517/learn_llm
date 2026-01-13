#分析连涨的ETF，最低从3天起,目标生成一个表格，包含ETF代码、ETF名称、连涨天数、累计涨幅，最近的3天每天的涨幅表头用日期
#ETF列表数据在.temp/data/base/etf_list.csv
#ETF历史信息在.temp/data/etf_data/


import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.file_util import FileUtil
from utils.log_util import LogUtil
from agent.tool.volume_price_analyzer import get_volume_price_summary

logger = LogUtil.get_logger(__name__)

# ==================== 目录结构定义 ====================
BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
ANALYZER_DIR = DATA_DIR / "analyzer"
FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ANALYZER_DIR)

ETF_LIST_FILE = BASE_DATA_DIR / "etf_list.csv"
ETF_DATA_DIR = DATA_DIR / "etf_data"


def _analyzer() -> pd.DataFrame:
    # 读取ETF列表
    etf_list_df = pd.read_csv(ETF_LIST_FILE, encoding="utf-8-sig", dtype={"代码": str})
    
    # 过滤掉总市值不到1亿的ETF
    if "总市值" in etf_list_df.columns:
        # 提取总市值的数字部分（去除"亿"单位）
        etf_list_df["总市值数值"] = pd.to_numeric(etf_list_df["总市值"].str.replace("亿", ""), errors="coerce")
        # 过滤总市值大于等于1亿的ETF
        etf_list_df = etf_list_df[etf_list_df["总市值数值"] >= 1.0]
        # 删除临时列
        etf_list_df = etf_list_df.drop(columns=["总市值数值"])
    
    # 结果列表
    result_data = []
    
    # 遍历每个ETF
    for _, etf in etf_list_df.iterrows():
        etf_code = etf["代码"]
        etf_name = etf["名称"]
        # 获取ETF的总市值、换手率和量比
        etf_total_market_value = etf.get("总市值", "")
        
        # 读取ETF历史数据
        etf_file = ETF_DATA_DIR / f"{etf_code}.csv"
        if not etf_file.exists():
            logger.warning(f"ETF数据文件不存在: {etf_file}")
            continue
        
        try:
            # 读取ETF历史数据，确保日期列是日期类型
            etf_df = pd.read_csv(etf_file, encoding="utf-8-sig", parse_dates=["日期"])
            
            # 按日期排序（从旧到新）
            etf_df = etf_df.sort_values(by="日期", ascending=True)
            
            # 使用ETF数据中的"涨跌幅"列
            
            # 找出连续上涨的天数
            consecutive_rising_days = 0
            cumulative_gain = 0
            recent_3_days = []
            recent_3_dates = []
            
            # 从最新日期开始往前找连续上涨的天数
            # 先按日期排序（从新到旧）
            etf_df_reverse = etf_df.sort_values(by="日期", ascending=False)
            
            for _, row in etf_df_reverse.iterrows():
                if row["涨跌幅"] > 0:
                    consecutive_rising_days += 1
                    cumulative_gain += row["涨跌幅"]
                    # 记录最近3天的涨幅和日期
                    if len(recent_3_days) < 3:
                        recent_3_days.append(row["涨跌幅"])
                        recent_3_dates.append(row["日期"].strftime("%Y-%m-%d"))
                else:
                    break
            
            # 只处理连涨天数≥3天的ETF
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
                
                # 分析量价形态
                volume_price_pattern = get_volume_price_summary(
                    etf_df_reverse,
                    price_change_col="涨跌幅",
                    volume_col="成交量",
                    turnover_col="换手率",
                    days=3
                )
                
                # 添加到结果列表
                result_row = {
                    "ETF代码": etf_code,
                    "ETF名称": etf_name,
                    "总市值": etf_total_market_value,
                    "连涨天数": consecutive_rising_days,
                    "累计涨幅": cumulative_gain,
                    "最近3日涨幅": recent_3_gain,
                    "量价形态": volume_price_pattern
                }
                
                # 添加最近3天的日期和涨幅
                for i in range(3):
                    if recent_3_dates[i]:
                        result_row[recent_3_dates[i]] = recent_3_days[i]
                
                result_data.append(result_row)
                
        except Exception as e:
            logger.error(f"分析ETF {etf_code} 时出错: {e}")
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
        fixed_columns = ["ETF代码", "ETF名称", "总市值", "连涨天数", "累计涨幅", "最近3日涨幅", "量价形态"]
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
        
        # 按连涨天数、累计涨幅、三日涨幅降序排序
        result_df = result_df.sort_values(
            by=["连涨天数", "累计涨幅", "最近3日涨幅"],
            ascending=False
        )
        result_df.to_csv(ANALYZER_DIR / "etf_rising.csv", index=False, encoding="utf-8-sig")
        return result_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    rs=_analyzer()
    print(rs)