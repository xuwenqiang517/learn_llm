import akshare as ak
import sys
from pathlib import Path
import pandas as pd
from datetime import date
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import LogUtil,print_green,print_red
from utils.file_util import FileUtil

logger = LogUtil.get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / ".temp"
DATA_DIR = TEMP_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "base"
ETF_DATA_DIR = DATA_DIR / "etf_data"
STOCK_DATA_DIR = DATA_DIR / "stock_data"

FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ETF_DATA_DIR, STOCK_DATA_DIR)

ETF_LIST_FILE = BASE_DATA_DIR / "etf_list.csv"
STOCK_LIST_FILE = BASE_DATA_DIR / "stock_list.csv"
CACHE_EXPIRY_DAYS = 1



def _calculate_ema(df: pd.DataFrame, col: str, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return df[col].ewm(span=period, adjust=False).mean()

def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """计算MACD指标"""
    df = df.copy()
    ema12 = _calculate_ema(df, "收盘", 12)
    ema26 = _calculate_ema(df, "收盘", 26)
    df["DIF"] = (ema12 - ema26).round(2)
    df["DEA"] = _calculate_ema(df, "DIF", 9).round(2)
    df["MACD"] = ((df["DIF"] - df["DEA"]) * 2).round(2)
    return df

def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算RSI指标"""
    df = df.copy()
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df["RSI"] = (100 - 100 / (1 + rs)).round(2)
    return df

def _calculate_boll(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """计算布林带"""
    df = df.copy()
    df["BOLL_MID"] = df["收盘"].rolling(window=period).mean().round(2)
    std = df["收盘"].rolling(window=period).std()
    df["BOLL_UP"] = (df["BOLL_MID"] + 2 * std).round(2)
    df["BOLL_LOW"] = (df["BOLL_MID"] - 2 * std).round(2)
    return df

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算真实波幅"""
    df = df.copy()
    high = df["最高"]
    low = df["最低"]
    close = df["收盘"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=period).mean().round(2)
    return df

def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = df.copy()
    
    df["MA5"] = df["收盘"].rolling(window=5).mean().round(2)
    df["MA10"] = df["收盘"].rolling(window=10).mean().round(2)
    df["MA20"] = df["收盘"].rolling(window=20).mean().round(2)
    
    if "成交量" in df.columns:
        df["VOL_MA5"] = df["成交量"].rolling(window=5).mean().round(2)
        df["VOL_MA10"] = df["成交量"].rolling(window=10).mean().round(2)
        df["VOL_MA20"] = df["成交量"].rolling(window=20).mean().round(2)
    
    df = _calculate_macd(df)
    df = _calculate_rsi(df)
    df = _calculate_boll(df)
    df = _calculate_atr(df)
    
    return df

def _process_file(file_path: Path) -> None:
    """处理单个文件"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        if len(df) < 5:
            return
        
        df = _calculate_indicators(df)
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print_red(f"处理文件失败 {file_path}: {e}")

def _calculate_data(max_workers: int = 20) -> None:
    """计算所有股票和ETF的移动平均线"""
    etf_files = list(ETF_DATA_DIR.glob("*.csv"))
    stock_files = list(STOCK_DATA_DIR.glob("*.csv"))
    
    all_files = etf_files + stock_files
    
    print_green(f"开始计算移动平均线，共 {len(all_files)} 个文件")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_process_file, all_files), desc="计算移动平均线", unit="个", total=len(all_files)))
    
    print_green("移动平均线计算完成")

if __name__ == "__main__":
    _calculate_data()

# 高优先级（推荐添加）：

# 1. MACD - 异同移动平均线
   
#    - 用途：判断趋势方向和买卖信号
#    - 计算：EMA(12) - EMA(26)，信号线 = EMA(9)
# 2. RSI - 相对强弱指标
   
#    - 用途：判断超买超卖（>70超买，<30超卖）
#    - 计算：基于涨跌幅的相对强度
# 3. BOLL - 布林带
   
#    - 用途：判断价格波动通道和支撑阻力
#    - 计算：MA20 ± 2倍标准差
# 4. VOL_MA - 成交量均线
   
#    - 用途：分析量价关系
#    - 计算：MA5、MA10、MA20
# 5. ATR - 真实波幅
   
#    - 用途：衡量波动率，用于止损
#    - 计算：基于最高价、最低价、收盘价