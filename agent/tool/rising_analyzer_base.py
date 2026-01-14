from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
import pandas as pd
from utils.file_util import FileUtil
from utils.log_util import LogUtil
from agent.tool.volume_price_analyzer import get_technical_signals

logger = LogUtil.get_logger(__name__)


def classify_rising_type(days: int, total_gain: float, recent_3_gain: float, volume_pattern: str,
                         strong_gain: float = 100, mild_gain: float = 30, stable_gain: float = 20) -> str:
    if days >= 10 and total_gain >= strong_gain:
        if "放量" in volume_pattern:
            return "强放量连涨"
        elif "价涨量缩" in volume_pattern:
            return "强加速连涨"
        else:
            return "强势连涨"
    elif days >= 7 and total_gain >= mild_gain:
        if "放量" in volume_pattern:
            return "温和放量连涨"
        elif "价涨量缩" in volume_pattern:
            return "温和加速连涨"
        else:
            return "温和连涨"
    elif days >= 5 and total_gain >= stable_gain:
        if recent_3_gain >= 30:
            return "后程发力"
        else:
            return "稳步连涨"
    elif "价涨量缩" in volume_pattern:
        return "缩量上涨"
    elif "放量" in volume_pattern:
        return "放量试盘"
    else:
        return "普通连涨"


class AnalyzerConfig:
    """分析器配置"""

    def __init__(
        self,
        name: str,
        list_file: str,
        data_dir: str,
        output_file: str,
        result_columns: List[str],
        filter_list: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        get_item_info: Optional[Callable[[pd.Series], Dict[str, Any]]] = None,
        analyze_volume_price: Optional[Callable[[pd.DataFrame], str]] = None,
        build_result_row: Optional[Callable] = None,
        format_result: Optional[Callable[[pd.DataFrame], None]] = None,
        strong_gain: float = 100,
        mild_gain: float = 30,
        stable_gain: float = 20,
        default_info: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.list_file = Path(list_file)
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.result_columns = result_columns
        self.filter_list = filter_list or (lambda df: df)
        self.get_item_info = get_item_info or (lambda item: {})
        self.analyze_volume_price = analyze_volume_price or (lambda df: "")
        self.build_result_row = build_result_row
        self.format_result = format_result or (lambda df: None)
        self.strong_gain = strong_gain
        self.mild_gain = mild_gain
        self.stable_gain = stable_gain
        self.default_info = default_info or {}


class RisingAnalyzer(ABC):
    """连涨分析器"""

    def __init__(self, config: AnalyzerConfig):
        self.config = config
        BASE_DIR = Path(__file__).parent.parent.parent
        TEMP_DIR = BASE_DIR / ".temp"
        DATA_DIR = TEMP_DIR / "data"
        BASE_DATA_DIR = DATA_DIR / "base"
        ANALYZER_DIR = DATA_DIR / "analyzer"
        FileUtil.ensure_dirs(TEMP_DIR, BASE_DATA_DIR, ANALYZER_DIR)

        self.data_dir = BASE_DIR / config.data_dir
        self.list_file = BASE_DIR / config.list_file
        self.output_path = ANALYZER_DIR / config.output_file

    def analyze(self) -> pd.DataFrame:
        try:
            item_list_df = pd.read_csv(self.list_file, encoding="utf-8-sig", dtype={"代码": str})
            item_list_df = self.config.filter_list(item_list_df)
            result_data = []

            for _, item in item_list_df.iterrows():
                code = item["代码"]
                name = item["名称"]
                item_info = self.config.get_item_info(item)

                item_file = self.data_dir / f"{code}.csv"
                if not item_file.exists():
                    continue

                try:
                    item_df = pd.read_csv(item_file, encoding="utf-8-sig", parse_dates=["日期"])
                    item_df = item_df.sort_values(by="日期", ascending=True)

                    if "涨幅" not in item_df.columns:
                        if "涨跌幅" in item_df.columns:
                            item_df = item_df.copy()
                            item_df["涨幅"] = item_df["涨跌幅"]
                        else:
                            item_df = item_df.copy()
                            item_df["涨幅"] = item_df["收盘"].pct_change() * 100

                    days, total_gain, recent_3_gain = self._calculate_rising(item_df)

                    if days >= 3 and recent_3_gain >= 10:
                        volume_pattern = self.config.analyze_volume_price(item_df)

                        tech_signals = get_technical_signals(
                            item_df.sort_values(by="日期", ascending=False),
                            close_col="收盘",
                            volume_col="成交量",
                            turnover_col="换手率"
                        )

                        rising_type = classify_rising_type(
                            days, total_gain, recent_3_gain, volume_pattern,
                            strong_gain=self.config.strong_gain,
                            mild_gain=self.config.mild_gain,
                            stable_gain=self.config.stable_gain
                        )

                        result_row = self.config.build_result_row(
                            code=code, name=name, info=item_info,
                            days=days, total_gain=total_gain, recent_3_gain=recent_3_gain,
                            rising_type=rising_type, volume_pattern=volume_pattern,
                            tech_signals=tech_signals
                        )
                        result_data.append(result_row)

                except Exception as e:
                    logger.error(f"分析 {code} 时出错: {e}")
                    continue

            if result_data:
                result_df = pd.DataFrame(result_data)
                result_df = self._format_result(result_df)
                result_df.to_csv(self.output_path, index=False, encoding="utf-8-sig")
                return result_df

        except Exception as e:
            logger.error(f"连涨分析出错: {e}")

        return pd.DataFrame()

    def _calculate_rising(self, df: pd.DataFrame) -> tuple:
        df_reverse = df.sort_values(by="日期", ascending=False)
        days, total_gain, recent_3_gain = 0, 0, 0

        for _, row in df_reverse.iterrows():
            if row["涨幅"] > 0:
                days += 1
                total_gain += row["涨幅"]
                if days <= 3:
                    recent_3_gain += row["涨幅"]
            else:
                break

        return days, total_gain, recent_3_gain

    def _format_result(self, df: pd.DataFrame):
        for col in ["累计涨幅", "最近3日涨幅", "MACD差值"]:
            if col in df.columns:
                df[col] = df[col].round(1)
        result = self.config.format_result(df)
        if result is not None:
            return result
        return df
