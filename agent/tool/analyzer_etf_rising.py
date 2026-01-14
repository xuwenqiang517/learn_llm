from agent.tool.rising_analyzer_base import RisingAnalyzer, AnalyzerConfig
import pandas as pd


def _analyzer():
    config = AnalyzerConfig(
        name="ETF连涨分析器",
        list_file=".temp/data/base/etf_list.csv",
        data_dir=".temp/data/etf_data",
        output_file="etf_rising.csv",
        result_columns=["ETF代码", "ETF名称", "总市值", "连涨天数", "累计涨幅", "最近3日涨幅",
                        "连涨类型", "量价形态", "量能均线", "MACD状态", "MACD差值"],
        filter_list=lambda df: df.copy() if "总市值" not in df.columns else (
            df.assign(**{
                "_市值数值": pd.to_numeric(df["总市值"].str.replace("亿", ""), errors="coerce")
            }).query("_市值数值 >= 1.0").drop(columns=["_市值数值"])
        ) if "总市值" in df.columns else df,
        get_item_info=lambda item: {"名称": item["名称"], "总市值": item.get("总市值", "")},
        analyze_volume_price=lambda df: _get_etf_volume_pattern(df),
        build_result_row=lambda **kwargs: _build_etf_row(**kwargs),
        format_result=lambda df: df.sort_values(by=["连涨天数", "累计涨幅", "最近3日涨幅"],
                                                ascending=[False, False, False]),
        strong_gain=50, mild_gain=20, stable_gain=15
    )
    return RisingAnalyzer(config).analyze()


def _get_etf_volume_pattern(df):
    from agent.tool.volume_price_analyzer import analyze_volume_price_pattern
    df_sorted = df.sort_values(by="日期", ascending=False)
    patterns = analyze_volume_price_pattern(
        df_sorted,
        price_change_col="涨跌幅",
        volume_col="成交量",
        days=3
    )
    return "、".join(patterns) if patterns else ""


def _build_etf_row(*, code: str, name: str, info: dict, days: float, total_gain: float,
                   recent_3_gain: float, rising_type: str, volume_pattern: str, tech_signals: dict):
    return {
        "ETF代码": code,
        "ETF名称": name,
        "总市值": info.get("总市值", ""),
        "连涨天数": days,
        "累计涨幅": total_gain,
        "最近3日涨幅": recent_3_gain,
        "连涨类型": rising_type,
        "量价形态": volume_pattern,
        "量能均线": tech_signals.get("volume_ma_pattern", ""),
        "MACD状态": tech_signals.get("macd_status", ""),
        "MACD差值": round(tech_signals.get("macd_diff", 0), 1),
    }


if __name__ == "__main__":
    import pandas as pd
    print(_analyzer())
