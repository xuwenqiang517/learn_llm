from agent.tool.rising_analyzer_base import RisingAnalyzer, AnalyzerConfig


def _analyzer():
    config = AnalyzerConfig(
        name="股票连涨分析器",
        list_file=".temp/data/base/stock_list.csv",
        data_dir=".temp/data/stock_data",
        output_file="stock_rising.csv",
        result_columns=["股票代码", "股票名称", "连涨天数", "累计涨幅", "最近3日涨幅",
                        "连涨类型", "量价形态", "量能均线", "MACD状态", "MACD差值"],
        filter_list=lambda df: df[
            ~df["板块类型"].isin(["科创板", "创业板", "北京股"]) &
            ~df["名称"].str.contains("ST")
        ],
        get_item_info=lambda item: {"名称": item["名称"]},
        analyze_volume_price=lambda df: _get_volume_price_pattern(df),
        build_result_row=lambda **kwargs: _build_stock_row(**kwargs),
        format_result=lambda df: _format_stock_result(df),
        strong_gain=100, mild_gain=30, stable_gain=20
    )
    return RisingAnalyzer(config).analyze()


def _get_volume_price_pattern(df):
    from agent.tool.volume_price_analyzer import analyze_volume_price_pattern
    df_sorted = df.sort_values(by="日期", ascending=False)
    patterns = analyze_volume_price_pattern(
        df_sorted,
        price_change_col="涨跌幅",
        volume_col="成交量",
        turnover_col="换手率",
        days=3
    )
    result = "、".join(patterns) if patterns else "无明显量价形态"
    return result.replace("【推荐】", "").strip("、")


def _build_stock_row(*, code: str, name: str, info: dict, days: float, total_gain: float,
                     recent_3_gain: float, rising_type: str, volume_pattern: str, tech_signals: dict):
    return {
        "股票代码": str(code).zfill(6),
        "股票名称": name,
        "连涨天数": days,
        "累计涨幅": total_gain,
        "最近3日涨幅": recent_3_gain,
        "连涨类型": rising_type,
        "量价形态": volume_pattern,
        "量能均线": tech_signals.get("volume_ma_pattern", ""),
        "MACD状态": tech_signals.get("macd_status", ""),
        "MACD差值": round(tech_signals.get("macd_diff", 0), 1),
    }


def _format_stock_result(df):
    if "股票代码" in df.columns:
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df = df.sort_values(by=["连涨天数", "累计涨幅", "最近3日涨幅"],
                        ascending=[False, False, False]).head(150)
    return df


if __name__ == "__main__":
    from agent.tool.volume_price_analyzer import get_volume_price_summary
    print(_analyzer())
