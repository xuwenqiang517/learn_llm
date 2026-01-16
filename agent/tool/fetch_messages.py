import akshare as ak
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.log_util import print_green, print_red
from utils.file_util import FileUtil

DATA_DIR = Path(__file__).parent.parent.parent / ".temp" / "data"
PICK_DIR = DATA_DIR / "pick"
MSG_DIR = DATA_DIR / "msg"

FileUtil.ensure_dirs(MSG_DIR)

today_str = datetime.now().strftime("%Y%m%d")
today_date = datetime.now().strftime("%Y-%m-%d")


def _fetch_stock_news(code: str) -> dict:
    """获取单只股票的消息面信息"""
    try:
        news_list = []

        try:
            df = ak.stock_news_em(symbol=code)
            if not df.empty and len(df) > 0:
                for _, row in df.head(5).iterrows():
                    title = str(row.get("新闻标题", ""))
                    if title and title != "nan":
                        news_list.append({
                            "标题": title[:100],
                            "发布时间": str(row.get("发布时间", ""))
                        })
        except Exception:
            pass

        return {
            "代码": code,
            "新闻列表": news_list[:8]
        }
    except Exception:
        return {"代码": code, "新闻列表": []}


def _fetch_policy_news() -> list:
    """获取整体政策消息"""
    news_list = []

    for keyword in ["政策", "要闻", "经济"]:
        try:
            df = ak.stock_news_em(symbol=keyword)
            if not df.empty and len(df) > 0:
                for _, row in df.head(5).iterrows():
                    title = str(row.get("新闻标题", ""))
                    if title and title != "nan":
                        news_list.append({
                            "标题": title[:150],
                            "发布时间": str(row.get("发布时间", ""))
                        })
        except Exception:
            continue

    seen = set()
    unique_news = []
    for news in news_list:
        if news["标题"] not in seen:
            seen.add(news["标题"])
            unique_news.append(news)

    return unique_news[:15]


def _fetch_market_summary() -> dict:
    """获取市场整体情况"""
    try:
        summary = {}

        try:
            df = ak.stock_zh_a_spot_em()
            if not df.empty:
                up_count = len(df[df["涨跌幅"] > 0])
                down_count = len(df[df["涨跌幅"] < 0])
                total = len(df)
                summary["市场状态"] = {
                    "上涨": int(up_count),
                    "下跌": int(down_count),
                    "平盘": int(total - up_count - down_count),
                    "涨停": int(len(df[df["涨跌幅"] >= 9.9])),
                    "跌停": int(len(df[df["涨跌幅"] <= -9.9]))
                }
        except Exception:
            pass

        try:
            df = ak.stock_market_hot_rank()
            if not df.empty:
                summary["热点排行"] = df.head(10)[["股票代码", "股票名称", "热度"]].to_dict("records")
        except Exception:
            pass

        return summary
    except Exception:
        return {}


def _fetch_concept_board() -> list:
    """获取概念板块信息"""
    try:
        df = ak.stock_board_concept_name_em()
        if not df.empty:
            result = []
            for _, row in df.head(20).iterrows():
                result.append({
                    "板块名称": str(row.get("板块名称", "")),
                    "板块代码": str(row.get("板块代码", "")),
                    "涨跌幅": float(row.get("涨跌幅", 0)),
                    "领涨股票": str(row.get("领涨股票", "")),
                    "领涨涨幅": float(row.get("领涨股票-涨跌幅", 0))
                })
            return result
    except Exception:
        pass
    return []


def _fetch_industry_board() -> list:
    """获取行业板块信息"""
    try:
        df = ak.stock_board_industry_name_em()
        if not df.empty:
            result = []
            for _, row in df.head(20).iterrows():
                result.append({
                    "板块名称": str(row.get("板块名称", "")),
                    "板块代码": str(row.get("板块代码", "")),
                    "涨跌幅": float(row.get("涨跌幅", 0)),
                    "领涨股票": str(row.get("领涨股票", "")),
                    "领涨涨幅": float(row.get("领涨股票-涨跌幅", 0))
                })
            return result
    except Exception:
        pass
    return []


def generate_message_report():
    """生成消息面报告"""
    print_green("开始生成消息面报告...")

    etf_file = PICK_DIR / f"etf_{today_str}.csv"
    stock_file = PICK_DIR / f"stock_{today_str}.csv"

    etf_df = pd.read_csv(etf_file, dtype={'代码': str}) if etf_file.exists() else pd.DataFrame()
    stock_df = pd.read_csv(stock_file, dtype={'代码': str}) if stock_file.exists() else pd.DataFrame()

    report = {
        "报告日期": today_date,
        "汇总": {
            "ETF数量": len(etf_df),
            "股票数量": len(stock_df)
        },
        "政策新闻": [],
        "市场状态": {},
        "概念板块": [],
        "行业板块": [],
        "个股新闻": []
    }

    print_green("获取政策消息...")
    report["政策新闻"] = _fetch_policy_news()

    print_green("获取市场整体情况...")
    report["市场状态"] = _fetch_market_summary()

    print_green("获取概念板块...")
    report["概念板块"] = _fetch_concept_board()

    print_green("获取行业板块...")
    report["行业板块"] = _fetch_industry_board()

    if len(stock_df) > 0:
        print_green(f"获取{len(stock_df)}只股票的消息面...")
        stock_codes = stock_df["代码"].astype(str).tolist()
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(_fetch_stock_news, stock_codes), desc="股票消息面", unit="只", total=len(stock_codes)))
        report["个股新闻"] = [r for r in results if len(r.get("新闻列表", [])) > 0]

    output_file = MSG_DIR / f"message_{today_str}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_green(f"消息面报告已保存到: {output_file}")
    print_green(f"  - 股票: {len(report['个股新闻'])} 只 (有消息)")
    print_green(f"  - 政策新闻: {len(report['政策新闻'])} 条")
    print_green(f"  - 概念板块: {len(report['概念板块'])} 个")
    print_green(f"  - 行业板块: {len(report['行业板块'])} 个")

    return report


def load_message_report(date_str: str = None) -> dict:
    """加载指定日期的消息面报告"""
    if date_str is None:
        date_str = today_str

    report_file = MSG_DIR / f"message_{date_str}.json"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    generate_message_report()
