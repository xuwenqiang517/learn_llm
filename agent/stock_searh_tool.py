"""
股票搜索工具模块（串联入口）

提供A股股票查询功能的统一入口，串联调用：
- stock_rising_calculator: 连续上涨股票计算工具
- stock_concept_analyzer: 大盘概念趋势分析工具
- send_stock_analysis: 飞书消息发送工具

依赖：
    pip install akshare tabulate tqdm

使用示例：
    # 直接运行
    python -m agent.stock_searh_tool
    
    # 作为工具使用
    from agent.stock_searh_tool import search_rising_stocks, search_concepts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.stock_rising_calculator import calculate_rising_stocks
from agent.stock_concept_analyzer import analyze_concepts
from agent.send_stock_analysis import send_rising_stocks_table, send_concept_analysis_table
from agent.stock_data_updater import update_stock_data
from utils.json_util import JsonUtil

BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / ".temp"


def main():
    print("=" * 70)
    print("股票搜索工具 - 开始查询")
    print("=" * 70)

    update_ok = update_stock_data(days=15, market="all", force_update=False)
    if not update_ok:
        print("⚠️  数据更新失败或不完整，继续使用现有缓存")

    result_json, _ = calculate_rising_stocks(days=3, market="all", min_increase=10.0, include_kc=False, include_cy=False, compress=False, save_table=True)
    send_rising_stocks_table(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f", temp_dir=str(TEMP_DIR))
    rising_result = JsonUtil.loads(result_json) or {}
    print(f"\n连续上涨股票查询结果: {rising_result.get('message', 'N/A')}")

    if rising_result.get("stocks"):
        table_path = rising_result.get("table_output_path")
        if table_path:
            print(f"✅ 表格数据已保存到: {table_path}")

    result_json, _ = analyze_concepts(days=5, market="all", include_kc=False, include_cy=False, compress=False, save_table=True, top_n=20)
    send_concept_analysis_table(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/c8278f54-8e18-4edc-97bd-0c0abc3ab17f", temp_dir=str(TEMP_DIR))
    concept_result = JsonUtil.loads(result_json) or {}
    print(f"\n大盘概念趋势查询结果: {concept_result.get('message', 'N/A')}")

    if concept_result.get("concepts"):
        concept_table_path = concept_result.get("table_output_path")
        if concept_table_path:
            print(f"✅ 概念分析表格已保存到: {concept_table_path}")

    print("\n" + "=" * 70)
    print("✅ 查询完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
