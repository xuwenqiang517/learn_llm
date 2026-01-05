"""
股票搜索工具模块（串联入口）

提供A股股票查询功能的统一入口，串联调用：
- stock_data_updater: 数据更新工具
- stock_rising_calculator: 连续上涨股票计算工具
- send_stock_analysis: 飞书消息发送工具

依赖：
    pip install akshare tabulate tqdm langchain-core mcp

使用示例：
    # 直接运行
    python -m agent.stock_searh_tool
    
    # 作为工具使用
    from agent.stock_searh_tool import search_rising_stocks, get_langchain_tool, get_mcp_tools
    
    # LangChain工具
    tool = get_langchain_tool()
    
    # MCP工具
    mcp_tools = get_mcp_tools()
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.json_util import JsonUtil
from utils.file_util import FileUtil
from utils.log_util import LogUtil

# 导入独立的工具模块
from agent.stock_data_updater import update_stock_data
from agent.stock_rising_calculator import calculate_rising_stocks
from agent.send_stock_analysis import send_latest_analysis
from agent.stock_concept_analyzer import analyze_concepts

logger = LogUtil.get_logger(__name__)

# ==================== 目录结构定义 ====================
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / ".temp"
OUTPUT_DIR = TEMP_DIR / "output"
TOOLS_OUTPUT_DIR = OUTPUT_DIR / "tools"


# ==================== 主入口函数 ====================

def search_rising_stocks(days: int = 3, market: str = "all", 
                        current_date: Optional[str] = None, 
                        save_result: bool = True, 
                        use_cache: bool = True, 
                        min_increase: float = 10.0, 
                        include_kc: bool = False, 
                        include_cy: bool = False,
                        auto_update_cache: bool = True,
                        send_to_feishu: bool = False) -> Dict:
    """
    搜索连续N天上涨的股票（服务主入口）
    
    串联调用数据更新、计算和发送功能。
    
    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        current_date: 查询日期，格式YYYYMMDD，默认为今天
        save_result: 是否保存结果到文件（表格md文件）
        use_cache: 是否使用缓存的查询结果（基于表格md文件）
        min_increase: 最小累计涨幅阈值（%），默认10.0%
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        auto_update_cache: 是否自动更新缓存，默认True
        send_to_feishu: 是否发送到飞书，默认False
        
    Returns:
        包含查询结果的字典（包含data和table两个字段）
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y%m%d")
    
    # 表格文件路径
    table_file = TOOLS_OUTPUT_DIR / f"rising_stocks_{current_date}_{days}days_{market}_{min_increase}pct_kc{include_kc}_cy{include_cy}.md"
    
    # 检查缓存的表格文件
    if use_cache and table_file.exists():
        table_content = FileUtil.read_text(table_file)
        if table_content:
            try:
                # 从文件内容中提取股票数量
                lines = table_content.split('\n')
                title_line = [l for l in lines if '股票数据汇总' in l]
                if title_line:
                    import re
                    match = re.search(r'\((\d+)只\)', title_line[0])
                    stock_count = int(match.group(1)) if match else 0
                else:
                    stock_count = 0
            except Exception:
                stock_count = 0
            
            # 从缓存读取时，重新生成JSON数据（但不保存）
            result_data = {}
            if stock_count > 0:
                from agent.stock_rising_calculator import analyze_rising_stocks
                result_json = analyze_rising_stocks(
                    days=days, market=market, min_increase=min_increase, 
                    include_kc=include_kc, include_cy=include_cy,
                    compress=False
                )
                result_data = JsonUtil.loads(result_json) or {}
            
            return {
                "success": True,
                "message": f"从缓存读取，找到 {stock_count} 只连续{days}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
                "data": result_data,
                "table": table_content,
                "table_path": str(table_file),
                "from_cache": True
            }
    
    # 1. 自动更新缓存（如果需要）
    if auto_update_cache:
        update_ok = update_stock_data(days=days + 12, market=market, force_update=False)
        if not update_ok:
            logger.warning("数据更新失败或不完整，继续使用现有缓存")
    
    # 2. 计算连续上涨股票
    result_json, table_content = calculate_rising_stocks(
        days=days, market=market, min_increase=min_increase, 
        include_kc=include_kc, include_cy=include_cy,
        compress=False, save_table=save_result, table_path=table_file if save_result else None
    )
    
    result = JsonUtil.loads(result_json) or {}
    
    if not result.get("stocks"):
        return {
            "success": True,
            "message": f"未找到符合条件的股票或缓存数据不足（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
            "data": result,
            "table": table_content,
            "table_path": None,
            "from_cache": False
        }
    
    # 3. 发送到飞书（如果需要）
    if send_to_feishu:
        try:
            send_ok = send_latest_analysis(include_table=True)
            if send_ok:
                logger.info("结果已发送到飞书")
            else:
                logger.warning("飞书发送失败")
        except Exception as e:
            logger.error(f"发送到飞书失败: {e}")
    
    return {
        "success": True,
        "message": f"找到 {len(result.get('stocks', []))} 只连续{result.get('rising_days', days)}天上涨的股票（累计涨幅>{min_increase}%，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
        "data": result,
        "table": table_content,
        "table_path": str(table_file) if save_result else None,
        "from_cache": False
    }


def search_concepts(days: int = 5, market: str = "all",
                   current_date: Optional[str] = None,
                   save_result: bool = True,
                   use_cache: bool = True,
                   include_kc: bool = False,
                   include_cy: bool = False,
                   auto_update_cache: bool = True,
                   send_to_feishu: bool = False,
                   top_n: int = 20) -> Dict:
    """
    搜索大盘概念趋势（服务主入口）
    
    串联调用数据更新、计算和发送功能。
    
    Args:
        days: 分析天数，默认5天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        current_date: 查询日期，格式YYYYMMDD，默认为今天
        save_result: 是否保存结果到文件（图表png文件）
        use_cache: 是否使用缓存的查询结果（基于图表png文件）
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False
        auto_update_cache: 是否自动更新缓存，默认True
        send_to_feishu: 是否发送到飞书，默认False
        top_n: 显示前N个概念，默认20
        
    Returns:
        包含查询结果的字典（包含data和chart两个字段）
    """
    if current_date is None:
        current_date = datetime.now().strftime("%Y%m%d")
    
    # 图表文件路径
    chart_file = TOOLS_OUTPUT_DIR / f"concept_analysis_{current_date}_{days}days_{market}_kc{include_kc}_cy{include_cy}.png"
    
    # 检查缓存的图表文件
    if use_cache and chart_file.exists():
        chart_content = str(chart_file)
        if chart_file.exists():
            try:
                # 从缓存读取时，重新生成JSON数据（但不保存）
                result_data = {}
                from agent.stock_concept_analyzer import analyze_concepts_simple
                result_json = analyze_concepts_simple(
                    days=days, market=market,
                    include_kc=include_kc, include_cy=include_cy,
                    compress=False
                )
                result_data = JsonUtil.loads(result_json) or {}
                concept_count = len(result_data.get('concepts', []))
            except Exception:
                concept_count = 0
                result_data = {}
            
            return {
                "success": True,
                "message": f"从缓存读取，找到 {concept_count} 个概念（分析周期{days}天，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
                "data": result_data,
                "chart": chart_content,
                "chart_path": str(chart_file),
                "from_cache": True
            }
    
    # 1. 自动更新缓存（如果需要）
    if auto_update_cache:
        update_ok = update_stock_data(days=days + 7, market=market, force_update=False)
        if not update_ok:
            logger.warning("数据更新失败或不完整，继续使用现有缓存")
    
    # 2. 分析概念趋势
    result_json, chart_content = analyze_concepts(
        days=days, market=market,
        include_kc=include_kc, include_cy=include_cy,
        compress=False, save_chart=save_result, chart_path=chart_file if save_result else None, top_n=top_n
    )
    
    result = JsonUtil.loads(result_json) or {}
    
    if not result.get("concepts"):
        return {
            "success": True,
            "message": f"未找到概念数据或缓存数据不足（分析周期{days}天，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
            "data": result,
            "chart": chart_content,
            "chart_path": None,
            "from_cache": False
        }
    
    # 3. 发送到飞书（如果需要）
    if send_to_feishu:
        try:
            send_ok = send_latest_analysis(include_table=True)
            if send_ok:
                logger.info("结果已发送到飞书")
            else:
                logger.warning("飞书发送失败")
        except Exception as e:
            logger.error(f"发送到飞书失败: {e}")
    
    return {
        "success": True,
        "message": f"找到 {len(result.get('concepts', []))} 个概念（分析周期{result.get('analysis_days', days)}天，科创板{'包含' if include_kc else '排除'}，创业板{'包含' if include_cy else '排除'}）",
        "data": result,
        "chart": chart_content,
        "chart_path": str(chart_file) if save_result else None,
        "from_cache": False
    }


# ==================== LangChain Tool 包装 ====================

def get_langchain_tool():
    """
    获取LangChain工具（使用最新语法）
    
    Returns:
        LangChain Tool对象列表
    """
    try:
        from langchain_core.tools import tool
        
        @tool
        def search_rising_stocks_tool(
            days: int = 3, 
            market: str = "all", 
            min_increase: float = 10.0, 
            include_kc: bool = False, 
            include_cy: bool = False
        ) -> str:
            """
            搜索连续N天上涨的A股股票（基于本地缓存分析，自动更新数据）
            
            功能说明：
            - 自动检查并更新股票数据缓存（本地有数据则跳过远程调用）
            - 从本地缓存读取股票历史数据，分析连续上涨走势
            - 自动过滤ST/*ST股票（避免退市风险）
            - 支持按市场筛选（上交所/深交所/科创板/创业板）
            - 按连涨天数和累计涨幅综合排序
            
            Args:
                days: 连续上涨天数，默认3天（从最近一天往前计算）
                market: 市场类型筛选
                    - "all": 全市场（默认）
                    - "sh": 上海主板（沪市）
                    - "sz": 深圳主板（深市）
                min_increase: 最小累计涨幅阈值，默认10.0%（连涨期间的累计涨幅）
                include_kc: 是否包含科创板（688xxx），默认False
                include_cy: 是否包含创业板（300xxx），默认False
            
            Returns:
                JSON格式的股票搜索结果（用于模型输入）
            """
            logger.info(f"LangChain工具被调用: days={days}, market={market}, min_increase={min_increase}")
            result = search_rising_stocks(
                days=days, 
                market=market, 
                min_increase=min_increase, 
                include_kc=include_kc, 
                include_cy=include_cy,
                auto_update_cache=True
            )
            
            if not result.get("success"):
                return JsonUtil.dumps({"message": result.get("message", "查询失败")})
            
            # 返回JSON数据（压缩格式，减少token）
            data = result.get("data", {})
            return JsonUtil.dumps({
                "success": True, 
                "message": result.get("message", ""), 
                "data": data
            })
        
        @tool
        def search_concepts_tool(
            days: int = 5,
            market: str = "all",
            include_kc: bool = False,
            include_cy: bool = False
        ) -> str:
            """
            分析大盘概念趋势（基于本地缓存分析，自动更新数据）
            
            功能说明：
            - 自动检查并更新股票数据缓存（本地有数据则跳过远程调用）
            - 从本地缓存读取股票历史数据，分析概念板块趋势
            - 自动过滤ST/*ST股票（避免退市风险）
            - 支持按市场筛选（上交所/深交所/科创板/创业板）
            - 按累计平均涨跌幅排序，定位大盘主线趋势
            - 生成可视化图表展示Top概念趋势
            
            Args:
                days: 分析天数，默认5天（最近N个交易日）
                market: 市场类型筛选
                    - "all": 全市场（默认）
                    - "sh": 上海主板（沪市）
                    - "sz": 深圳主板（深市）
                include_kc: 是否包含科创板（688xxx），默认False
                include_cy: 是否包含创业板（300xxx），默认False
            
            Returns:
                JSON格式的概念分析结果（用于模型输入）
            """
            logger.info(f"LangChain工具被调用: days={days}, market={market}")
            result = search_concepts(
                days=days,
                market=market,
                include_kc=include_kc,
                include_cy=include_cy,
                auto_update_cache=True
            )
            
            if not result.get("success"):
                return JsonUtil.dumps({"message": result.get("message", "查询失败")})
            
            # 返回JSON数据（压缩格式，减少token）
            data = result.get("data", {})
            return JsonUtil.dumps({
                "success": True,
                "message": result.get("message", ""),
                "data": data
            })
        
        return [search_rising_stocks_tool, search_concepts_tool]
    except ImportError:
        logger.error("langchain_core未安装，无法创建LangChain工具")
        return None


# ==================== MCP Server 包装 ====================

def get_mcp_tools() -> list:
    """
    获取MCP工具列表
    
    Returns:
        MCP工具列表
    """
    from agent.stock_data_updater import get_mcp_tool as get_updater_tool
    from agent.stock_rising_calculator import get_mcp_tool as get_calculator_tool
    from agent.stock_concept_analyzer import get_mcp_tool as get_analyzer_tool
    
    return [
        get_updater_tool(),
        get_calculator_tool(),
        get_analyzer_tool()
    ]


def handle_mcp_call(tool_name: str, arguments: Dict) -> Dict:
    """
    处理MCP工具调用
    
    Args:
        tool_name: 工具名称
        arguments: 工具参数
        
    Returns:
        工具执行结果
    """
    from agent.stock_data_updater import handle_mcp_call as handle_updater_call
    from agent.stock_rising_calculator import handle_mcp_call as handle_calculator_call
    from agent.stock_concept_analyzer import handle_mcp_call as handle_analyzer_call
    
    if tool_name == "update_stock_data":
        return handle_updater_call(arguments)
    elif tool_name == "analyze_rising_stocks":
        return handle_calculator_call(arguments)
    elif tool_name == "analyze_concepts":
        return handle_analyzer_call(arguments)
    else:
        return {
            "content": [
                {
                    "type": "text",
                    "text": JsonUtil.dumps({"error": f"未知工具: {tool_name}"})
                }
            ],
            "isError": True
        }


# ==================== 主函数 ====================

def main():
    """主函数 - CLI入口"""
    try:
        print("=" * 70)
        print("股票搜索工具 - 开始查询")
        print("=" * 70)
        
        result = search_rising_stocks(
            days=3, 
            market="all", 
            min_increase=10.0, 
            use_cache=False, 
            save_result=True,
            auto_update_cache=True,
            send_to_feishu=False
        )
        
        # 打印查询结果消息
        print(f"\n查询结果: {result.get('message', 'N/A')}")
        
        # 检查是否成功
        if not result.get("success"):
            print("❌ 查询失败")
            return
        
        # 检查是否有表格数据
        table_content = result.get("table", "")
        table_path = result.get("table_path")
        
        if table_path:
            print(f"✅ 表格数据已保存到: {table_path}")
        
        # 如果有表格内容且不是错误消息，打印前几行预览
        if table_content and table_content != "未找到符合条件的股票":
            print("\n" + "=" * 70)
            print("表格内容预览（前30行）:")
            print("=" * 70)
            lines = table_content.split('\n')
            preview_lines = lines[:30]
            print('\n'.join(preview_lines))
            if len(lines) > 30:
                print(f"\n... (共 {len(lines)} 行，完整内容请查看文件)")
            print("=" * 70)
        elif table_content == "未找到符合条件的股票":
            print("⚠️  未找到符合条件的股票")
        
        # 打印数据统计
        data = result.get("data", {})
        if data and isinstance(data, dict):
            stocks = data.get("stocks", [])
            if stocks:
                print(f"\n📊 数据统计:")
                print(f"   - 股票数量: {len(stocks)} 只")
                print(f"   - 连涨天数: {data.get('rising_days', 'N/A')} 天")
                print(f"   - 查询时间: {data.get('query_time', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("查询完成")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
