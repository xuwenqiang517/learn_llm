import json
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from agent.stock_searh_tool import search_rising_stocks
from agent.stock_rising_calculator import generate_table_from_results
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
TOOLS_OUTPUT_DIR = BASE_DIR / ".temp" / "output" / "tools"
ANALYSIS_OUTPUT_DIR = BASE_DIR / ".temp" / "output" / "analysis"
from agent.send_stock_analysis import send_latest_analysis
from agent.common import create_ollama_chat, create_dashscope_chat


MODEL_CONFIG = {
    "remote": {
        "model": "qwen-plus",
        "temperature": 0.1,
        "params": "未知参数量"
    },
    "local": {
        "model": "qwen3:32b",
        "temperature": 0.3,
        "params": "32B"
    }
}


class StockCallbackHandler(BaseCallbackHandler):
    """股票分析回调处理器"""

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.iteration = 0
        self.last_tool_result: Optional[str] = None
        self.tool_call_args: Optional[dict] = None

    def on_chat_model_start(self, serialized, inputs, **kwargs) -> None:
        self.iteration += 1
        if self.debug:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"迭代 {self.iteration}")
            logger.info(f"{'=' * 60}")

    def on_llm_end(self, response, **kwargs) -> None:
        if hasattr(response, 'generations') and response.generations:
            generation = response.generations[0][0] if response.generations[0] else None
            if generation:
                content = generation.text or ""
                logger.info(f"模型响应: 内容长度 {len(content)} 字符")
                if self.debug and len(content) > 200:
                    logger.info(f"内容预览: {content[:200]}...")
                
                if not content or len(content.strip()) == 0:
                    logger.info("无内容返回，可能需要调用工具")
                else:
                    logger.info("返回最终结果")
                    self._save_and_send_result(content)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        if self.debug:
            function_name = serialized.get("name", "unknown")
            try:
                function_args = json.loads(input_str)
                logger.info(f"执行工具: {function_name}")
                logger.info(f"参数: {function_args}")
            except json.JSONDecodeError:
                logger.info(f"执行工具: {function_name}")

    def on_tool_end(self, output, **kwargs) -> None:
        if self.debug:
            logger.info(f"工具返回: {len(str(output))} 字符")

    def _save_and_send_result(self, result_content: str, config: dict = None) -> None:
        """保存模型分析结果到 output/analysis/ 目录"""
        ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = ANALYSIS_OUTPUT_DIR / f"stock_analysis_{timestamp}.md"
        output_file.write_text(result_content, encoding="utf-8")
        logger.info(f"分析结果已保存到: {output_file}")

        logger.info("分析完成，自动发送结果到飞书...")
        send_latest_analysis()


@tool
def search_rising_stocks_tool(days: int = 3, market: str = "all", min_increase: float = 10.0, include_kc: bool = False, include_cy: bool = False) -> str:
    """
    搜索连续N天上涨的A股股票（基于本地缓存分析，完全离线运行）

    功能说明：
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
        JSON格式的股票搜索结果（压缩模式，减少token消耗）

        **字段说明（压缩格式）：**
        - **c**: 股票代码（code）
        - **n**: 股票名称（name）
        - **i**: 所属行业（industry，前4字符）
        - **con**: 概念板块列表（concepts，前8字符，最多3个）
        - **p**: 最新收盘价（price，保留2位小数）
        - **inc3**: 最近3天累计涨幅（3日涨幅，格式：+XX.X%）
        - **inc_r**: 连涨期间累计涨幅（连涨涨幅，格式：+XX.X%）
        - **r**: 实际连涨天数（rising_days）
        - **d**: 最近3天每日涨跌数据（daily_data，逗号分隔，格式：MM-DD:+X.X%）

        **完整格式字段说明：**
        - code: 股票代码
        - name: 股票名称
        - industry: 所属行业
        - concepts: 所属概念板块列表
        - rising_days: 实际连涨天数
        - rising_total_increase: 连涨期间累计涨幅（从连涨第1天到最后一天的涨幅总和）
        - total_increase: 最近3天累计涨幅（包含非连涨日的总涨幅）
        - daily_data: 最近3天每日涨跌数据
        - last_close: 最新收盘价
    """
    logger.info(f"工具被调用: days={days}, market={market}, min_increase={min_increase}, include_kc={include_kc}, include_cy={include_cy}")
    result = search_rising_stocks(days=days, market=market, min_increase=min_increase, include_kc=include_kc, include_cy=include_cy)
    logger.info(f"工具返回结果: {result.get('message', 'N/A')}")
    
    if not result.get("success"):
        return json.dumps({"message": result.get("message", "查询失败")}, ensure_ascii=False)
    
    data = result.get("data", {})
    table_content = result.get("table", "")
    
    logger.info(f"表格内容类型: {type(table_content)}, 长度: {len(table_content)}")
    
    # 表格文件已由 search_rising_stocks 保存到 output/tools/ 目录
    # 这里只记录日志，不再重复保存
    table_path = result.get("table_path")
    if table_path:
        logger.info(f"表格数据已保存到: {table_path}")
    elif table_content:
        logger.info("表格内容已生成，但未保存到文件")
    else:
        logger.warning("表格内容为空")
    
    return json.dumps({"success": True, "message": result.get("message", ""), "data": data}, ensure_ascii=False, indent=2)


tools = [search_rising_stocks_tool]

SYSTEM_PROMPT = """你是专业的A股股票分析师，精通通过数据分析发现投资机会。

## 工作流程
1. 当用户询问股票分析时，首先调用 search_rising_stocks_tool 工具获取股票数据
2. 工具会返回JSON格式的股票数据，包含股票代码、名称、行业、概念、涨幅等信息
3. 收到工具返回的数据后，**立即停止调用工具**，基于数据生成分析报告
4. 不要重复调用工具，直接基于已获取的数据进行分析

## 分析任务
基于搜索结果，深入分析连续上涨股票的投资价值。

## 分析要点

1. **概念板块分析**
   - 统计所有股票的概念分布，找出出现频次最高的热门概念
   - 分析概念板块的联动效应（多只同概念股票同时上涨）
   - 识别概念炒作的持续性

2. **成交量分析**
   - 对比近期成交量与历史均值，识别放量上涨的股票
   - 分析成交量放大的股票是否配合价格上涨
   - 关注量价配合良好的标的

3. **强势股筛选**
   - 累计涨幅适中（15%-50%），避免追高
   - 连续上涨天数越多越强势
   - 主营业务清晰、概念热度高的优先

4. **风险识别**
   - 涨幅过大（>60%）的股票谨慎追高
   - 无量上涨的股票风险较高
   - 业绩亏损或基本面恶化的标的规避

## 输出要求
1. **热门概念排行榜**（Top 5-8，按出现频次排序，标注每概念包含的股票数）
2. **重点关注股票池**（5-10只，放量上涨+热门概念+量价配合）
3. **风险提示**（规避高位无量、业绩亏损等风险标的）

用简洁专业的语言给出分析结论，数据支撑论点。"""


def run_stock_analysis(user_query: str, debug: bool = False, model_type: str = "remote") -> str:
    """
    运行股票分析Agent（使用工具调用模式）

    Args:
        user_query: 用户查询
        debug: 是否开启调试模式
        model_type: 模型类型，可选 "remote"（远程DashScope）或 "local"（本地Ollama）
    """
    logger.info(f"开始处理用户查询: {user_query}")
    logger.info(f"使用模型类型: {model_type}")

    callback_handler = StockCallbackHandler(debug=debug)

    config = MODEL_CONFIG.get(model_type, MODEL_CONFIG["remote"])

    if model_type == "remote":
        llm = create_dashscope_chat(
            model=config["model"],
            temperature=config["temperature"]
        )
        logger.info(f"创建远程模型完成: {config['model']}")
    else:
        llm = create_ollama_chat(
            model=config["model"],
            base_url="http://localhost:11434",
            temperature=config["temperature"]
        )
        logger.info(f"创建本地模型完成: {config['model']}")

    return _run_with_tools(llm, config, user_query, callback_handler)


def _run_with_tools(llm, config, user_query, callback_handler) -> str:
    """使用工具调用的模式运行（适用于支持 tools 的模型）"""
    logger.info("使用工具调用模式运行...")

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]
    max_iterations = 2
    tool_success = False

    llm_with_tools = llm.bind_tools(tools)

    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages, config={"callbacks": [callback_handler]})

        response_message = response

        logger.info(f"响应类型: {type(response_message)}")
        logger.info(f"响应内容: {response_message.content if hasattr(response_message, 'content') else 'N/A'}")
        logger.info(f"响应 tool_calls: {response_message.tool_calls if hasattr(response_message, 'tool_calls') else 'N/A'}")

        if not hasattr(response_message, 'tool_calls') or not response_message.tool_calls:
            if not tool_success:
                error_msg = "工具调用失败，无法获取股票数据，分析终止"
                logger.error(error_msg)
                callback_handler._save_and_send_result(error_msg, config)
                return error_msg
            result_content = response_message.content or ""
            model_info = f"\n\n---\n**分析模型**: {config['model']} ({config['params']})\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            final_result = result_content + model_info
            callback_handler._save_and_send_result(final_result, config)
            return final_result

        for call in response_message.tool_calls:
            call_id = call["id"]
            function_name = call["name"]
            function_args = call["args"]

            tool_func = next((t for t in tools if t.name == function_name), None)
            if tool_func:
                try:
                    tool_result = tool_func.invoke(function_args)
                    tool_success = True
                    callback_handler.last_tool_result = tool_result
                    callback_handler.tool_call_args = function_args
                except Exception as e:
                    tool_result = f"工具执行错误: {str(e)}"
                    logger.error(f"工具执行错误: {e}")
                    callback_handler.last_tool_result = None
                    error_msg = f"工具执行失败: {str(e)}，分析终止"
                    callback_handler._save_and_send_result(error_msg, config)
                    return error_msg

                messages.append(AIMessage(
                    content="",
                    tool_calls=[call]
                ))
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=call_id,
                    name=function_name
                ))
            else:
                logger.warning(f"未找到工具: {function_name}")
                messages.append(AIMessage(content=f"错误：未找到工具 {function_name}"))

    logger.warning("达到最大迭代次数")
    return "Agent执行超时"


def main():
    """主函数 - 股票分析Agent演示"""
    print("=" * 70)
    print("股票分析 Agent")
    print("=" * 70)

    user_query = "分析最近连续3天上涨的A股股票"
    print(f"\n用户查询: {user_query}")
    print("-" * 70)

    result = run_stock_analysis(user_query, debug=True, model_type="remote")
    print("\n" + "=" * 70)
    print("远程模型结果:")
    print("=" * 70)
    print(result)
    print("-" * 70)

if __name__ == "__main__":
    main()
