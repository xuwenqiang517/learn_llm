import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableSequence

from agent.common import create_dashscope_chat
from agent.stock_searh_tool import search_rising_stocks, generate_table_from_results
from agent.send_stock_analysis import send_latest_analysis


@tool
def search_rising_stocks_tool(days: int = 3, market: str = "all", min_increase: float = 10.0, include_kc: bool = False, include_cy: bool = False) -> str:
    """
    搜索连续N天上涨的A股股票

    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        min_increase: 最小累计涨幅阈值，默认5%
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False

    Returns:
        JSON格式的股票搜索结果（压缩模式，减少token消耗）
    """
    logger.info(f"工具被调用: days={days}, market={market}, min_increase={min_increase}, include_kc={include_kc}, include_cy={include_cy}")
    df = search_rising_stocks(days=days, market=market, min_increase=min_increase, include_kc=include_kc, include_cy=include_cy)
    logger.info(f"工具返回结果: {len(df)} 只股票")

    from agent.stock_searh_tool import format_stock_result
    return format_stock_result(df, days, compress=True)


tools = [search_rising_stocks_tool]

SYSTEM_PROMPT = """你是专业的A股股票分析师，精通通过数据分析发现投资机会。

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


def run_stock_analysis(user_query: str, debug: bool = False) -> str:
    """
    运行股票分析Agent（使用DashScope远程模型 + OpenAI兼容API）

    Args:
        user_query: 用户查询
        debug: 是否开启调试模式

    Returns:
        Agent响应结果
    """
    import os
    from openai import OpenAI

    logger.info(f"开始处理用户查询: {user_query}")

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    logger.info("创建 DashScope Client 完成")

    messages = [{"role": "user", "content": user_query}]
    max_iterations = 5
    iteration = 0
    last_tool_result = None
    tool_call_args = None

    while iteration < max_iterations:
        iteration += 1
        if debug:
            logger.info(f"\n{'='*60}")
            logger.info(f"迭代 {iteration}/{max_iterations}")
            logger.info(f"{'='*60}")

        response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                    temperature=0.1,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "search_rising_stocks_tool",
                            "description": "搜索连续N天上涨的A股股票",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "days": {
                                        "type": "integer",
                                        "description": "连续上涨天数",
                                        "default": 3
                                    },
                                    "market": {
                                        "type": "string",
                                        "description": "市场类型",
                                        "enum": ["all", "sh", "sz"],
                                        "default": "all"
                                    },
                                    "min_increase": {
                                        "type": "number",
                                        "description": "最小累计涨幅阈值（%）",
                                        "default": 5.0
                                    },
                                    "include_kc": {
                                        "type": "boolean",
                                        "description": "是否包含科创板",
                                        "default": False
                                    },
                                    "include_cy": {
                                        "type": "boolean",
                                        "description": "是否包含创业板",
                                        "default": False
                                    }
                                },
                                "required": ["days", "market", "min_increase"]
                            }
                        }
                    }]
                )

        response_message = response.choices[0].message
        logger.info(f"模型响应: role={response_message.role}")

        if response_message.content:
            content = response_message.content.strip()
            if debug:
                logger.info(f"内容长度: {len(content)} 字符")
                logger.info(f"内容预览: {content[:200]}...")

        tool_calls = response_message.tool_calls
        if not tool_calls:
            logger.info("无需调用工具，返回最终结果")

            result_content = response_message.content or ""
            output_dir = Path("/Users/JDb/Desktop/github/learn_llm/.temp")
            output_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"stock_analysis_{timestamp}.md"
            output_file.write_text(result_content, encoding="utf-8")
            logger.info(f"分析结果已保存到: {output_file}")

            if last_tool_result and tool_call_args:
                try:
                    tool_result_data = json.loads(last_tool_result)
                    table_content = generate_table_from_results(tool_result_data)
                    table_file = output_dir / f"stock_table_{timestamp}.md"
                    table_file.write_text(table_content, encoding="utf-8")
                    logger.info(f"表格数据已保存到: {table_file}")
                except json.JSONDecodeError as e:
                    logger.warning(f"解析工具结果失败，跳过表格生成: {e}")
                except Exception as e:
                    logger.warning(f"生成表格失败: {e}")

            try:
                logger.info("分析完成，自动发送结果到飞书...")
                send_success = send_latest_analysis()
                if send_success:
                    logger.info("✅ 飞书消息发送成功")
                else:
                    logger.warning("⚠️ 飞书消息发送失败")
            except Exception as e:
                logger.error(f"发送飞书消息异常: {e}")

            return result_content

        for call in tool_calls:
            call_id = call.id
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)

            if debug:
                logger.info(f"执行工具: {function_name}")
                logger.info(f"参数: {function_args}")

            tool_func = next((t for t in tools if t.name == function_name), None)
            if tool_func:
                try:
                    tool_result = tool_func.invoke(function_args)
                    if debug:
                        logger.info(f"工具返回: {len(tool_result)} 字符")
                    last_tool_result = tool_result
                    tool_call_args = function_args
                except Exception as e:
                    tool_result = f"工具执行错误: {str(e)}"
                    logger.error(f"工具执行错误: {e}")
                    last_tool_result = None

                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": call.function.arguments
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result
                })
            else:
                logger.warning(f"未找到工具: {function_name}")
                messages.append({
                    "role": "assistant",
                    "content": f"错误：未找到工具 {function_name}"
                })

    logger.warning("达到最大迭代次数")
    return "Agent执行超时"


def _run_normal(agent, inputs: dict) -> str:
    """正常运行模式"""
    logger.info("使用正常模式运行...")
    
    final_output = ""
    step = 0

    for chunk in agent.stream(inputs, stream_mode="debug"):
        step += 1
        chunk_type = chunk.get("type", "unknown")
        logger.debug(f"Step {step} - Chunk Type: {chunk_type}")

        if chunk_type == "task_result":
            payload = chunk.get("payload", {})
            messages = payload.get("result", {}).get("messages", [])
            
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        logger.debug(f"  工具调用: {len(msg.tool_calls)} 个")
                    else:
                        logger.debug(f"  最终响应: {msg.content[:100]}...")
                        final_output = msg.content

        elif chunk_type == "task":
            logger.debug(f"任务状态: {chunk.get('status')}")

    logger.info(f"运行完成，最终输出长度: {len(final_output)}")
    return final_output if final_output else "未获取到有效响应"


def _run_with_debug_stream(agent, inputs: dict) -> str:
    """调试模式 - 使用 debug stream 模式获取详细信息"""
    logger.info("使用 debug 模式运行...")

    final_output = ""
    step = 0

    for chunk in agent.stream(inputs, stream_mode="debug"):
        step += 1
        chunk_type = chunk.get("type", "unknown")
        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step} - Chunk Type: {chunk_type}")
        logger.info(f"{'='*60}")

        if chunk_type == "task_result":
            payload = chunk.get("payload", {})
            name = payload.get("name", "unknown")
            error = payload.get("error")
            result = payload.get("result", {})

            logger.info(f"节点名称: {name}")
            if error:
                logger.error(f"错误: {error}")

            messages = result.get("messages", [])
            for msg in messages:
                msg_type = type(msg).__name__
                logger.info(f"  消息类型: {msg_type}")

                if isinstance(msg, AIMessage):
                    logger.info(f"  内容长度: {len(msg.content)} 字符")
                    logger.info(f"  工具调用: {len(msg.tool_calls)} 个")
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            logger.info(f"    - {tc.get('name', 'unknown')}")
                    else:
                        # 这是最终响应
                        final_output = msg.content
                        logger.info(f"  最终响应预览: {msg.content[:200]}...")

                elif isinstance(msg, ToolMessage):
                    content_len = len(msg.content)
                    logger.info(f"  内容长度: {content_len} 字符")
                    logger.info(f"  工具名称: {msg.name}")
                    if content_len > 200:
                        logger.info(f"  内容预览: {msg.content[:200]}...")
                    else:
                        logger.info(f"  内容: {msg.content}")

        elif chunk_type == "task":
            logger.info(f"任务类型: {chunk.get('status')}")

        elif chunk_type == "input":
            logger.info(f"输入: {chunk.get('input')}")

        else:
            logger.info(f"完整内容: {json.dumps(chunk, ensure_ascii=False, indent=2)[:500]}")

    logger.info(f"\n{'='*60}")
    logger.info(f"运行完成，最终输出长度: {len(final_output)}")
    logger.info(f"{'='*60}\n")

    return final_output if final_output else "未获取到有效响应"


def main():
    """主函数 - 股票分析Agent演示"""
    print("=" * 70)
    print("股票分析 Agent")
    print("=" * 70)

    user_query = "分析最近连续3天上涨的A股股票"
    print(f"\n用户查询: {user_query}")
    print("-" * 70)

    # 使用调试模式运行
    result = run_stock_analysis(user_query, debug=True)
    print("\n" + "=" * 70)
    print("最终结果:")
    print("=" * 70)
    print(result)
    print("-" * 70)


if __name__ == "__main__":
    main()
