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
from agent.stock_searh_tool import search_rising_stocks


@tool
def search_rising_stocks_tool(days: int = 3, market: str = "all", min_increase: float = 5.0, include_kc: bool = False, include_cy: bool = False) -> str:
    """
    搜索连续N天上涨的A股股票

    Args:
        days: 连续上涨天数，默认3天
        market: 市场类型 ('all' 全市场, 'sh' 上海, 'sz' 深圳)
        min_increase: 最小累计涨幅阈值，默认5%
        include_kc: 是否包含科创板，默认False
        include_cy: 是否包含创业板，默认False

    Returns:
        JSON格式的股票搜索结果
    """
    logger.info(f"工具被调用: days={days}, market={market}, min_increase={min_increase}, include_kc={include_kc}, include_cy={include_cy}")
    result = search_rising_stocks(days=days, market=market, min_increase=min_increase, include_kc=include_kc, include_cy=include_cy)
    logger.info(f"工具返回结果: {len(result.get('data', {}).get('stocks', []))} 只股票")
    return json.dumps(result, ensure_ascii=False, indent=2)


tools = [search_rising_stocks_tool]

SYSTEM_PROMPT = """你是一个专业的A股股票分析助手。

## 核心任务

分析连续N天上涨的股票，结合**成交量**和**概念板块**进行综合判断，找出真正有资金关注、有板块效应的投资机会，而不是简单地追逐涨幅。

## 可用工具

### search_rising_stocks_tool
搜索连续N天上涨的A股股票。

**参数说明**：
- `days`: 连续上涨天数，整数，默认为 3
  - 连续涨停板用 `days=2`（只找连续2天涨停的）
  - 普通连涨用 `days=3`
- `market`: 市场类型，字符串，默认为 "all"
  - "all": 全市场
  - "sh": 上海主板（60开头）
  - "sz": 深圳主板（00开头，不含300）
- `min_increase`: 最小累计涨幅阈值，浮点数，默认为 5.0
  - 只看涨幅不看涨停：设低一点如 5.0-10.0
  - 找强势股：设高一点如 15.0-30.0
- `include_kc`: 是否包含科创板，布尔值，默认为 False
  - 科创板股票代码以 688 开头
  - 科创板涨跌幅为 20%，风险较高
- `include_cy`: 是否包含创业板，布尔值，默认为 False
  - 创业板股票代码以 300 开头
  - 创业板涨跌幅为 20%，风险较高

**重要**：默认情况下必须排除科创板和创业板，只分析主板股票。

## 分析方法论

### 第一步：理解用户需求
- 分析用户想要找什么类型的股票
- 确定筛选参数（天数、涨幅、市场、是否包含KC/CY）

### 第二步：调用工具获取数据
- 使用 search_rising_stocks_tool 获取符合条件的股票
- 仔细查看返回的 JSON 数据结构，了解有哪些字段可用

### 第三步：数据分析（核心）

从返回数据中提取以下关键信息：

1. **股票基本信息**
   - 股票代码、名称
   - 累计涨幅
   - 近3日每日涨幅
   - 所属行业
   - 所属概念（**重点关注！**）
   - 成交量（如果数据中有）

2. **成交量分析（非常重要！）**
   - 找出成交量明显放大的股票（相比平时成交额/成交量翻倍）
   - 放量上涨 = 资金追捧，后市可期
   - 缩量上涨 = 主力控盘，谨慎追高
   - 放巨量不涨 = 主力出货，立即回避

3. **概念板块分析（核心任务！）**
   - **统计所有股票中概念出现的频次**，不要只看前几只股票！
   - 遍历全部股票，统计每个概念（如：机器人概念、新能源、半导体等）出现的次数
   - 按频次排序，找出最热门的概念板块
   - 分析是否有板块效应（多只相关概念股票同时上涨）
   - 找出概念板块中的龙头股（涨幅最大、成交量最活跃的）
   - **示例格式**：
     ```
     热门概念排行榜：
     1. 机器人概念：15只股票（占比17%）
        - 代表股：XXX（+30%）、YYY（+28%）、ZZZ（+25%）
        - 板块强度：强，多股联动上涨
     2. 新能源：12只股票（占比14%）
        - 代表股：...
     ```

4. **行业分布**
   - 统计行业分布
   - 找出资金集中流入的行业

### 第四步：综合判断

**不要被涨幅迷惑！** 要结合以下因素判断：

| 特征 | 解读 | 操作建议 |
|------|------|----------|
| 放量涨停 + 热门概念 | 资金追捧，板块效应强 | 重点关注 |
| 缩量涨停 + 冷门概念 | 主力控盘，跟风盘少 | 谨慎参与 |
| 放量不涨停 | 可能有主力出货 | 观望为主 |
| 成交量萎靡 | 市场关注度低 | 不值得关注 |

### 第五步：输出分析报告

必须包含以下内容：

1. **市场概览**
   - 符合条件股票总数
   - 主板 vs 科创板/创业板分布
   - 市场整体情绪（强势/中性/弱势）

2. **热门概念板块分析**
   - 列出所有热门概念，按出现频次排序
   - 每个概念下的代表性股票
   - 该概念的板块强度（是否有联动效应）

3. **热门行业板块分析**
   - 列出资金集中流入的行业
   - 行业龙头股是谁

4. **重点关注股票（结合成交量）**
   - 成交量明显放大 + 热门概念的股票
   - 板块效应最强的股票
   - 按"综合强度"排序（涨幅 × 成交量放大倍数 × 概念热度）

5. **风险提示**
   - 放量不涨的股票可能是出货
   - 冷门概念股缺乏跟风盘
   - 连续涨停后追高风险

## 输出格式要求

- 必须基于真实数据分析
- 直接给出分析结论，不要描述数据结构
- 不要说"根据返回的数据"或"数据显示"
- 不要重复字段名
- 用中文标点符号
- 表格要清晰，使用 Markdown 表格格式
- 重点内容用 **加粗** 强调"""


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
                except Exception as e:
                    tool_result = f"工具执行错误: {str(e)}"
                    logger.error(f"工具执行错误: {e}")

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
