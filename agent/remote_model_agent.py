import json
from pathlib import Path
import sys
from typing import Callable
from langchain_core.messages import human
from pydantic import BaseModel
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))



from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import HumanMessage, ToolMessage, SystemMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver  
from langchain.tools import tool
from datetime import datetime

from agent.common import create_ollama_chat, create_dashscope_chat
from utils.log_util import print_green, print_red, print_yellow

from agent.tool.analyzer_etf_rising import _analyzer as etf_analyzer
from agent.tool.analyzer_stock_rising import _analyzer as stock_analyzer
from agent.tool.send_msg import send_email


# 获取当前日期
today_date = datetime.now().strftime("%Y-%m-%d")

@wrap_model_call
def log_model_call(request: ModelRequest,handler: Callable[[ModelRequest], ModelResponse],) -> ModelResponse:
    print_yellow("before_model_call:"+request.messages[-1].content)
    response = handler(request)
    for res in response.result:
        if res.tool_calls:
            print_yellow(f"after_model_call 调用工具:{res.tool_calls}")
        else:
            print_yellow(f"after_model_call: 输出:{res.content}")
    return response

@wrap_tool_call
def log_tool_call(request: ToolCallRequest,handler: Callable[[ToolCallRequest], ToolMessage | Command],) -> ToolMessage | Command:
    print_green(f"before_tool_call: 工具:{request.tool_call.get('name')},参数:{request.tool_call.get('args')}")
    response = handler(request)
    print_green("after_tool_call:"+str(response))
    return response


@tool(description="获取ETF连涨分析 没有参数，默认返回最近3天连涨的etf")
def get_etf_analyzer() -> pd.DataFrame:
    return etf_analyzer()

@tool(description="获取股票连涨分析 没有参数，默认返回最近3天连涨的股票")
def get_stock_analyzer() -> pd.DataFrame:
    return stock_analyzer()


model_name="qwen-plus"

def main():
    # llm=create_ollama_chat(model="llama3.1:8b")
    llm = create_dashscope_chat(
            model=model_name,
            temperature=0.1
        )
    agent=create_agent(llm
                        ,checkpointer=InMemorySaver()
                        ,tools=[get_etf_analyzer,get_stock_analyzer]
                        ,middleware=[log_model_call,log_tool_call]
    )
    system_msg = SystemMessage("""你是专业的A股证券分析师，精通通过数据分析发现投资机会。

## 工作流程
1. 当用户询问证券分析时，根据问题调用 get_etf_analyzer 或 get_stock_analyzer 工具获取数据
2. 工具会返回DataFrame格式的数据，包含代码、名称、涨幅等信息
3. 收到工具返回的数据后，基于数据生成分析报告,包含基本面、技术面、资金面、消息面

## 分析任务
基于搜索结果，深入分析连续上涨证券的投资价值。

## 分析要点

1. **基本面分析**
   - 分析股票的营收结构、估值水平、业务竞争力、盈利结构 在时序上的变化趋势

2. **技术面分析**
   - 均线趋势，结合个股BBI、均线（5日/20日/20日/60日）指标，分析当前趋势方向并推演是否支持趋势的延续
   - 动量多空，结合KDJ、MACD指标，判断该股是否处于超买或超卖区间，并分析短期或中期中期反转概率
   - 相对强弱，对比个股和指数与所属板块的走势是否强势？分析强势或弱势的原因

3. **强势标的筛选**
   - 累计涨幅适中（15%-50%），避免追高
   - 连续上涨天数越多越强势
   - 主营业务清晰、概念热度高的优先

4. **风险识别**
   - 无量上涨的证券风险较高
   - 业绩亏损或基本面恶化的标的规避

## 输出要求

1. **重点关注标的池**（5-10只，放量上涨+热门概念+量价配合）
2. **风险提示**（规避高位无量、业绩亏损等风险标的）

用简洁专业的语言给出分析结论，数据支撑论点。""")
    human_msg=HumanMessage(content="根据最近的市场情况，推荐一下etf和股票，不要仅根据连涨分析，考虑其他因素")
    messages=[system_msg,human_msg]
    res=agent.invoke({"messages": messages},{"configurable": {"thread_id": "1"}})

    print("="*60)
    print_red(res)
    send_email(subject=f"【JDb】{model_name}选股助手_大模型分析{today_date}",msg_content=res["messages"][-1].content+f"\n分析由{model_name}模型生成")

    
        
if __name__ == "__main__":
    main()