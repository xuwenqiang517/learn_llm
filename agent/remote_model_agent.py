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
from utils.data_path_util import get_pick_dir, get_msg_dir, get_pick_etf_file, get_pick_stock_file, get_message_file
from agent.tool.send_msg import send_email

today_str = datetime.now().strftime("%Y%m%d")

PICK_DIR = get_pick_dir()
MSG_DIR = get_msg_dir()

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


@tool(description="获取精选ETF列表 读取今日筛选结果，包含技术面和基本面指标")
def get_pick_etf() -> pd.DataFrame:
    etf_file = get_pick_etf_file(today_str)
    if etf_file.exists():
        print_green(f"读取{etf_file}")
        return pd.read_csv(etf_file, dtype={'代码': str})
    print_red(f"未找到{etf_file}")
    return pd.DataFrame()

@tool(description="获取精选股票列表 读取今日筛选结果，包含技术面和基本面指标")
def get_pick_stock() -> pd.DataFrame:
    stock_file = get_pick_stock_file(today_str)
    if stock_file.exists():
        print_green(f"读取{stock_file}")
        return pd.read_csv(stock_file, dtype={'代码': str})
    print_red(f"未找到{stock_file}")
    return pd.DataFrame()

@tool(description="获取今日消息面数据 包含政策新闻、市场涨跌、概念板块、行业板块、宏观经济、个股消息等")
def get_message_report() -> dict:
    msg_file = get_message_file(today_str)
    if msg_file.exists():
        print_green(f"读取{msg_file}")
        with open(msg_file, "r", encoding="utf-8") as f:
            return json.load(f)
    print_red(f"未找到{msg_file}")
    return {}


model_name="qwen-plus"

def main():
    llm=create_ollama_chat(model="llama3.1:8b")
    # llm = create_dashscope_chat(
    #         model=model_name,
    #         temperature=0.1
    #     )
    agent=create_agent(llm
                        ,checkpointer=InMemorySaver()
                        ,tools=[get_pick_etf,get_pick_stock,get_message_report]
                        ,middleware=[log_model_call,log_tool_call]
    )
    system_msg = SystemMessage("""你是专业的A股证券分析师，基于精选数据和消息面为用户提供投资建议。

## 工作流程
1. 调用 get_pick_etf 和 get_pick_stock 工具获取今日精选的ETF和股票
2. 调用 get_message_report 工具获取消息面数据
3. 数据已包含基本面和技术面指标：PE、PB、总市值、净利润同比、营收同比、MA5/10/20、VOL_MA5/10/20、MACD、RSI、BOLL、ATR、连涨天数、3日涨幅、5日涨幅等
4. 消息面数据包含：政策新闻、市场涨跌统计、概念板块涨幅、行业板块领涨、宏观经济指标、个股相关新闻等
5. 基于数据和消息面深入分析，生成投资建议报告

## 分析要点

1. **基本面分析**
   - 营收增长：查看营收同比、净利润同比指标
   - 估值水平：分析PE（市盈率）、PB（市净率）是否合理
   - 市值特征：总市值、流通市值，判断盘子大小
   - 盈利质量：净利润增长是否稳定

2. **技术面分析**
   - 均线趋势：MA5 > MA10 > MA20 多头排列表示上升趋势
   - 量能配合：VOL_MA5 > VOL_MA10 > VOL_MA20 量能均线多头
   - 动量指标：MACD金叉/死叉、RSI超买超卖
   - 波动性：ATR反映波动幅度，BOLL反映震荡区间
   - 短期涨幅：3日涨幅、5日涨幅判断是否追高

3. **消息面分析**
   - 查看政策新闻，判断对相关板块的影响
   - 查看概念板块领涨情况，匹配精选股票的行业归属
   - 查看个股相关新闻，如资产重组、业绩公告等重大事件
   - 结合宏观经济数据判断市场整体环境

4. **强势标的筛选**
   - 涨幅适中（10%-30%），连续上涨
   - 量能配合（量能均线多头）
   - 基本面良好（营收/净利润增长为正）
   - 换手率合理（3%-10%最佳）
   - 有热点概念或政策利好支撑

5. **风险识别**
   - 涨幅过大（>50%）谨慎追高
   - 无量上涨风险（换手率过低）
   - 业绩亏损（净利润同比为负）
   - 高估值（PE/PB异常高）
   - 个股有利空新闻或停牌风险

## 输出要求

1. **ETF精选分析**（2-5只，分析其投资价值）
2. **股票重点关注**（5-10只，排序推荐）
3. **消息面热点解读**（解读与精选标的相关的政策和行业新闻）
4. **风险提示**（规避高位无量、业绩亏损、高估值、有利空消息的标的）

用简洁专业的语言给出分析结论，数据支撑论点。""")
    human_msg=HumanMessage(content=f"分析{today_str}精选的ETF和股票，基于基本面和技术面指标给出投资建议")
    messages=[system_msg,human_msg]
    res=agent.invoke({"messages": messages},{"configurable": {"thread_id": "1"}})

    print("="*60)
    
    print_red(res["messages"][-1].content)
    send_email(subject=f"【JDb】{model_name}_大模型分析{today_str}",msg_content=res["messages"][-1].content+f"\n分析由{model_name}模型生成")
    
    
        
if __name__ == "__main__":
    main()
