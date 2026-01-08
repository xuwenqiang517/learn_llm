import json
from pathlib import Path
import sys
from typing import Callable
from langchain_core.messages import human
from pydantic import BaseModel

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

from agent.common import create_ollama_chat
from utils.log_util import print_green, print_red, print_yellow





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

class CustomState(AgentState):
    user_name: str


@tool(description="获取当前时间")
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    llm=create_ollama_chat(model="llama3.1:8b")
    agent=create_agent(llm
                        ,state_schema=CustomState
                        ,checkpointer=InMemorySaver()
                        ,tools=[get_current_time]
                        ,middleware=[log_model_call,log_tool_call]
    )
    system_msg = SystemMessage("You are a helpful assistant.")
    human_msg=HumanMessage(content="你好,我是张三三,现在几点了")
    messages=[system_msg,human_msg]
    res=agent.invoke({"messages": messages},{"configurable": {"thread_id": "1"}})
    print("="*60)
    print_red(res)

    
        
if __name__ == "__main__":
    main()