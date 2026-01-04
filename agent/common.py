"""
Ollama 本地大模型调用封装

基于 LangChain Ollama 官方 API，提供简洁的 LLM 实例创建方式。

依赖：
    pip install langchain-ollama

使用示例：
    from agent.common import create_ollama_chat

    # 创建 ChatOllama 实例
    llm = create_ollama_chat(
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
        temperature=0.7
    )

    # 调用
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content="你好")])
"""

import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def create_ollama_chat(
    model: str = "deepseek-r1:32b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int | None = None,
    top_k: int = 40,
    top_p: float = 0.9,
    num_predict: int = 256,
    repeat_penalty: float = 1.1,
    stop: list | None = None,
    format: str | None = None,
) -> ChatOllama:
    """
    创建 ChatOllama 实例

    Args:
        model: 模型名称，如 "qwen2.5:7b"、"llama3.1:8b"
        base_url: Ollama 服务地址，默认 http://localhost:11434
        temperature: 温度参数，越高越有创意，越低越确定性，取值 (0, 1]
        max_tokens: 最大输出 token 数，None 表示不限制
        top_k: 采样时考虑的候选词数量，默认 40
        top_p: 核采样阈值，默认 0.9
        num_predict: 最多预测的 token 数
        repeat_penalty: 重复惩罚系数，默认 1.1
        stop: 停止词列表
        format: 输出格式，"json" 或 None

    Returns:
        ChatOllama 实例
    """
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        num_predict=num_predict,
        repeat_penalty=repeat_penalty,
        stop=stop,
        format=format,
    )



def create_dashscope_chat(
    model: str = "qwen-plus",
    api_key: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """
    创建 DashScope (阿里云百炼) 远程模型实例

    使用远程 Qwen Plus 模型，支持更好的工具调用功能。

    Args:
        model: 模型名称，如 "qwen-plus"、"qwen-max"
        api_key: API Key，若为 None 则从环境变量 DASHSCOPE_API_KEY 读取
        temperature: 温度参数，越低越确定性
        max_tokens: 最大输出 token 数，None 表示不限制

    Returns:
        ChatOpenAI 实例

    Requires:
        pip install langchain-openai
        export DASHSCOPE_API_KEY="your-api-key"
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量，请配置后再使用")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )
