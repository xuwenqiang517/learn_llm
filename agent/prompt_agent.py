"""
Prompt 优化 Agent

将普通 prompt 优化为更结构化、更易被 LLM 理解的 prompt。

依赖：
    pip install langchain-core

使用示例：
    from agent.prompt_agent import optimize_prompt

    optimized = optimize_prompt("写一首关于春天的诗")
    print(optimized)
"""

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from agent.common import create_ollama_chat


OPTIMIZER_SYSTEM_PROMPT = """你是一个专业的 Prompt Engineer，擅长将用户的简单需求转化为结构化、高质量的 prompt。

优化原则：
1. 明确角色定义：指定 AI 扮演什么角色
2. 清晰任务描述：具体说明要做什么
3. 提供上下文：给出必要的背景信息
4. 定义输出格式：指定输出的结构、长度、风格
5. 添加约束条件：明确不要什么、限制条件
6. 分解复杂任务：必要时拆分为步骤

输出格式要求：
- 如果原 prompt 简单，优化后保持简洁但更清晰
- 如果原 prompt 复杂，添加结构化标记和详细说明
- 始终用中文回复（因为用户用中文提问）
"""


def optimize_prompt(
    user_prompt: str,
    task_type: Literal["general", "code", "writing", "analysis", "creative"] = "general",
    extra_requirements: str = "",
) -> str:
    """
    优化用户输入的 prompt

    Args:
        user_prompt: 用户原始 prompt
        task_type: 任务类型
        extra_requirements: 额外要求

    Returns:
        优化后的 prompt
    """
    llm = create_ollama_chat(temperature=0.3)

    task_instructions = {
        "general": "这是一个通用问题，请按最佳实践优化",
        "code": "这是编程任务，需要明确编程语言、代码风格、注释要求等",
        "writing": "这是写作任务，需要明确文体、风格、受众、长度等",
        "analysis": "这是分析任务，需要明确分析框架、输出结构、关键点等",
        "creative": "这是创意任务，需要明确创意方向、风格、限制条件等",
    }

    messages = [
        SystemMessage(content=OPTIMIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""请优化以下 prompt：

【原始 prompt】
{user_prompt}

【任务类型】
{task_type}

【任务说明】
{task_instructions.get(task_type, task_instructions["general"])}

【额外要求】
{extra_requirements if extra_requirements else "无"}

请直接输出优化后的 prompt，不要添加解释。"""
        ),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def create_prompt_chain():
    """创建可复用的 prompt 优化链"""
    llm = create_ollama_chat(temperature=0.3)

    chain = (
        {
            "user_prompt": RunnablePassthrough(),
            "task_type": RunnablePassthrough(),
        }
        | SystemMessage(template=OPTIMIZER_SYSTEM_PROMPT)
        | llm
        | StrOutputParser()
    )

    return chain


def evaluate_prompt_quality(prompt: str) -> dict:
    """
    评估 prompt 的质量并给出改进建议

    Args:
        prompt: 待评估的 prompt

    Returns:
        评估结果
    """
    llm = create_ollama_chat(temperature=0.3)

    messages = [
        SystemMessage(
            content="你是一个 Prompt 评估专家，请评估以下 prompt 的质量并给出改进建议。"
        ),
        HumanMessage(
            content=f"""请评估这个 prompt 的质量，并从以下维度打分（1-10）：

{prompt}

请按以下格式回复：
1. 完整性评分：X/10
2. 清晰度评分：X/10
3. 具体性评分：X/10
4. 总体评价：XXX
5. 改进建议：XXX
"""
        ),
    ]

    response = llm.invoke(messages)
    return {"evaluation": response.content.strip()}


if __name__ == "__main__":
    # 测试
    result = optimize_prompt("写一首关于春天的诗")
    print("优化后的 prompt：")
    print(result)
