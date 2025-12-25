# learn_llm
4 周学习计划（概览）

第1周：理论与基础：掌握概率基础、线性代数、优化、NLP 基本概念、Transformer 原理。

目标：理解注意力机制与自注意力的数学直觉。
练习：实现简单的注意力机制（NumPy）；阅读并笔记《Attention Is All You Need》核心段落。
推荐资源：斯坦福 CS224n 片段、Distill.pub 注意力可视化文章。
第2周：工具与工程：学习 PyTorch 基础、Hugging Face Transformers 用法、数据管线。

目标：能用 HF 加载预训练模型并做推理。
练习：用 transformers 做一次端到端文本生成示例；准备小型数据集并写 DataLoader。
推荐资源：Hugging Face 官方教程、PyTorch 入门教程。
第3周：微调与数据工程（实战）：掌握传统微调、LoRA/PEFT 方法、评估指标与数据增强。

目标：完成对小模型的微调（本地或云端），并能评估效果。
练习：用示例数据做微调；比较全量微调与 LoRA 的性能与成本。
推荐资源：PEFT 文档、示例脚本、权重与训练技巧博文。
第4周：部署、优化与项目：API 封装、推理优化（量化/蒸馏）、监控与成本估算。

目标：把微调模型包装成可调用的服务并部署（本地/云）。
练习：用 FastAPI 或 Gradio 搭建演示；尝试 8-bit 量化或 ONNX 导出。
推荐资源：FastAPI + Transformers 示例、模型压缩教程。
持续项（贯穿与后续）：每周安排 1-2 篇论文阅读（如 RLHF、指令微调、多模态），并在第4周完成一个小型 Capstone 项目用于展示。

## 可执行笔记本
已添加一个包含 4 周学习内容的可执行 Jupyter 笔记本：
- learn_llm.ipynb — 包含 Week1..Week4 的可运行示例（NumPy 注意力、HF 推理、微调示例、FastAPI 部署代码）。

运行建议：
1. 创建虚拟环境并安装依赖：
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
2. 在 Jupyter 中打开并按单元格顺序运行 learn_llm.ipynb。