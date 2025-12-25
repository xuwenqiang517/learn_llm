# Week 3：微调实战（Fine-tuning Practice）

本目录涵盖大模型微调的核心技术与实践方法，包括全量微调、高效参数微调（LoRA/PEFT）、模型评估指标等内容。通过本模块学习，学员将掌握针对预训练大模型进行高效定制的完整技术栈。

## 目录结构

| 文件 | 功能说明 |
|------|----------|
| `full_finetuning.py` | 全量微调完整实现，涵盖数据准备、训练配置、模型保存与推理全流程 |
| `lora_peft_finetuning.py` | LoRA、QLoRA、PEFT高效微调技术实现，显著降低显存占用与训练成本 |
| `evaluation_metrics.py` | 大语言模型评估指标实现，包括困惑度、BLEU、ROUGE、精确率/召回率/F1等 |
| `README.md` | 本模块学习指南与文档说明 |

## 学习目标

### 1．全量微调

深入理解并实现针对预训练模型的全量参数微调。具体包括：全量微调的适用场景与技术原理分析；针对不同任务（文本分类、问答、生成）的微调策略设计；高效训练配置优化，包括学习率调度、梯度累积、混合精度等；训练过程中的监控与调试技巧；模型保存与部署的最佳实践。

### 2．高效参数微调

掌握以最小参数量实现高质量模型微调的先进技术。具体包括：LoRA（Low-Rank Adaptation）低秩适配原理与实现；QLoRA量化低秩适配，在消费级GPU上微调大模型；PEFT（Parameter-Efficient Fine-Tuning）库的高效使用方法；Prefix Tuning、Prompt Tuning、Adapter等多样化PEFT技术对比；多模态场景下的PEFT应用实践。

### 3．模型评估

系统掌握大语言模型的评估方法与指标计算。具体包括：困惑度（Perplexity）评估语言模型质量；生成任务评估指标BLEU、ROUGE、METEOR实现与应用；分类任务评估指标精确率、召回率、F1、Accuracy计算；语义相似度评估与Embedding-based指标；多任务综合评估策略与基准测试。

## 先修知识

- 完成Week 1数学基础学习（线性代数、优化理论）
- 完成Week 2工程实践学习（PyTorch、Hugging Face）
- 扎实的深度学习模型训练经验
- 理解Transformer架构与注意力机制

## 建议学习顺序

第一阶段学习`full_finetuning.py`，通过完整代码实现理解传统全量微调的完整流程，包括数据准备、模型配置、训练循环、评估保存等环节。第二阶段学习`lora_peft_finetuning.py`，深入掌握LoRA、PEFT等高效微调技术的原理与实现，理解如何以极低资源成本完成高质量模型定制。第三阶段学习`evaluation_metrics.py`，掌握大语言模型多维度评估方法，能够科学全面地评价模型性能。

## 关键概念速查

### 微调技术核心概念

| 概念 | 说明 |
|------|------|
| 全量微调 | 更新预训练模型的所有参数，适用于任务与预训练差异大或追求最高精度的场景，计算成本高 |
| LoRA | Low-Rank Adaptation，通过低秩矩阵分解实现参数高效微调，仅训练注入的适配器权重 |
| QLoRA | 量化LoRA，结合4位量化与LoRA技术，可在消费级GPU上微调65B参数模型 |
| PEFT | Parameter-Efficient Fine-Tuning，统一接口封装多种高效微调技术，包括LoRA、Prefix Tuning等 |
| Adapter | 在Transformer层间插入小型可训练模块，保持原模型参数不变 |
| Prefix Tuning | 在每层Transformer前添加可训练的virtual tokens，保留模型原始能力 |

### 评估指标核心概念

| 概念 | 说明 |
|------|------|
| Perplexity | 困惑度，衡量语言模型对测试数据的预测能力，值越低表示模型越好 |
| BLEU | Bilingual Evaluation Understudy，基于n-gram重叠的翻译评估指标，取值0到1 |
| ROUGE | Recall-Oriented Understudy for Gisting Evaluation，基于召回率的摘要评估指标族 |
| F1 Score | 精确率与召回率的调和平均，综合衡量分类模型性能 |
| Embedding Similarity | 基于语义 Embedding 的相似度评估，适用于开放域生成质量评估 |

## 实践提示

关于全量微调，建议在小规模数据上先验证训练流程，使用较小的预训练模型（如BERT-base、RoBERTa-base）进行实验。关于LoRA微调，选择合适的目标模块（通常是Attention层）可以显著影响效果，r维度和alpha超参数需要根据任务调优。关于评估指标，应根据任务类型选择合适的评估体系，避免单一指标的局限性。关于显存优化，大模型训练推荐使用梯度累积和混合精度训练，PEFT技术可大幅降低显存需求。

## 下一步

完成Week 3学习后，将进入Week 4的部署优化模块，学习模型量化、蒸馏、API封装、分布式部署等生产级技术，为模型的实际应用做好全面准备。
