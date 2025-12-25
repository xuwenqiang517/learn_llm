# Week 2：工程实践（Engineering Practice）

本目录涵盖大模型工程实践的核心内容，系统讲解PyTorch框架基础、高效数据处理管道设计，以及Hugging Face生态系统集成方法。通过本模块学习，学员将掌握大模型训练和部署所必需工程技能。

## 目录结构

| 文件 | 功能说明 |
|------|----------|
| `pytorch_basics.py` | PyTorch张量操作、自动求导机制、神经网络模块构建、训练循环实现、GPU加速与混合精度训练基础 |
| `data_pipeline.py` | 自定义Dataset实现、动态padding处理、内存优化策略、流式数据加载、分布式采样、Hugging Face Datasets集成 |
| `huggingface_integration.py` | AutoModel与AutoTokenizer自动加载、Pipeline推理接口、Trainer API高效微调、模型保存与部署最佳实践 |
| `README.md` | 本模块学习指南与文档说明 |

## 学习目标

### 1．PyTorch基础

掌握PyTorch框架的核心概念与操作方法，为后续大模型开发奠定坚实基础。具体包括：张量的多维度创建方式与形状变换操作；自动求导（Autograd）机制的原理与高级用法；使用nn.Module构建复杂神经网络架构；各类损失函数与优化器的选择与配置；完整训练循环的实现与调试技巧；GPU内存管理与混合精度训练入门。

### 2．数据管道

深入理解大模型训练中的数据处理最佳实践，解决实际工程中的效率与内存问题。具体包括：Dataset与DataLoader的正确实现方式；动态padding与attention mask的高效处理；大规模数据的流式加载与内存优化；多种采样策略（加权采样、类别平衡采样）实现；分布式训练中的数据分片与同步机制；Hugging Face Datasets库的便捷集成方法。

### 3．Hugging Face集成

熟练运用工业级工具链进行模型开发与部署。具体包括：AutoModel、AutoTokenizer的自动加载与配置；各类Pipeline的零代码推理用法；TrainingArguments与Trainer API的高级配置；模型量化、蒸馏与推理优化技术；完整的微调流程实现与最佳实践。

## 先修知识

- 完成Week 1数学基础学习（线性代数、概率论、信息论、优化理论）
- 扎实的Python编程能力
- 基本的深度学习概念理解（神经网络、前向传播、反向传播）
- 了解GPU计算基本原理更佳

## 建议学习顺序

第一阶段学习`pytorch_basics.py`，通过大量示例掌握PyTorch框架基础，包括张量运算、自动求导、模型定义、训练循环等核心概念。第二阶段学习`data_pipeline.py`，理解高效数据处理的工程实践，包括自定义数据集、内存优化、分布式加载等关键技能。第三阶段学习`huggingface_integration.py`，使用工业级工具完成模型加载、推理、微调与部署全流程。每个阶段都应结合代码实践，深入理解各模块的设计思想与实现细节。

## 关键概念速查

### PyTorch核心概念

| 概念 | 说明 |
|------|------|
| Tensor | PyTorch核心数据结构，类似于NumPy数组但支持GPU加速与自动求导，是神经网络运算的基本单元 |
| Autograd | 自动微分系统，通过动态构建计算图自动计算梯度，是PyTorch实现反向传播的核心机制 |
| nn.Module | 神经网络模块的基类，提供参数管理、模型保存、设备迁移、训练模式切换等丰富功能 |
| Optimizer | 优化器，负责根据梯度更新模型参数，SGD、Adam、AdamW等是常用优化算法实现 |
| GradScaler | 梯度缩放器，用于混合精度训练，防止FP16梯度下溢 |

### 数据处理核心概念

| 概念 | 说明 |
|------|------|
| Dataset | 抽象数据集类，定义数据获取逻辑，实现__len__与__getitem__方法即可自定义数据集 |
| DataLoader | 数据加载器，提供批量加载、随机打乱、多进程加载、内存固定（pin_memory）等功能 |
| Collate Function | 批处理函数，负责将单个样本组合为批次，处理变长序列的padding与对齐问题 |
| DistributedSampler | 分布式采样器，在多GPU训练中确保各进程处理不同的数据分片 |
| IterableDataset | 可迭代数据集，适用于流式加载大规模数据或从数据库实时读取场景 |

### Hugging Face核心概念

| 概念 | 说明 |
|------|------|
| AutoModel/AutoTokenizer | 自动加载接口，根据预训练模型名称或路径自动推断模型架构并加载对应权重与分词器 |
| Transformers Trainer | 高效训练工具类，内置训练循环、日志记录、评估指标、分布式训练、混合精度等支持 |
| TrainingArguments | 训练参数配置类，涵盖批大小、学习率、训练轮数、日志间隔、保存策略等全部训练配置 |
| Pipeline | 端到端推理接口，支持文本分类、命名实体识别、问答、文本生成等常见任务的零代码使用 |
| Config | 模型配置文件，包含隐藏层大小、注意力头数、层数、词表大小等模型架构与超参数信息 |
| BitsAndBytesConfig | 量化配置类，支持8位与4位量化，可显著减少模型显存占用并加速推理 |

## 实践提示

关于GPU使用，确保安装CUDA版本的PyTorch，使用`.to('cuda')`将模型和数据移动到GPU，使用`.cuda.is_available()`检查GPU可用性。关于内存管理，大模型训练时善用梯度累积（gradient_accumulation_steps）模拟更大batch，使用梯度检查点（gradient checkpointing）以计算换空间，使用混合精度训练（fp16）减少显存占用并加速计算。关于调试技巧，遇到梯度异常时可使用`torch.set_anomaly_enabled(True)`检测NaN和Inf，使用`torch.autograd.set_detect_anomaly(True)`定位梯度问题。关于性能优化，启用`pin_memory=True`加速CPU到GPU的数据传输，根据CPU核心数合理设置`num_workers`，使用`persistent_workers=True`避免worker频繁重启。

## 下一步

完成Week 2学习后，将进入Week 3的微调实战模块，学习全量微调、LoRA、PEFT等高效微调技术，以及模型评估与指标计算方法。这些技能将帮助学员在实际项目中高效定制预训练大模型。
