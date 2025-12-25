"""
分词技术详解 - NLP 基础系列

本模块涵盖大模型中常用的分词算法和实践。

核心知识点：
1. 词分词 (Word Tokenization) - 最基本的分词方式
2. 子词分词 (Subword Tokenization) - 平衡词表大小和表达能力
3. BPE (Byte Pair Encoding) - GPT/RoBERTa 使用
4. WordPiece - BERT 使用
5. SentencePiece - 多语言模型支持
6. Tokenizer 使用实践 - Hugging Face

为什么需要分词？
- 文本需要转换为数字才能被模型处理
- 词表大小直接影响模型参数量和速度
- 好的分词策略能平衡：
  - 词表大小（内存效率）
  - 表示能力（语义完整性）
  - 序列长度（计算效率）
"""

import os
import json
import base64
from collections import Counter, defaultdict


# ============================================================
# 1. 基础分词概念
# ============================================================

def explain_tokenization_basics():
    """
    分词基础概念
    
    分词是将文本切分成模型可处理的单元的过程。
    
    分词策略对比：
    ┌─────────────────┬─────────────┬─────────────┬──────────────┐
    │ 策略            │ 词表大小     │ OOV 处理    │ 适用场景      │
    ├─────────────────┼─────────────┼─────────────┼──────────────┤
    │ 词分词          │ 大           │ ❌ 无法处理 │ 特定领域      │
    │ 字符分词        │ 很小         │ ✓ 支持      │ 中文、日文    │
    │ 子词分词        │ 中等         │ ✓ 支持      │ 通用          │
    └─────────────────┴─────────────┴─────────────┴──────────────┘
    
    OOV (Out-of-Vocabulary) 问题：
    - 训练集外的词无法处理
    - 传统方法：<unk> token
    - 子词方法：拆分成已知子词
    """
    
    print("=" * 70)
    print("1. 分词基础概念")
    print("=" * 70)
    
    # 示例1：不同分词策略对比
    print("\n示例1：不同分词策略对比")
    print("-" * 50)
    
    text = "大语言模型正在快速发展"
    
    # 词分词
    word_tokens = text.split()
    print(f"词分词: {word_tokens}")
    print(f"  词数: {len(word_tokens)}")
    
    # 字符分词
    char_tokens = list(text)
    print(f"\n字符分词: {char_tokens}")
    print(f"  词数: {len(char_tokens)}")
    
    # 假设的子词分词（模拟）
    # 实际中需要训练 tokenizer
    subword_tokens = ["大", "语言", "模型", "正在", "快速", "发展"]
    print(f"\n子词分词: {subword_tokens}")
    print(f"  词数: {len(subword_tokens)}")
    
    # 示例2：OOV 问题演示
    print("\n示例2：OOV (Out-of-Vocabulary) 问题")
    print("-" * 50)
    
    # 模拟一个简单的词表
    vocab = {"大", "语言", "模型", "正在", "快速", "发展"}
    
    test_texts = [
        "大语言模型",          # 全部在词表中
        "大语言模型很强大",     # "很"、"强大" OOV
        "Transformer 很强大",  # "Transformer" OOV
    ]
    
    print(f"词表: {vocab}")
    print(f"\n{'文本':<20} {'分词结果':<30} {'OOV 词'}")
    print("-" * 70)
    
    for text in test_texts:
        tokens = text.split()
        oov = [t for t in tokens if t not in vocab]
        oov_str = ", ".join(oov) if oov else "无"
        
        # 替换 OOV
        masked = [t if t in vocab else "<unk>" for t in tokens]
        
        print(f"{text:<20} {masked:<30} {oov_str}")


# ============================================================
# 2. BPE (Byte Pair Encoding)
# ============================================================

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) 分词器实现
    
    BPE 是一种数据压缩算法，后被应用于分词。
    
    算法步骤：
    1. 初始化词表为所有字符
    2. 统计相邻字符对的出现频率
    3. 合并频率最高的字符对，加入词表
    4. 重复步骤 2-3 直到达到目标词表大小
    
    优点：
    - 平衡词表大小和表示能力
    - 能处理任意新词（通过子词组合）
    - GPT、RoBERTa 等模型使用
    """
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
    
    def get_frequencies(self, text):
        """统计字符频率"""
        chars = list(text)
        freq = Counter(chars)
        return freq
    
    def get_pair_frequencies(self, text):
        """统计字符对频率"""
        chars = list(text)
        pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
        return Counter(pairs)
    
    def train(self, texts, max_vocab_size=None):
        """
        训练 BPE 分词器
        
        参数：
            texts: 训练文本列表
            max_vocab_size: 最大词表大小
        """
        if max_vocab_size:
            self.vocab_size = max_vocab_size
        
        # 初始化词表为所有字符
        self.vocab = {}
        for text in texts:
            for char in list(text):
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)
        
        # 合并字符对
        for merge_step in range(self.vocab_size - len(self.vocab)):
            # 统计所有文本中的字符对频率
            pair_freq = defaultdict(int)
            for text in texts:
                chars = list(text)
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    pair_freq[pair] += 1
            
            if not pair_freq:
                break
            
            # 找到最常见的字符对
            best_pair = max(pair_freq, key=pair_freq.get)
            
            # 合并字符对
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            # 更新所有文本中的字符对
            for i, text in enumerate(texts):
                texts[i] = self._merge_pair(text, best_pair)
            
            if merge_step % 100 == 0:
                print(f"  Merge step {merge_step}: '{best_pair}' -> '{new_token}'")
        
        print(f"最终词表大小: {len(self.vocab)}")
    
    def _merge_pair(self, text, pair):
        """合并文本中的指定字符对"""
        result = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and (text[i], text[i+1]) == pair:
                result.append(''.join(pair))
                i += 2
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)
    
    def encode(self, text):
        """将文本编码为 token IDs"""
        # 首先按字符分割
        tokens = list(text)
        
        # 应用合并规则
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge:
                    new_tokens.append(''.join(merge))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 转换为 IDs
        return [self.vocab.get(t, self.vocab.get('<unk>')) for t in tokens]
    
    def decode(self, ids):
        """将 token IDs 解码为文本"""
        tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(i)] 
                  for i in ids if i in self.vocab.values()]
        return ''.join(tokens)


def explain_bpe():
    """
    BPE 详解与实现
    """
    
    print("\n" + "=" * 70)
    print("2. BPE (Byte Pair Encoding)")
    print("=" * 70)
    
    # 示例1：手动演示 BPE 过程
    print("\n示例1：BPE 合并过程演示")
    print("-" * 50)
    
    # 模拟 BPE 过程
    texts = ["低频词  低频词 高频词 高频词 高频词"]
    
    # 初始字符频率
    text = "低频词  低频词 高频词 高频词 高频词"
    print(f"原始文本: '{text}'")
    print(f"字符: {list(text)}")
    
    # 统计字符对
    chars = list(text.replace(" ", ""))
    pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
    pair_freq = Counter(pairs)
    
    print(f"\n初始字符对频率:")
    for pair, freq in pair_freq.most_common(5):
        print(f"  '{pair[0]}{pair[1]}': {freq}")
    
    # 模拟第一次合并（最高频的字符对）
    print(f"\n第一次合并: '词' 和 '词' 合并为 '词词'")
    text = text.replace("词词", "词词")  # 简化演示
    
    # 词表增长
    print(f"\n词表大小变化:")
    initial_chars = len(set(list(text.replace(" ", ""))))
    print(f"  初始字符数: {initial_chars}")
    print(f"  第一次合并后: {initial_chars + 1}")
    print(f"  第二次合并后: {initial_chars + 2}")
    print(f"  ...")
    print(f"  最终词表大小: ~{initial_chars + 10} (假设)")
    
    # 示例2：使用 BPE 分词器
    print("\n示例2：训练和测试 BPE 分词器")
    print("-" * 50)
    
    # 训练语料
    train_texts = [
        "深度 学习 是 人工 智能 的 重要 分支",
        "神经 网络 是 深度 学习 的 基础",
        "大 语言 模型 基于 Transformer 架构",
        "自然 语言 处理 涉及 文本 生成 和 理解",
        "机器 学习 算法 能 从 数据 中 学习 模式",
    ]
    
    # 训练 BPE
    tokenizer = BPETokenizer(vocab_size=200)
    print("训练 BPE 分词器...")
    tokenizer.train(train_texts, max_vocab_size=50)
    
    # 测试编码
    test_texts = [
        "深度学习",
        "大语言模型",
        "计算机视觉",
    ]
    
    print(f"\n编码测试:")
    for text in test_texts:
        ids = tokenizer.encode(text)
        print(f"  '{text}' -> {ids}")
    
    # 示例3：BPE 在实际模型中的应用
    print("\n示例3：BPE 在 GPT 中的应用")
    print("-" * 50)
    
    print("""
GPT 使用的 BPE 特点：
    
    1. 词表大小：50,257
       - 包含：26个英文字母 + 基础标点 + 常用单词 + 子词单元
    
    2. 训练语料：大规模网页文本
       - BookCorpus, WebText, Wikipedia 等
    
    3. 分词示例：
       "Transformer" -> ["Trans", "former"]
       "tokenization" -> ["token", "ization"]
       "LLMs" -> ["LL", "Ms"]
    
    4. 优点：
       - 有效控制词表大小
       - 能处理任意新词
       - 平衡序列长度和语义完整性
    """)


# ============================================================
# 3. WordPiece
# ============================================================

def explain_wordpiece():
    """
    WordPiece 分词详解
    
    WordPiece 是 BERT 使用的分词算法。
    
    与 BPE 的区别：
    - BPE：基于频率合并字符对
    - WordPiece：基于语言模型似然合并（考虑是否增加词的频率）
    
    算法步骤：
    1. 初始化词表为所有字符
    2. 对于每个候选字符对，计算合并后的语言模型似然
    3. 选择使似然增加最多的字符对
    4. 重复直到词表达到目标大小
    
    WordPiece 的合并准则：
    选择使 P(文本 | 新词表) 最大的合并
    即：选择能最大化 (频率 × 似然) 的字符对
    """
    
    print("\n" + "=" * 70)
    print("3. WordPiece (BERT 使用)")
    print("=" * 70)
    
    print("""
WordPiece vs BPE:

┌─────────────────────┬──────────────────────┬──────────────────────┐
│ 方面                │ BPE                  │ WordPiece            │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 合并准则            │ 最高频率             │ 最大语言模型似然     │
│ 目标函数            │ 最小化文本长度       │ 最大化语言模型似然   │
│ 典型应用            │ GPT, RoBERTa         │ BERT, DistilBERT     │
│ 词表大小            │ ~50K                 │ ~30K                 │
└─────────────────────┴──────────────────────┴──────────────────────┘

BERT WordPiece 分词特点：

    1. 词表大小：30,522
       - 包含：英文单词、子词单元
    
    2. 特殊 Token：
       - [PAD]: 填充
       - [UNK]: 未知词
       - [SEP]: 句子分隔
       - [CLS]: 分类标记
       - [MASK]: 掩码标记
    
    3. 分词示例：
       "playing" -> ["play", "ing"]
       "international" -> ["inter", "national"]
       "unhappiness" -> ["un", "happiness"]
    
    4. 特点：
       - 以单词为单位初始化
       - 更倾向于保留完整单词
       - 对形态丰富语言效果好
    """)


# ============================================================
# 4. SentencePiece
# ============================================================

def explain_sentencepiece():
    """
    SentencePiece 分词详解
    
    SentencePiece 是一个独立的分词库，支持多种分词算法。
    
    主要特点：
    1. 语言无关：基于原始文本（无需预处理）
    2. 支持多种算法：BPE、Unigram、Char、Word
    3. 直接处理原始文本：空格作为普通字符
    4. T5、Llama 使用
    
    与其他 Tokenizer 的区别：
    - 其他：需要预先分词（空格分词）
    - SentencePiece：直接处理原始文本
    """
    
    print("\n" + "=" * 70)
    print("4. SentencePiece (T5, Llama 使用)")
    print("=" * 70)
    
    print("""
SentencePiece 特点：

    1. 直接处理原始文本
       - 不需要预先进行空格分词
       - 将空格视为普通字符
       - 最终解码时添加空格
    
    2. 支持的分词算法：
       - BPE (Byte Pair Encoding)
       - Unigram Language Model
       - Char-level (字符级)
       - Word-level (词级)
    
    3. 特殊处理：
       - ▁ (下划线) 表示空格
       - 自动处理空白字符
       - 支持 Unicode
    
    4. 典型应用：
       - T5: SentencePiece BPE
       - Llama: SentencePiece BPE
       - mBERT: 多语言支持
    
    5. 分词示例：
       "Hello world" -> ["▁Hello", "▁world"]
       "大语言模型" -> ["▁大语言", "▁模型"]
    """)


# ============================================================
# 5. Hugging Face Tokenizer 使用
# ============================================================

def explain_hf_tokenizer():
    """
    Hugging Face Tokenizer 使用指南
    
    HF 提供了统一的 Tokenizer 接口。
    """
    
    print("\n" + "=" * 70)
    print("5. Hugging Face Tokenizer 使用")
    print("=" * 70)
    
    print("""
使用 Hugging Face Tokenizer：

    1. 加载预训练 Tokenizer：
    
       from transformers import AutoTokenizer
       
       tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
       # 或
       tokenizer = AutoTokenizer.from_pretrained("gpt2")
       # 或
       tokenizer = AutoTokenizer.from_pretrained("llama2")
    
    2. 基本操作：
    
       # 编码
       encoded = tokenizer("Hello, world!")
       # {'input_ids': [...], 'attention_mask': [...]}
       
       # 批量编码
       encoded = tokenizer(["Hello", "World"], padding=True)
       
       # 解码
       text = tokenizer.decode([101, 7592, 9999, 102])
    
    3. 特殊参数：
    
       - max_length: 最大序列长度
       - padding: 填充策略 (True, 'longest', 'max_length')
       - truncation: 截断策略
       - return_tensors: 返回格式 ('pt', 'tf', 'np')
    
    4. 获取词汇表：
    
       vocab = tokenizer.get_vocab()
       vocab_size = tokenizer.vocab_size
    
    5. 特殊 Token：
    
       tokenizer.bos_token  # 句子开始
       tokenizer.eos_token  # 句子结束
       tokenizer.pad_token  # 填充
       tokenizer.unk_token  # 未知词
    """)


# ============================================================
# 6. Tokenizer 实战
# ============================================================

def tokenizer_practice():
    """
    Tokenizer 实战示例
    """
    
    print("\n" + "=" * 70)
    print("6. Tokenizer 实战")
    print("=" * 70)
    
    try:
        from transformers import AutoTokenizer
        
        # 示例：使用 GPT-2 Tokenizer
        print("\n示例：GPT-2 Tokenizer 实践")
        print("-" * 50)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        texts = [
            "Hello, world!",
            "大语言模型正在快速发展。",
            "The quick brown fox jumps over the lazy dog.",
            "1234567890",
            "URL: https://example.com/path?query=value",
        ]
        
        print(f"词表大小: {tokenizer.vocab_size}")
        print(f"特殊 Token: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")
        print()
        
        print(f"{'文本':<45} {'Tokens':<35} {'长度'}")
        print("-" * 100)
        
        for text in texts:
            encoded = tokenizer(text)
            tokens = encoded['input_ids']
            token_str = str(tokens)[:32]
            
            # 解码验证
            decoded = tokenizer.decode(tokens)
            
            print(f"{text:<45} {token_str:<35} {len(tokens)}")
        
        # 示例2：特殊 Token 处理
        print("\n示例2：批量编码和填充")
        print("-" * 50)
        
        texts = ["Hello", "Hello world", "Hello world!"]
        encoded = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=10,
            return_tensors="np"
        )
        
        print(f"输入文本: {texts}")
        print(f"input_ids:\\n{encoded['input_ids']}")
        print(f"attention_mask:\\n{encoded['attention_mask']}")
        
        # 示例3：不同模型的 Tokenizer 对比
        print("\n示例3：不同模型的 Tokenization 对比")
        print("-" * 50)
        
        text = "Transformer is powerful."
        
        models = ["bert-base-uncased", "gpt2", "t5-small"]
        
        print(f"测试文本: '{text}'")
        print(f"\n{'模型':<25} {'词表大小':<12} {'Token 数':<10} {'Tokens'}")
        print("-" * 80)
        
        for model_name in models:
            try:
                tok = AutoTokenizer.from_pretrained(model_name)
                encoded = tok(text)
                tokens = encoded['input_ids']
                token_str = str(tokens)
                print(f"{model_name:<25} {tok.vocab_size:<12} {len(tokens):<10} {token_str[:40]}")
            except Exception as e:
                print(f"{model_name:<25} Error: {e}")
    
    except ImportError:
        print("\n未安装 transformers 库，跳过 HF Tokenizer 示例")
        print("请运行: pip install transformers")
        print("\n使用内置 BPE 分词器进行演示...")
        
        # 使用内置 BPE
        train_texts = [
            "hello world",
            "transformer is powerful",
            "natural language processing",
            "machine learning is fun",
            "deep learning neural networks",
        ]
        
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(train_texts, max_vocab_size=100)
        
        test_texts = ["hello", "world", "transformer", "learning"]
        print(f"\nBPE 词表大小: {len(tokenizer.vocab)}")
        print(f"\n{'文本':<15} {'Token IDs':<30} {'Token数'}")
        print("-" * 60)
        
        for text in test_texts:
            ids = tokenizer.encode(text)
            print(f"{text:<15} {str(ids):<30} {len(ids)}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("分词技术详解 - NLP 基础系列")
    print("=" * 70)
    
    # 运行各个示例
    explain_tokenization_basics()
    explain_bpe()
    explain_wordpiece()
    explain_sentencepiece()
    explain_hf_tokenizer()
    tokenizer_practice()
    
    print("\n" + "=" * 70)
    print("分词技术学习总结")
    print("=" * 70)
    print("""
核心概念速查表：

┌─────────────────────┬────────────────┬─────────────────────────┐
│ 分词方法            │ 典型模型       │ 特点                    │
├─────────────────────┼────────────────┼─────────────────────────┤
│ BPE                 │ GPT, RoBERTa   │ 基于频率合并，高效      │
├─────────────────────┼────────────────┼─────────────────────────┤
│ WordPiece           │ BERT           │ 基于似然，单词导向      │
├─────────────────────┼────────────────┼─────────────────────────┤
│ SentencePiece       │ T5, Llama      │ 语言无关，直接处理文本  │
└─────────────────────┴────────────────┴─────────────────────────┘

分词最佳实践：

    1. 英文文本：
       - 使用 BPE 或 WordPiece
       - 词表大小：30K-50K
    
    2. 中文文本：
       - 字符级或词级
       - 可结合 BPE
    
    3. 多语言：
       - 使用 SentencePiece
       - 共享词表或独立词表
    
    4. 特殊场景：
       - 代码：特殊 Token 处理关键字
       - 数学：特殊 Token 处理公式
       - 多模态：特殊 Token 处理图像

与 LLM 的关系：
- Tokenization 影响词表大小和序列长度
- 好的分词策略能提高训练效率
- 不同模型使用不同的分词方法
- 需要与模型匹配使用

下一步学习：
- 词嵌入：如何将 Token 转换为向量
- 位置编码：如何处理序列顺序
- 注意力机制：如何建模依赖关系
    """)
