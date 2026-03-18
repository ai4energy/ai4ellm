# AI4LLM 语料库构建工具完整文档

适用于Qwen3.5-9B等大语言模型的预训练和监督微调语料库构建工具。

---

## 目录

1. [项目概述](#1-项目概述)
2. [安装指南](#2-安装指南)
3. [快速开始](#3-快速开始)
4. [预训练语料处理](#4-预训练语料处理)
5. [SFT监督微调语料](#5-sft监督微调语料)
6. [PDF文本提取与OCR](#6-pdf文本提取与ocr)
7. [API调用详解](#7-api调用详解)
8. [配置文件说明](#8-配置文件说明)
9. [输出格式规范](#9-输出格式规范)
10. [与HuggingFace训练集成](#10-与huggingface训练集成)
11. [常见问题](#11-常见问题)

---

## 1. 项目概述

### 1.1 功能特点

本工具提供完整的语料库构建流水线，支持：

- **预训练语料**: GitHub代码、PDF文档、Web数据的采集、清洗、去重
- **SFT语料**: 通过API调用自动生成问答对、多轮对话、摘要等训练数据
- **PDF OCR**: 支持本地OCR（MinerU）和云端OCR API（DeepSeek、通义千问）
- **多API支持**: DeepSeek、OpenAI、Claude、通义千问、智谱AI
- **格式转换**: 支持多种输入格式转换为TRL训练格式

### 1.2 项目结构

```
ai4ellm/
├── pretrain/                        # 预训练模块
│   ├── data-collection/             # 数据采集（预留）
│   ├── text-extraction/             # 文本提取
│   │   └── pdf_extractor.py         # PDF提取（MinerU/PyMuPDF/OCR API）
│   ├── cleaning/                    # 数据清洗
│   │   └── text_cleaner.py          # 文本清洗器
│   ├── deduplication/               # 去重处理
│   │   └── semantic_dedup.py        # 精确/MinHash/语义去重
│   └── format/                      # 格式转换
│       └── convert_to_pretrain.py   # 预训练格式转换
│
├── sft/                             # 监督微调模块
│   ├── qa-construction/             # QA构建（预留）
│   ├── data-generation/             # 数据生成
│   │   ├── generate_qa.py           # API问答对生成
│   │   └── generate_multi_type.py   # 多类型数据生成
│   ├── format-conversion/           # 格式转换
│   │   └── convert_to_sft.py        # SFT格式转换
│   └── validation/                  # 数据验证
│       └── validate_dataset.py      # 数据集验证
│
├── tools/                           # 通用工具
│   ├── dataset/
│   │   └── dataset_utils.py         # 数据集划分/合并/采样
│   └── utils/                       # 通用工具（预留）
│
├── scripts/                         # 一键脚本
│   ├── run_pretrain_pipeline.py     # 预训练流水线
│   └── run_sft_pipeline.py          # SFT流水线
│
└── configs/                         # 配置文件
    ├── pretrain.yaml                # 预训练配置
    └── sft.yaml                     # SFT配置
```

---

## 2. 安装指南

### 2.1 基础依赖

```bash
# 克隆项目
cd /home/mingtaoli/ai4ellm

# 安装基础依赖
pip install tqdm pyyaml requests aiohttp loguru
```

### 2.2 预训练模块依赖

```bash
# PDF提取 - MinerU（推荐，高质量OCR）
pip install magic-pdf[full]

# PDF提取 - PyMuPDF（快速文本提取）
pip install PyMuPDF

# PDF提取 - pdfplumber（表格友好）
pip install pdfplumber

# 去重 - MinHash LSH
pip install datasketch

# 去重 - 语义去重
pip install sentence-transformers

# 编码检测（代码语料）
pip install charset-mnbvc
```

### 2.3 SFT模块依赖

```bash
# API调用（必需）
pip install openai

# 可选API客户端
pip install anthropic      # Claude
pip install dashscope      # 通义千问
pip install zhipuai        # 智谱AI
```

### 2.4 完整安装

```bash
pip install -r requirements.txt
```

---

## 3. 快速开始

### 3.1 预训练语料处理

```bash
# 一键运行预训练流水线
python scripts/run_pretrain_pipeline.py --config configs/pretrain.yaml

# 或分步执行
# 1. PDF提取
python -m pretrain.text-extraction.pdf_extractor \
    --input ./data/pdfs --output ./output/extracted

# 2. 数据清洗
python -m pretrain.cleaning.text_cleaner \
    --input ./output/extracted --output ./output/cleaned

# 3. 去重
python -m pretrain.deduplication.semantic_dedup \
    --input ./output/cleaned/merged.jsonl \
    --output ./output/deduped.jsonl \
    --method hybrid

# 4. 格式转换
python -m pretrain.format.convert_to_pretrain \
    --input ./output/deduped.jsonl \
    --output ./output/pretrain.jsonl
```

### 3.2 SFT语料生成

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your-api-key"

# 使用API生成问答对
python -m sft.data-generation.generate_qa \
    --api deepseek \
    --input ./data/contexts.txt \
    --output ./output/qa_pairs.jsonl \
    --num-questions 3 \
    --domain energy

# 格式转换
python -m sft.format-conversion.convert_to_sft \
    --input ./output/qa_pairs.jsonl \
    --output ./output/sft_data.jsonl \
    --format messages

# 数据验证
python -m sft.validation.validate_dataset \
    --input ./output/sft_data.jsonl

# 数据集划分
python -m tools.dataset.dataset_utils split \
    --input ./output/sft_data.jsonl \
    --output ./dataset \
    --test-size 0.1 --val-size 0.1

# 或一键运行SFT流水线
python scripts/run_sft_pipeline.py --config configs/sft.yaml
```

---

## 4. 预训练语料处理

### 4.1 数据采集

#### 4.1.1 GitHub代码采集

```bash
# 准备GitHub Token
echo "ghp_xxxx" > github_tokens.txt

# 爬取仓库元数据
python code-to-corpus/auto-metedata.py \
    -p github \
    --github_tokens_file github_tokens.txt \
    --start 1 --end 1000000

# 提取仓库列表
python code-to-corpus/repos_list.py

# 下载仓库
python code-to-corpus/github_downloader.py
```

#### 4.1.2 PDF文档采集

将PDF文件放入指定目录，或使用数据采集脚本下载。

### 4.2 文本提取

#### 4.2.1 PDF提取方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **MinerU** | 高质量、支持OCR、保留格式 | 需要GPU、安装复杂 | 学术论文、扫描件 |
| **PyMuPDF** | 快速、轻量 | 不支持OCR | 文本型PDF |
| **OCR API** | 无需本地GPU、高准确率 | 需要付费API | 扫描件、复杂布局 |

#### 4.2.2 使用MinerU提取

```bash
# 安装MinerU
pip install magic-pdf[full]

# 运行提取
python -m pretrain.text-extraction.pdf_extractor \
    --input ./data/pdfs \
    --output ./output/extracted \
    --method mineru
```

#### 4.2.3 使用OCR API提取

```bash
# DeepSeek Vision OCR
python -m pretrain.text-extraction.pdf_extractor \
    --input ./data/pdfs \
    --output ./output/extracted \
    --ocr-api-key $DEEPSEEK_API_KEY \
    --ocr-api-type deepseek

# 通义千问 Vision OCR
python -m pretrain.text-extraction.pdf_extractor \
    --input ./data/pdfs \
    --output ./output/extracted \
    --ocr-api-key $DASHSCOPE_API_KEY \
    --ocr-api-type qwen
```

### 4.3 数据清洗

```bash
# 基础清洗
python -m pretrain.cleaning.text_cleaner \
    --input ./output/extracted \
    --output ./output/cleaned \
    --type text

# Markdown清洗
python -m pretrain.cleaning.text_cleaner \
    --input ./output/markdown \
    --output ./output/cleaned \
    --type markdown
```

#### 清洗功能

- 去除HTML标签
- 去除URL和邮箱
- 去除乱码和特殊字符
- 去除页眉页脚和页码
- 规范化空白字符
- 去除空行和短行

### 4.4 数据去重

#### 4.4.1 去重方法对比

| 方法 | 原理 | 速度 | 准确率 | 适用规模 |
|------|------|------|--------|----------|
| **精确去重** | MD5哈希 | 最快 | 100% | 任意规模 |
| **MinHash** | 近似去重 | 快 | 高 | 大规模 |
| **语义去重** | 向量相似度 | 慢 | 最高 | 中小规模 |
| **混合去重** | 精确+语义 | 中等 | 最高 | 推荐 |

```bash
# 精确去重（最快）
python -m pretrain.deduplication.semantic_dedup \
    --input ./data.jsonl --output ./output.jsonl \
    --method exact

# MinHash去重（大规模）
python -m pretrain.deduplication.semantic_dedup \
    --input ./data.jsonl --output ./output.jsonl \
    --method minhash --threshold 0.9

# 语义去重（最准确）
python -m pretrain.deduplication.semantic_dedup \
    --input ./data.jsonl --output ./output.jsonl \
    --method semantic --threshold 0.9

# 混合去重（推荐）
python -m pretrain.deduplication.semantic_dedup \
    --input ./data.jsonl --output ./output.jsonl \
    --method hybrid --threshold 0.9
```

### 4.5 格式转换

```bash
# 代码转预训练格式
python -m pretrain.format.convert_to_pretrain \
    --input ./code.jsonl --output ./pretrain.jsonl \
    --type code

# PDF转预训练格式
python -m pretrain.format.convert_to_pretrain \
    --input ./pdf.jsonl --output ./pretrain.jsonl \
    --type pdf

# 合并多个文件
python -m pretrain.format.convert_to_pretrain \
    --input ./code.jsonl ./pdf.jsonl ./web.jsonl \
    --output ./pretrain.jsonl \
    --type merge
```

---

## 5. SFT监督微调语料

### 5.1 支持的API

| API | 模型 | 特点 | 推荐场景 |
|-----|------|------|----------|
| **DeepSeek** | deepseek-chat | 性价比高、中文友好 | 通用问答 |
| **OpenAI** | gpt-4o-mini | 质量高、稳定 | 英文内容 |
| **Claude** | claude-3-haiku | 长文本、高质量 | 学术内容 |
| **通义千问** | qwen-turbo | 中文优秀 | 中文问答 |
| **智谱AI** | glm-4-flash | 快速、便宜 | 大规模生成 |

### 5.2 生成问答对

```bash
# 基础用法
python -m sft.data-generation.generate_qa \
    --api deepseek \
    --input ./contexts.txt \
    --output ./qa_pairs.jsonl \
    --num-questions 3

# 指定领域
python -m sft.data-generation.generate_qa \
    --api deepseek \
    --input ./energy_contexts.txt \
    --output ./energy_qa.jsonl \
    --num-questions 5 \
    --domain energy

# 批量生成
python -m sft.data-generation.generate_qa \
    --api deepseek \
    --input ./contexts.jsonl \
    --output ./qa_pairs.jsonl \
    --num-questions 3 \
    --concurrency 10
```

### 5.3 领域类型

| 领域 | 说明 | 适用场景 |
|------|------|----------|
| `general` | 通用领域 | 一般问答 |
| `technical` | 技术领域 | 技术文档、API说明 |
| `academic` | 学术领域 | 论文、研究报告 |
| `code` | 编程领域 | 代码解释、编程问题 |
| `energy` | 能源领域 | 能源系统、能源政策 |

### 5.4 多类型数据生成

除了问答对，还支持生成：

- **多轮对话**
- **文本摘要**
- **翻译数据**
- **代码解释**
- **指令数据**

```python
# 示例：生成多轮对话
from sft.data_generation.generate_multi_type import ConversationGenerator

generator = ConversationGenerator(client, conversation_type="energy")
conversation = await generator.generate_multi_turn(
    topic="太阳能发电系统设计",
    num_turns=3
)
```

### 5.5 格式转换

```bash
# QA格式转Messages
python -m sft.format-conversion.convert_to_sft \
    --input ./qa_pairs.jsonl \
    --output ./sft_data.jsonl \
    --input-format qa \
    --output-format messages

# Alpaca格式转Messages
python -m sft.format-conversion.convert_to_sft \
    --input ./alpaca_data.jsonl \
    --output ./sft_data.jsonl \
    --input-format alpaca

# 自动检测格式
python -m sft.format-conversion.convert_to_sft \
    --input ./mixed_data.jsonl \
    --output ./sft_data.jsonl \
    --input-format auto
```

### 5.6 数据验证

```bash
# 验证数据集
python -m sft.validation.validate_dataset \
    --input ./sft_data.jsonl \
    --format sft

# 严格模式
python -m sft.validation.validate_dataset \
    --input ./sft_data.jsonl \
    --strict

# 生成报告
python -m sft.validation.validate_dataset \
    --input ./sft_data.jsonl \
    --output ./validation_report.txt
```

### 5.7 数据集划分

```bash
# 划分训练集和测试集
python -m tools.dataset.dataset_utils split \
    --input ./sft_data.jsonl \
    --output ./dataset \
    --test-size 0.1

# 划分训练集、验证集和测试集
python -m tools.dataset.dataset_utils split \
    --input ./sft_data.jsonl \
    --output ./dataset \
    --test-size 0.1 --val-size 0.1

# 合并多个数据集
python -m tools.dataset.dataset_utils merge \
    --input ./data1 ./data2 ./data3 \
    --output ./merged.jsonl \
    --deduplicate

# 采样
python -m tools.dataset.dataset_utils sample \
    --input ./large_data.jsonl \
    --output ./sampled.jsonl \
    --ratio 0.1

# 分析数据集
python -m tools.dataset.dataset_utils analyze \
    --input ./sft_data.jsonl
```

---

## 6. PDF文本提取与OCR

### 6.1 MinerU本地OCR（推荐）

MinerU是目前最优秀的开源PDF提取工具，支持：

- 自动判断文本模式和OCR模式
- 保留文档结构和格式
- 支持表格识别
- 输出Markdown格式

```bash
# 安装
pip install magic-pdf[full]

# 使用
python -m pretrain.text-extraction.pdf_extractor \
    --input ./pdfs --output ./output --method mineru
```

### 6.2 OCR API（扫描件推荐）

对于扫描件PDF或复杂布局，推荐使用OCR API：

#### DeepSeek Vision OCR

```bash
export DEEPSEEK_API_KEY="sk-xxxx"

python -m pretrain.text-extraction.pdf_extractor \
    --input ./scanned_pdfs \
    --output ./output \
    --ocr-api-key $DEEPSEEK_API_KEY \
    --ocr-api-type deepseek
```

#### 通义千问 Vision OCR

```bash
export DASHSCOPE_API_KEY="sk-xxxx"

python -m pretrain.text-extraction.pdf_extractor \
    --input ./scanned_pdfs \
    --output ./output \
    --ocr-api-key $DASHSCOPE_API_KEY \
    --ocr-api-type qwen
```

### 6.3 自动降级提取

系统支持自动降级：先用高质量方法，失败则尝试其他方法。

```bash
python -m pretrain.text-extraction.pdf_extractor \
    --input ./pdfs --output ./output --method auto
```

提取顺序：MinerU → PyMuPDF → OCR API

---

## 7. API调用详解

### 7.1 配置API密钥

**方式一：环境变量**

```bash
# DeepSeek
export DEEPSEEK_API_KEY="sk-xxxx"

# OpenAI
export OPENAI_API_KEY="sk-xxxx"

# Claude
export ANTHROPIC_API_KEY="sk-xxxx"

# 通义千问
export DASHSCOPE_API_KEY="sk-xxxx"

# 智谱AI
export ZHIPU_API_KEY="xxxx"
```

**方式二：配置文件**

```yaml
# configs/sft.yaml
api:
  type: "deepseek"
  api_key: "${DEEPSEEK_API_KEY}"
  model: "deepseek-chat"
  max_tokens: 2048
  temperature: 0.7
  concurrency: 5
```

### 7.2 API参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_tokens` | 最大生成token数 | 2048 |
| `temperature` | 生成温度（0-1） | 0.7 |
| `top_p` | 核采样参数 | 0.9 |
| `max_retries` | 最大重试次数 | 3 |
| `concurrency` | 并发请求数 | 5 |

### 7.3 并发控制

```bash
# 低并发（稳定但慢）
--concurrency 3

# 中并发（推荐）
--concurrency 5

# 高并发（快但可能限流）
--concurrency 10
```

### 7.4 错误处理

系统自动处理以下错误：

- **API限流**: 自动等待并重试
- **网络错误**: 自动重试最多3次
- **无效响应**: 记录错误并跳过

---

## 8. 配置文件说明

### 8.1 预训练配置 (configs/pretrain.yaml)

```yaml
# 数据采集
collection:
  github:
    enabled: true
    tokens_file: "github_tokens.txt"
    start_id: 1
    end_id: 1000000

# 文本提取
extraction:
  pdf:
    enabled: true
    method: "auto"          # mineru, pymupdf, auto
    use_ocr_api: false      # 是否使用OCR API
    ocr_api_type: "deepseek"
    ocr_api_key: "${OCR_API_KEY}"
    input_dir: "data/pdfs"
    output_dir: "output/pdf_extracted"

# 数据清洗
cleaning:
  enabled: true
  remove_html: true
  remove_urls: true
  remove_garbled: true
  normalize_whitespace: true
  min_line_length: 10

# 去重
deduplication:
  enabled: true
  method: "hybrid"          # exact, minhash, semantic, hybrid
  threshold: 0.9

# 格式转换
format:
  min_length: 100
  max_length: 1000000
  include_metadata: true

# 数据集划分
split:
  enabled: true
  test_size: 0.01           # 预训练测试集可以很小
  val_size: 0.01
  seed: 42
```

### 8.2 SFT配置 (configs/sft.yaml)

```yaml
# API配置
api:
  type: "deepseek"          # deepseek, openai, claude, qwen, zhipu
  api_key: "${API_KEY}"
  model: ""                 # 留空使用默认模型
  max_tokens: 2048
  temperature: 0.7
  max_retries: 3
  concurrency: 5

# 数据生成
generation:
  qa:
    enabled: true
    domain: "energy"        # general, technical, academic, code, energy
    num_questions_per_text: 3
    input_file: "data/contexts.txt"
    output_file: "output/sft_qa.jsonl"

  conversation:
    enabled: false
    type: "qa"
    num_turns: 3

  summary:
    enabled: false
    style: "brief"          # brief, detailed, bullet

# 格式转换
format:
  input_format: "auto"
  output_format: "messages"
  min_length: 10
  max_length: 50000

# 数据验证
validation:
  enabled: true
  strict: false

# 数据集划分
split:
  enabled: true
  test_size: 0.1
  val_size: 0.1
  seed: 42
```

---

## 9. 输出格式规范

### 9.1 预训练格式

```json
{"text": "这是一段预训练文本内容..."}
```

### 9.2 SFT格式

**Messages格式（推荐）**:

```json
{
  "messages": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回答"}
  ]
}
```

**多轮对话**:

```json
{
  "messages": [
    {"role": "user", "content": "第一个问题"},
    {"role": "assistant", "content": "第一个回答"},
    {"role": "user", "content": "追问"},
    {"role": "assistant", "content": "后续回答"}
  ]
}
```

**带上下文**:

```json
{
  "messages": [
    {"role": "user", "content": "背景信息：\n...\n\n问题：..."},
    {"role": "assistant", "content": "回答"}
  ],
  "context": "原始上下文..."
}
```

**Prompt-Completion格式**:

```json
{
  "prompt": "用户问题或指令",
  "completion": "期望的回答"
}
```

### 9.3 DPO格式（偏好学习）

```json
{
  "prompt": "用户问题",
  "chosen": "优选回答",
  "rejected": "拒绝回答"
}
```

---

## 10. 与HuggingFace训练集成

### 10.1 预训练

```python
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# 加载处理好的预训练数据
dataset = load_dataset("json", data_files="output/pretrain_dataset/train.jsonl")

# 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 数据预处理
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=2048)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="./pretrain_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()
```

### 10.2 SFT微调

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# 加载处理好的SFT数据
dataset = load_dataset("json", data_files="output/sft_dataset/train.jsonl")

# 创建训练器
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./sft_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        push_to_hub=True,
        hub_model_id="username/your-model",
    ),
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    ),
)

# 开始训练
trainer.train()

# 推送到Hub
trainer.push_to_hub()
```

### 10.3 使用HuggingFace Jobs

```python
# 提交到HuggingFace Jobs进行云端训练
hf_jobs("uv", {
    "script": """
# /// script
# dependencies = ["trl>=0.12.0", "peft>=0.7.0"]
# ///

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

dataset = load_dataset("json", data_files="train.jsonl")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="output",
        push_to_hub=True,
        hub_model_id="username/your-model",
    ),
    peft_config=LoraConfig(r=16, lora_alpha=32),
)

trainer.train()
""",
    "flavor": "a10g-large",
    "timeout": "4h",
    "secrets": {"HF_TOKEN": "$HF_TOKEN"}
})
```

---

## 11. 常见问题

### Q1: API调用频率限制怎么办？

```yaml
# 降低并发数
api:
  concurrency: 3  # 默认5

# 增加重试等待
api:
  retry_delay: 2.0  # 默认1.0秒
```

### Q2: PDF提取质量不好？

1. **文本型PDF**: 使用MinerU或PyMuPDF
2. **扫描件PDF**: 使用OCR API
3. **复杂布局**: 使用MinerU的OCR模式

```bash
# 强制使用OCR
python -m pretrain.text-extraction.pdf_extractor \
    --input ./pdfs --output ./output --method mineru
```

### Q3: 去重太慢？

```bash
# 大数据集使用精确去重或MinHash
python -m pretrain.deduplication.semantic_dedup \
    --input data.jsonl --output output.jsonl \
    --method exact  # 最快

# 或使用MinHash
python -m pretrain.deduplication.semantic_dedup \
    --input data.jsonl --output output.jsonl \
    --method minhash
```

### Q4: 生成的问答质量不高？

1. 提供更详细的上下文
2. 选择合适的领域类型
3. 调整温度参数

```yaml
generation:
  qa:
    domain: "energy"      # 选择匹配的领域
    num_questions_per_text: 5  # 生成更多问题

api:
  temperature: 0.5  # 降低温度提高一致性
```

### Q5: 如何处理多语言数据？

系统会自动检测文本语言，建议：
- 中文问答：使用DeepSeek或通义千问
- 英文问答：使用OpenAI或Claude
- 混合语言：根据主要语言选择API

### Q6: 数据集划分比例如何选择？

| 场景 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| 预训练 | 98% | 1% | 1% |
| SFT微调 | 80% | 10% | 10% |
| 数据少 | 90% | 5% | 5% |

### Q7: 内存不足怎么办？

```bash
# 减小批处理大小
python -m pretrain.deduplication.semantic_dedup \
    --input data.jsonl --output output.jsonl \
    --batch-size 64  # 默认256

# 使用精确去重代替语义去重
python -m pretrain.deduplication.semantic_dedup \
    --input data.jsonl --output output.jsonl \
    --method exact
```

---

## 附录

### A. 命令速查表

```bash
# === 预训练 ===
# PDF提取
python -m pretrain.text-extraction.pdf_extractor -i ./pdfs -o ./output

# 清洗
python -m pretrain.cleaning.text_cleaner -i ./raw -o ./cleaned

# 去重
python -m pretrain.deduplication.semantic_dedup -i ./data.jsonl -o ./out.jsonl -m hybrid

# 格式转换
python -m pretrain.format.convert_to_pretrain -i ./data.jsonl -o ./pretrain.jsonl

# === SFT ===
# 生成问答
python -m sft.data-generation.generate_qa --api deepseek -i ./ctx.txt -o ./qa.jsonl

# 格式转换
python -m sft.format-conversion.convert_to_sft -i ./qa.jsonl -o ./sft.jsonl

# 验证
python -m sft.validation.validate_dataset -i ./sft.jsonl

# 划分
python -m tools.dataset.dataset_utils split -i ./data.jsonl -o ./dataset

# === 流水线 ===
python scripts/run_pretrain_pipeline.py --config configs/pretrain.yaml
python scripts/run_sft_pipeline.py --config configs/sft.yaml
```

### B. 环境变量

```bash
# API密钥
export DEEPSEEK_API_KEY="sk-xxxx"
export OPENAI_API_KEY="sk-xxxx"
export ANTHROPIC_API_KEY="sk-xxxx"
export DASHSCOPE_API_KEY="sk-xxxx"
export ZHIPU_API_KEY="xxxx"

# GitHub Token
export GITHUB_TOKEN="ghp_xxxx"
```

### C. 参考链接

- [HuggingFace Skills](https://github.com/huggingface/skills) - 训练方案参考
- [TRL Documentation](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning
- [MinerU](https://github.com/opendatalab/MinerU) - PDF提取工具
- [MNBVC](https://github.com/esbatmop/MNBVC) - 中文语料集
- [DeepSeek API](https://platform.deepseek.com/) - DeepSeek API文档
- [OpenAI API](https://platform.openai.com/) - OpenAI API文档