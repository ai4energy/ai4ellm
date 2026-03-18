# AI4LLM - 大语言模型语料库构建工具

适用于Qwen3.5-9B等大语言模型的**预训练**和**监督微调**语料库构建工具。

> **完整文档**: [docs/README.md](docs/README.md)

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **预训练语料** | PDF提取、代码采集、数据清洗、去重、格式转换 |
| **SFT语料** | API自动生成问答对、多轮对话、摘要、代码解释 |
| **PDF OCR** | 支持本地OCR(MinerU)和云端OCR API(DeepSeek/通义千问) |
| **多API支持** | DeepSeek、OpenAI、Claude、通义千问、智谱AI |

---

## 项目结构

```
ai4ellm/
├── pretrain/                        # 预训练模块
│   ├── text-extraction/pdf_extractor.py    # PDF提取
│   ├── cleaning/text_cleaner.py            # 数据清洗
│   ├── deduplication/semantic_dedup.py     # 去重
│   └── format/convert_to_pretrain.py       # 格式转换
│
├── sft/                             # SFT模块
│   ├── data-generation/generate_qa.py      # API生成问答
│   ├── format-conversion/convert_to_sft.py # 格式转换
│   └── validation/validate_dataset.py      # 数据验证
│
├── tools/dataset/dataset_utils.py   # 数据集工具
├── scripts/                         # 一键流水线
├── configs/                         # 配置文件
└── docs/README.md                   # 完整文档
```

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 预训练语料

```bash
# 一键流水线
python scripts/run_pretrain_pipeline.py --config configs/pretrain.yaml

# 或分步执行
python -m pretrain.text-extraction.pdf_extractor -i ./pdfs -o ./output
python -m pretrain.cleaning.text_cleaner -i ./raw -o ./cleaned
python -m pretrain.deduplication.semantic_dedup -i ./data.jsonl -o ./out.jsonl -m hybrid
python -m pretrain.format.convert_to_pretrain -i ./data.jsonl -o ./pretrain.jsonl
```

### SFT语料生成

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your-api-key"

# 生成问答对
python -m sft.data-generation.generate_qa \
    --api deepseek \
    --input ./contexts.txt \
    --output ./qa_pairs.jsonl \
    --num-questions 3 \
    --domain energy

# 格式转换和验证
python -m sft.format-conversion.convert_to_sft -i ./qa.jsonl -o ./sft.jsonl
python -m sft.validation.validate_dataset -i ./sft.jsonl
python -m tools.dataset.dataset_utils split -i ./sft.jsonl -o ./dataset

# 或一键流水线
python scripts/run_sft_pipeline.py --config configs/sft.yaml
```

---

## PDF OCR支持

### 本地OCR（MinerU）- 推荐

```bash
pip install magic-pdf[full]
python -m pretrain.text-extraction.pdf_extractor -i ./pdfs -o ./output --method mineru
```

### 云端OCR API

```bash
# DeepSeek Vision OCR
python -m pretrain.text-extraction.pdf_extractor \
    -i ./scanned_pdfs -o ./output \
    --ocr-api-key $DEEPSEEK_API_KEY \
    --ocr-api-type deepseek

# 通义千问 Vision OCR
python -m pretrain.text-extraction.pdf_extractor \
    -i ./scanned_pdfs -o ./output \
    --ocr-api-key $DASHSCOPE_API_KEY \
    --ocr-api-type qwen
```

---

## 支持的API

| API | 使用场景 | 设置环境变量 |
|-----|----------|-------------|
| DeepSeek | 中文问答、性价比高 | `DEEPSEEK_API_KEY` |
| OpenAI | 英文内容、稳定 | `OPENAI_API_KEY` |
| Claude | 学术内容、长文本 | `ANTHROPIC_API_KEY` |
| 通义千问 | 中文优秀 | `DASHSCOPE_API_KEY` |
| 智谱AI | 快速、便宜 | `ZHIPU_API_KEY` |

---

## 输出格式

### 预训练格式
```json
{"text": "预训练文本内容..."}
```

### SFT格式
```json
{
  "messages": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回答"}
  ]
}
```

---

## 与HuggingFace训练集成

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# 加载数据
dataset = load_dataset("json", data_files="output/sft_dataset/train.jsonl")

# SFT训练
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B",
    train_dataset=dataset["train"],
    args=SFTConfig(
        output_dir="./sft_model",
        num_train_epochs=3,
        push_to_hub=True,
        hub_model_id="username/your-model",
    ),
    peft_config=LoraConfig(r=16, lora_alpha=32),
)
trainer.train()
```

---

## 命令速查

```bash
# 预训练
python -m pretrain.text-extraction.pdf_extractor -i ./pdfs -o ./out
python -m pretrain.cleaning.text_cleaner -i ./raw -o ./cleaned
python -m pretrain.deduplication.semantic_dedup -i ./in.jsonl -o ./out.jsonl -m hybrid
python -m pretrain.format.convert_to_pretrain -i ./in.jsonl -o ./out.jsonl

# SFT
python -m sft.data-generation.generate_qa --api deepseek -i ./ctx.txt -o ./qa.jsonl
python -m sft.format-conversion.convert_to_sft -i ./qa.jsonl -o ./sft.jsonl
python -m sft.validation.validate_dataset -i ./sft.jsonl
python -m tools.dataset.dataset_utils split -i ./data.jsonl -o ./dataset

# 流水线
python scripts/run_pretrain_pipeline.py --config configs/pretrain.yaml
python scripts/run_sft_pipeline.py --config configs/sft.yaml
```

---

## 更多文档

- **完整文档**: [docs/README.md](docs/README.md)
- **预训练配置**: [configs/pretrain.yaml](configs/pretrain.yaml)
- **SFT配置**: [configs/sft.yaml](configs/sft.yaml)

---

## 参考资料

- [HuggingFace Skills](https://github.com/huggingface/skills) - 训练方案参考
- [TRL Documentation](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning
- [MinerU](https://github.com/opendatalab/MinerU) - PDF提取工具
- [DeepSeek API](https://platform.deepseek.com/) - DeepSeek API文档