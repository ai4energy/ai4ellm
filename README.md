# AI4LLM - 大语言模型语料库构建工具

适用于Qwen3.5-9B等大语言模型的**预训练**和**监督微调**语料库构建工具。

> **完整文档**: [docs/README.md](docs/README.md)

---

## 容器化部署

### 海光 DCU (ROCm)
```bash
docker build -f Dockerfile.rocm -t ai4ellm:rocm .
docker run --rm --device=/dev/dri -v ./data:/app/data -v ./output:/app/output ai4ellm:rocm
```

### NVIDIA GPU (CUDA)
```bash
docker build -f Dockerfile.cuda -t ai4ellm:cuda .
docker run --rm --gpus all -v ./data:/app/data -v ./output:/app/output ai4ellm:cuda
```

详见 [docs/09容器化部署.md](docs/09容器化部署.md)

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
