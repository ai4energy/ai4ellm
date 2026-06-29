# 语料库构建流水线

从 PDF/Word/PPT/TXT/Markdown 提取文本，章节感知清洗、4 维质量评分、结构化切块、
规则/语义去重，生成富格式语料 + 预训练 JSONL。

> 这是 `ai4ellm` 的语料库子项目，融合了原 `corpus_pipeline_代码版` 的成熟流水线
> 骨架与当前 repo 的 PyMuPDF→MinerU 多引擎提取器。仅预训练语料，SFT/训练后续阶段。

## 快速开始

```bash
cd corpus_pipeline
pip install -r requirements.txt          # 核心：PyMuPDF + tqdm + pyyaml + loguru

# 冒烟（单类目）
python main.py --input ../origin-files-organized/102工程热力学 --output ./output_smoke

# 全量（907 PDF，支持断点续跑）
python main.py --input ../origin-files-organized --output ./output
```

## 流水线步骤

| 步骤 | 说明 | 默认 |
|------|------|------|
| 1 扫描 | 递归扫描输入目录 | 开 |
| 2 提取 | PDF(PyMuPDF→MinerU→pdfplumber) / Office(LibreOffice) / TXT | 开 |
| 3 清洗 | 章节感知 markdown 清洗（skip_sections / 水印 / 乱码 / 参考文献） | 开 |
| 4 切块 | 结构化切分（按章节，带 source 追踪） | 开 |
| 5 评分 | 4 维质量评分（readability/coherence/info_density/noise_ratio） | 开 |
| 6 规则去重 | chunk 级归一化 hash 去重 | 开 |
| 7 语义去重 | Sentence-BERT chunk 级去重 | 关 |
| 8 导出 | 富 JSONL/JSON（含 quality_score + 溯源字段） | 开 |
| 9 预训练 | 富 chunk → 预训练 JSONL（standard/rich 格式） | 开 |

## 输出格式

### 富 JSONL（`output/jsonl/corpus.jsonl`）
```json
{"section": "第一章", "full_path": "第一章 > 1.1", "content": "...",
 "source_file": "xxx.pdf", "source_type": "pdf", "chunk_index": 0,
 "char_count": 423, "quality_score": 0.85,
 "quality": {"readability": 0.9, "coherence": 0.8, "info_density": 0.85, "noise_ratio": 0.02}}
```

### 预训练 JSONL（`output/pretrain.jsonl`，rich 格式）
```json
{"text": "# 第一章\n\n正文内容...",
 "meta": {"source_file": "xxx.pdf", "source_type": "pdf", "section": "第一章",
          "category": "102工程热力学", "chunk_index": 0, "quality_score": 0.85}}
```

## 配置

编辑 `config.yaml` 调整：路径、提取引擎、清洗规则、chunk 大小、质量阈值、去重、格式。
关键：`pretrain.format` 选 `standard`（仅 `{"text":...}`）或 `rich`（带 meta 溯源）。

## 断点续跑

提取阶段（PDF/Office/TXT）自动记录进度到 `output/processed_files.json`，重跑跳过已处理文件
（size+hash+output 三重校验）。清洗及以后幂等覆写。

## 与训练的衔接

预训练 JSONL 直接喂 TRL/HF：
```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="corpus_pipeline/output/pretrain.jsonl")
```
