# 纯文本语料全量抽取报告（2026-11-02）

## 1. 范围与命令

- 输入：`/home/ai4e-hermes/books-processed`（3,544 本；1 本 316MB 病态扫描件已隔离到 `books-problem/`）
- 输出：`/home/ai4e-hermes/corpus/full-20261102-renamed`
- 命令：

```bash
cd ~/ai4ellm-remote/corpus_pipeline
~/ai4ellm/.venv/bin/python main.py --config ~/ai4ellm/configs/corpus_full.yaml
```

- 本轮为**纯文本**：不含公式 LaTeX、表格 HTML、图片裁切；扫描件只标记，不 OCR。

## 2. 原始统计（pipeline 报告）

```text
扫描文件数:            3544
PDF 提取成功:          3327
清洗章节数:            3458
切分 Chunk 数:         3934381
质量过滤:              3934381 → 3782263（过滤 152118）
chunk 级规则去重后:    3414607
富 JSONL 输出记录数:   3414607
预训练 JSONL 记录数:   3414607
错误数:                217
总耗时:                6881.8 秒
```

产物：

```text
extracted_text  2.7G   3465 个 txt
cleaned_text    5.5G
chunks          3.7G
jsonl           3.7G
pretrain.jsonl  4.0G   3,414,607 条
```

## 3. 最终 pretrain 统计

```text
records 3414607
chars 2393332883
tokens_est ~608,680,794（3.932 字符/token）
cjk_ratio 0.0036
len min/avg/max 105 / 700 / 1912
over_1000 2435（0.07%）
quality n/min/avg/max 3414607 / 0.4000 / 0.5936 / 0.9360
```

- 质量过滤已生效：最低分 0.4，平均 0.594。
- 99.93% 的 chunk ≤1000 字符；极少数因章节标题前缀超过 1000。

## 4. 失败与 OCR 清单

`needs_ocr.jsonl`：

```text
flagged_books: 449
flagged_pages: 177967
all_books: 3544
est_gpu_hours_at_2s_page_1gpu: 98.9
est_gpu_hours_at_2s_page_2gpu: 49.4
```

原因分布（一本可多因）：

```text
chars_per_page<100 : 434
no_text_file       : 217
broken_encoding    : 15
```

- 217 个提取失败主要是扫描件、超大 PDF 与字体编码损坏；绝大多数应走 OCR。
- 15 个“有文本但坏编码”文件（控制字符/Caesar 乱码）已单独标记。
- 修复项：pdfplumber 对 >80MB 或 >300 页的 PDF 直接跳过并建议 OCR；抽取重启跳过已有文本。

## 5. 质量结论

- 正文文本可用，适合 RAG 的文本检索、问答与 SFT 上下文。
- **公式/表格/图片未结构化**：公式会散架（例 `W mg 1920 N`），表格退化为文本行。
- 页眉/水印（如 EasyEngineering、PROPRIETARY MATERIAL）需在清洗后做文档级去重。
- 规模：约 24 亿字符 / 6.1 亿 tokens；2×5090 上全量 CPT 不现实，确认走 RAG + 执行验证 SFT。

## 6. 后续

1. 按 `needs_ocr.jsonl` 优先级跑 OCR（PaddleOCR GPU，约 2 秒/页；双卡约 49 小时）。
2. 选 100–300 本核心教材跑 PP-StructureV3（公式/表格/图区）；图区交本地 Qwen3.8-27B 描述。
3. 用过滤后的 `pretrain.jsonl` 建 RAG 索引；SFT 仍以 ESML 执行验证数据为主。
