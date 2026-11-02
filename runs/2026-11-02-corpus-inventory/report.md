# `~/books` PDF 全量清单报告（2026-11-02）

目的：在不移动/重命名原件的前提下，建立 sha256/页数/元数据清单，为全量抽取与 OCR 估算提供依据。

## 1. 命令

```bash
.venv/bin/python scripts/08_pdf_inventory.py
```

- 输入：`/home/ai4e-hermes/books`（只读）
- 输出：`data/corpus-inventory/manifest.jsonl`、`duplicates.jsonl`

## 2. 原始输出

```text
scan /home/ai4e-hermes/books: 4877 pdfs
...
4877/4877  1553s  140.0GB  dups=27
done: 4877 pdfs, 27 duplicates, 1553s
manifest: /home/ai4e-hermes/ai4ellm/data/corpus-inventory/manifest.jsonl
duplicates: /home/ai4e-hermes/ai4ellm/data/corpus-inventory/duplicates.jsonl
```

统计：

```text
books 4877
pages_total 1930859
pages_avg 395
pages_max 2639
size_gb 140.0
duplicates 27
encrypted 0
metadata_errors 0
```

## 3. 结论与用途

- 4,877 个 PDF、**193 万页**、140GB；按 20 样本的 2.5M tokens/0.94GB 外推，
  全量约 **3.7 亿 tokens**（未计扫描件低文本量）。
- 27 个 sha256 完全重复文件；全量抽取可跳过或后续按 hash 去重。
- 原件保持原位，清单作为“已处理/待 OCR/重复”的唯一索引。
- OCR 成本取决于 `needs_ocr` 的实际页数：若 15% 的书需要 OCR（约 730 本、29 万页），
  按 1.5 秒/页、单卡约 **120 GPU 小时**，双卡约 **60 GPU 小时**；
  实际值以全量抽取后的 `needs_ocr.jsonl` 为准。
