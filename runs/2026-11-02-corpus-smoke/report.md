# `corpus_pipeline` 20-PDF 冒烟报告（2026-11-02）

目的：验证 `~/ai4ellm-remote/corpus_pipeline` 在 `~/books` 上的抽取质量、扫描件比例、
chunk/字符/token 规模与耗时，为全量 4,877 个 PDF 做决策。

## 1. 环境与命令

- Python venv：`~/ai4ellm/.venv`
- PyMuPDF 2.0 / pdfplumber 0.11.10 / loguru / pypdf / tqdm
- 样本：20 个 PDF（10 小 <10MB、5 中 10–80MB、5 大 ≥80MB），共 943,510,420 B（0.94GB）
- 命令：

```bash
cd ~/ai4ellm-remote/corpus_pipeline
~/ai4ellm/.venv/bin/python main.py \
  --input  ~/ai4ellm/runs/2026-11-02-corpus-smoke/input \
  --output ~/ai4ellm/runs/2026-11-02-corpus-smoke/output
```

## 2. 原始统计报告

```text
============================================================
  语料库构建流水线 统计报告
============================================================
  扫描文件数:            20
  Office 转换成功:       0
  PDF 提取成功:          19
  TXT/MD 读取成功:       0
  清洗章节数:            19
  切分 Chunk 数:         20980
  chunk 级规则去重后:    20956
  语义去重后记录数:      0
  富 JSONL 输出记录数:   41912
  预训练 JSONL 记录数:   20432
  错误数:                1
  总耗时: 38.2 秒
============================================================
错误详情:
  - PDF 提取失败: .../Power System Voltage Stability-Carson W. Taylor(1993).pdf
    | pdfplumber: 文本过少
```

## 3. 提取质量（关键）

按“每页字符数”检查，19 个“成功”里有 2 个实际是扫描件：

| 字符数 | 页数 | 字符/页 | 文件 |
|---:|---:|---:|---|
| 1,680 | 406 | 4.1 | Chemical Reactors-H. Scott Fogler |
| 1,633 | 358 | 4.6 | DISTILLATION DESIGN AND CONTROL USING ASPEN |
| — | 122MB | 提取失败 | Power System Voltage Stability-Carson W. Taylor |

其余 17 个均在 598–3690 字符/页。**样本扫描件/低质量比例约 3/20 = 15%**，
全量 4,877 个 PDF 约 700 个可能需要 OCR。

质量分（20,432 条 chunk）：

```text
p1:  0.4149   p5: 0.4668   p10: 0.4904   p25: 0.5345
p50: 0.5921   p75: 0.6572  p90: 0.7167   p95: 0.7440   p99: 0.7920
below_0.3: 51   below_0.4: 150   below_0.5: 2585   total: 20432
```

默认 `quality.filter_enabled: false`，即只打分不删除。

## 4. 规模外推

```json
{
  "sample_pdf_bytes": 943510420,
  "sample_pdf_gb": 0.94,
  "full_pdf_bytes": 140010070568,
  "full_pdf_gb": 140.0,
  "byte_factor": 148.4,
  "sample_chunks": 20432,
  "sample_chars": 9832258,
  "sample_tokens_est": 2500574,
  "full_chunks_est": 3031960,
  "full_chars_est": 1459035435,
  "full_tokens_est": 371066998,
  "note": "token estimate from 2000-chunk sample with Qwen3.8-27B tokenizer, 3.932 chars/token"
}
```

- 20 个 PDF → 9.83M 字符 ≈ **2.50M tokens**，20,432 个 chunk。
- 全量按体积外推：~1.46B 字符 ≈ **3.71 亿 tokens**，~303 万 chunk。
- 纯 CPU 抽取耗时：38.2 秒 / 0.94GB → 140GB 约 **1.6 小时**（不含 OCR）。
- 2×5090 QLoRA 约 300–400 tok/s，全量 CPT 1 epoch ≈ **11–15 天**，不现实。
- RAG：303 万 chunk，按 50–100 chunk/s 做 embedding 约 **8–17 小时**；
  1024 维 fp32 向量约 **12GB**，可放 `/data`。

## 5. 发现的问题

1. **扫描件比例约 15%**：全量约 700 本需要 MinerU/OCR；这是最大成本项。
2. **chunk 长度未真正封顶 500 字符**：
   - >500 字符：9,203 条；
   - >1000：57 条；>2000：14 条；>8000：3 条；最大 **25,121** 字符。
   - 长 chunk 多为索引/目录/参考文献，RAG 前应重新切分或按结构过滤。
3. **目录/索引噪声**：`skip_sections` 只能过滤标题明确的“目录”，大量书末 Index 仍会进入语料。
4. **元数据页数为 0**：2 个文件 `page_count` 缺失；不影响文本，但影响质量报表。
5. **PyMuPDF 2.0 可用但 `fitz` API 已弃用**（`FutureWarning`）；生产环境建议钉 1.28.2。

## 6. 工具链结论

- **PyMuPDF**：2.0 能跑；但 `corpus_pipeline` 只用到 `get_text()`，生产建议钉
  `pymupdf==1.28.2`，避免 2.0 的 layout/onnx 依赖和 API 变化。
- **pdfplumber 0.11.10**：可保留。
- **PyPDF2**：代码未使用，建议移除，需要时用 `pypdf==6.21.4`。
- **MinerU**：不要原地升级。仓库绑定的是 magic-pdf 1.3.12 API；PyPI 最新是
  MinerU 4.1.3（依赖 docvortex / mineru-llama-cpp / gradio，API 重写）。
  建议在独立 conda/容器里对比 magic-pdf 1.3.12 与 MinerU 4.1.3 在
  **上述 3 个扫描 PDF** 上的质量/耗时/显存，再决定。
- 全量前把实测版本写入 `requirements-corpus.lock`。

## 7. 建议

1. **不做全量 CPT**：3.71 亿 token 在 2×5090 上不划算；先把 PDF 用于 RAG，
   SFT 用 Execution-verified 的 ESML 数据。
2. 全量跑 `corpus_pipeline` 前先加两道后处理：
   - 按 `quality_score >= 0.4` 且 `noise_ratio <= 0.1` 过滤；
   - 对 >800 字符的 chunk 重新切分，过滤 Index/参考文献页。
3. 对约 700 个扫描件单独建 `needs_ocr.jsonl`，用 MinerU 补；原件不移动、不重命名。
4. 建 `manifest.jsonl`（sha256/路径/页数/字符数/质量/状态），作为“已处理”的唯一判据。
