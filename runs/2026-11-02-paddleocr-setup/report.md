# PaddleOCR / PP-StructureV3 能力验证报告（2026-11-02）

## 1. 环境

- host venv：`~/ai4ellm/.venv-paddle`（Python 3.12）
- `paddlepaddle-gpu==3.4.0`（官方 cu130 wheel）
- `paddleocr==3.7.0`、`paddlex[ocr]==3.7.2`
- GPU2/3：RTX 5090，`capability (12,0)`，CUDA runtime 13.0，驱动 13.3

原始验证输出：

```text
paddle 3.4.0
compiled_with_cuda True
device_count 1
device_name NVIDIA GeForce RTX 5090
capability (12, 0)
PaddlePaddle works well on 1 GPU.
paddleocr 3.7.0
```

## 2. 纯 OCR（PP-OCRv6 medium）

`Power System Voltage Stability` p50（2509×4039，200 DPI）：

```text
OCR_RESULT {"device":"gpu","elapsed_s":1.74,"lines":51,"chars":1676,...}
```

20 页批量（150 DPI，包含模型加载）：

```text
{"done": 20, "last_page": 69, "pages_per_s": 0.522}
{"pages_done": 20, "elapsed_s": 38.4}
```

- 稳态约 **1.9–2.0 秒/页**；改用 `PP-OCRv6_small` 没有明显提升（40.3s/20页），
  瓶颈在渲染/检测而不是识别模型。
- 约 29 万扫描页 → 单卡约 160 小时，双卡约 80 小时；需要按领域子集/扫描件优先级控制。

## 3. PP-StructureV3（表格/公式/图区）

测试页：`Power System Voltage Stability` p50（含公式与图）

```text
STRUCTURE_RESULT {"elapsed_s": 21.65, ...,
  "formula_res_count": 6, "parsing_res_count": 7, ...}
```

- 公式 LaTeX 示例：`P=0.25+0.75n^{2}GV^{2}`、`n=1.1`、`n=1`
- 图区裁图：`page0/imgs/img_in_chart_box_134_391_2218_1885.jpg`
- Markdown 中包含图片与行内 LaTeX。

测试页：ASHRAE Handbook p43（真实表格）

```text
STRUCTURE_RESULT {"elapsed_s": 24.85, ..., "table_html_count": 1, "parsing_res_count": 24}
```

- `table_res_list[0].pred_html` 输出完整 `<table>` HTML（含 colspan/单元格文本）。
- 结论：**公式 LaTeX、表格 HTML、图区裁图都可提取**；但约 20–25 秒/页，
  只适合“按需结构识别”，不能全量 193 万页跑。

## 4. Docker 镜像

- `containers/ocr/Dockerfile` → `ai4ellm-ocr:paddleocr3.7`（33.4GB）
- 基于 `aigc-base:torch213-cu134`，独立 `/venv-paddle`，不污染训练环境。
- GPU3 容器实测：

```text
{"paddle":"3.4.0","compiled_with_cuda":true,"device_count":1}
OCR_RESULT {"device":"gpu","elapsed_s":7.92,"lines":51,"chars":1676,...}
```

与 host 输出完全一致。

## 5. 建议策略

1. 扫描件先跑纯 OCR（~2s/页），拿到文本；
2. 只对含公式/表格/图的页面（可用 PP-DocLayout 先判定）跑 PP-StructureV3；
3. 公式/表格/图区按页缓存，避免重复计算；
4. 图表裁图后续交给本地 Qwen3.8-27B 多模态生成描述（GPU2/3，不碰生产）。

## 6. 依赖锁

- host：`requirements-corpus.lock`
- OCR 镜像：`containers/ocr/requirements.lock`
