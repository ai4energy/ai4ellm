#!/usr/bin/env bash
# ============================================================
# AI4ELLM Corpus Pipeline — Docker 入口脚本
#
# 职责：
#   1. 部署 magic-pdf.json 到 /root/magic-pdf.json
#   2. 创建必要目录
#   3. 下载 MinerU 模型（PDF-Extract-Kit-1.0, ~3GB）
#   4. 处理 OCR 模型文件别名兼容性
#   5. 启动 Streamlit Web UI 或 CLI 批处理
#
# 注意：MinerU 装在 /venv-mineru，Pipeline 也在此 venv 中运行
# ============================================================
set -euo pipefail

cd /app

# ── 激活 MinerU venv ──
export PATH="/venv-mineru/bin:$PATH"
export VIRTUAL_ENV="/venv-mineru"

# ── 1. 部署 magic-pdf 配置 ──
mkdir -p /root
if [ -f /app/magic-pdf.json ]; then
    cp /app/magic-pdf.json /root/magic-pdf.json
    echo "[entrypoint] magic-pdf.json → /root/magic-pdf.json"
fi

# ── 2. 创建目录 ──
mkdir -p /app/input_docs /app/output /app/output/logs /app/output/reports /app/models_cache

# ── 3. 环境变量 ──
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HOME="${HF_HOME:-/app/models_cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/app/models_cache/torch}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/app/models_cache/modelscope}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── 4. 下载 MinerU 模型 + OCR 别名处理 ──
python3 - <<'PYEOF'
import sys
from pathlib import Path

model_root = Path("/app/models_cache/PDF-Extract-Kit-1.0")
required_model = model_root / "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
ocr_torch = model_root / "models/OCR/paddleocr_torch"

# OCR 模型别名：MinerU 内部可能用 ch_PP-OCRv3_det_infer.pth 查找
# 但下载下来的文件名可能是 Multilingual_PP-OCRv3_det_infer.pth
ocr_aliases = [
    (
        ocr_torch / "Multilingual_PP-OCRv3_det_infer.pth",
        ocr_torch / "ch_PP-OCRv3_det_infer.pth",
    ),
]

if required_model.exists():
    print(f"[entrypoint] MinerU models OK: {model_root}", flush=True)
else:
    print("[entrypoint] MinerU models not found; downloading opendatalab/PDF-Extract-Kit-1.0 ...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit-1.0",
            local_dir=str(model_root),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as e:
        print(f"[entrypoint] WARNING: HuggingFace download failed: {e}", flush=True)
        print("[entrypoint] Trying ModelScope mirror ...", flush=True)
        try:
            from modelscope import snapshot_download as ms_download
            ms_download("opendatalab/PDF-Extract-Kit-1.0", local_dir=str(model_root))
        except Exception as e2:
            print(f"[entrypoint] ERROR: All download attempts failed: {e2}", flush=True)
            print("[entrypoint] You can manually download and place models in /app/models_cache/", flush=True)

    if not required_model.exists():
        print(f"[entrypoint] WARNING: MinerU model not found after download attempt.", flush=True)
        print(f"[entrypoint] PDF OCR will not work until models are available.", flush=True)
    else:
        print(f"[entrypoint] MinerU models ready: {model_root}", flush=True)

# 处理 OCR 别名
for src, dst in ocr_aliases:
    if src.exists() and not dst.exists():
        import shutil
        shutil.copy2(str(src), str(dst))
        print(f"[entrypoint] OCR alias created: {src.name} → {dst.name}", flush=True)

# 下载 text2vec-base-chinese（语义去重用，可选）
text2vec_dir = Path("/app/models_cache/text2vec-base-chinese")
if not text2vec_dir.exists():
    try:
        print("[entrypoint] Pre-downloading text2vec-base-chinese ...", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="shibing624/text2vec-base-chinese",
            local_dir=str(text2vec_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[entrypoint] text2vec-base-chinese ready: {text2vec_dir}", flush=True)
    except Exception as e:
        print(f"[entrypoint] WARNING: text2vec download failed (non-critical): {e}", flush=True)

print("[entrypoint] Model preparation complete.", flush=True)
PYEOF

# ── 5. 启动模式 ──
mode="${1:-${APP_MODE:-web}}"
shift || true

case "$mode" in
  web)
    echo "[entrypoint] Starting Streamlit Web UI on port ${STREAMLIT_SERVER_PORT:-7860} ..."
    exec streamlit run /app/src/web_app.py \
      --server.port="${STREAMLIT_SERVER_PORT:-7860}" \
      --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
      --server.headless=true \
      --browser.gatherUsageStats=false \
      "$@"
    ;;
  cli)
    echo "[entrypoint] Starting CLI pipeline ..."
    exec python /app/src/main.py --config "${PIPELINE_CONFIG:-/app/config.yaml}" "$@"
    ;;
  bash|shell)
    exec /bin/bash "$@"
    ;;
  *)
    exec "$mode" "$@"
    ;;
esac
