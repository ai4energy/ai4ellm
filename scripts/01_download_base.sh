#!/usr/bin/env bash
# P0 §2 下载 Qwen3.8-27B bf16 基座（HF 直连被墙，走 hf-mirror）
# 用法: bash scripts/01_download_base.sh
# 产出: /data/models/llm/Qwen3.8-27B-bf16/ + runs/<日期>-download-base/{download.log,verify.txt}
# 约束: 不删除 /data/models/llm 下任何既有目录；下载失败保留日志，不自动重试。
set -uo pipefail

DATE="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-Qwen/Qwen3.8-27B}"
DEST="${DEST:-/data/models/llm/Qwen3.8-27B-bf16}"
OUT="$ROOT/runs/${DATE}-download-base"
HF="${HF:-$ROOT/.venv/bin/hf}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # hf-mirror 不支持 xet 存储
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

mkdir -p "$OUT" "$DEST"

{
    echo "# download base"
    echo "# date: $(date -Is)"
    echo "# repo: $REPO"
    echo "# dest: $DEST"
    echo "# HF_ENDPOINT=$HF_ENDPOINT HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
    echo "# command: $HF download $REPO --local-dir $DEST --max-workers 4"
    echo
} | tee "$OUT/download.log"

"$HF" download "$REPO" --local-dir "$DEST" --max-workers 4 >> "$OUT/download.log" 2>&1
rc=$?
echo "# exit: $rc" >> "$OUT/download.log"

{
    echo "# verify (date: $(date -Is))"
    echo
    echo "## files"
    ls -lh "$DEST"
    echo
    echo "## shard count"
    shards=$(find "$DEST" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
    echo "safetensors_shards=$shards (expect 18)"
    echo
    echo "## total bytes of safetensors shards"
    bytes=$(find "$DEST" -maxdepth 1 -name 'model-*-of-*.safetensors' -printf '%s\n' | awk '{s+=$1} END{printf "%d", s}')
    echo "safetensors_bytes=$bytes"
    echo
    echo "## index"
    if [[ -f "$DEST/model.safetensors.index.json" ]]; then
        echo "index_exists=yes"
        jq -r '"index_total_size=\(.metadata.total_size) tensors=\(.weight_map|length)"' \
            "$DEST/model.safetensors.index.json"
    else
        echo "index_exists=no"
    fi
    echo
    echo "## disk"
    du -sh "$DEST"
    df -h /data
} > "$OUT/verify.txt" 2>&1

cat "$OUT/verify.txt"
if (( rc != 0 )); then
    echo "DOWNLOAD FAILED rc=$rc — log: $OUT/download.log" >&2
fi
exit $rc
