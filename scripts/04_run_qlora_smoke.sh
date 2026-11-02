#!/usr/bin/env bash
# P0 §4 启动 QLoRA 100 步冒烟：容器内只暴露 GPU2，用户 1001，产物落 runs/<日期>-qlora-smoke/。
# 用法: bash scripts/04_run_qlora_smoke.sh [YYYY-MM-DD]
set -uo pipefail

DATE="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL_OUT="runs/${DATE}-qlora-smoke"
KEY="${ROOT_SSH_KEY:-$HOME/.ssh/id_ed25519.mingtaoli}"
IMAGE="${IMAGE:-aigc-train:torch213-cu134}"
UID_GID="$(id -u):$(id -g)"

mkdir -p "$ROOT/$REL_OUT"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no root@localhost \
    bash -s -- "$ROOT" "$REL_OUT" "$IMAGE" "$UID_GID" <<'REMOTE' | tee "$ROOT/$REL_OUT/train.log"
set -uo pipefail
ROOT="$1"; REL_OUT="$2"; IMAGE="$3"; UID_GID="$4"
echo "# qlora-smoke $(date -Is)"
echo "# image=$IMAGE gpus=device=2 user=$UID_GID"
docker run --rm --name "ai4ellm-qlora-smoke-$$" \
    --gpus '"device=2"' --user "$UID_GID" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v /data/models:/models:ro -v "$ROOT":/work -w /work \
    "$IMAGE" /venv/bin/python /work/train/qlora_smoke.py --config /work/train/qlora_smoke.yaml
rc=$?
echo "# qlora-smoke exit: $rc"
exit $rc
REMOTE
rc=${PIPESTATUS[0]}
echo "# ssh exit: $rc"
exit $rc
