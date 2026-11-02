#!/usr/bin/env bash
# 双卡 DDP QLoRA 验证：容器只暴露 GPU2/3，torchrun 2 进程。
# 用法: MAX_STEPS=20 bash scripts/06_run_qlora_ddp.sh [YYYY-MM-DD]
set -uo pipefail

DATE="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL_OUT="${OUT_REL:-runs/${DATE}-qlora-ddp-smoke}"
MAX_STEPS="${MAX_STEPS:-20}"
KEY="${ROOT_SSH_KEY:-$HOME/.ssh/id_ed25519.mingtaoli}"
IMAGE="${IMAGE:-aigc-train:torch213-cu134}"
UID_GID="$(id -u):$(id -g)"

mkdir -p "$ROOT/$REL_OUT"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no root@localhost \
    bash -s -- "$ROOT" "$REL_OUT" "$MAX_STEPS" "$IMAGE" "$UID_GID" <<'REMOTE' | tee "$ROOT/$REL_OUT/train.log"
set -uo pipefail
ROOT="$1"; REL_OUT="$2"; MAX_STEPS="$3"; IMAGE="$4"; UID_GID="$5"
echo "# qlora-ddp-smoke $(date -Is) max_steps=$MAX_STEPS"
docker run --rm --name "ai4ellm-qlora-ddp-$$" \
    --gpus '"device=2,3"' --user "$UID_GID" --shm-size=16g \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e NCCL_DEBUG=WARN \
    -v /data/models:/models:ro -v "$ROOT":/work -w /work \
    "$IMAGE" torchrun --standalone --nproc_per_node=2 \
    /work/train/qlora_ddp.py --config /work/train/qlora_smoke.yaml \
    --max-steps "$MAX_STEPS" --output-dir "/work/$REL_OUT"
rc=$?
echo "# qlora-ddp exit: $rc"
exit $rc
REMOTE
rc=${PIPESTATUS[0]}
echo "# ssh exit: $rc"
exit $rc
