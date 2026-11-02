#!/usr/bin/env bash
# P0 §3 环境冒烟：容器内只暴露 GPU2，记录版本/pip freeze，跑 sm_120 + bnb 前后向预检。
# 用法: bash scripts/03_env_smoke.sh [YYYY-MM-DD]
# 产出: runs/<日期>-env-smoke/{container-env.txt,pip-freeze.txt,preflight.json,preflight.log}
set -uo pipefail

DATE="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REL_OUT="runs/${DATE}-env-smoke"
KEY="${ROOT_SSH_KEY:-$HOME/.ssh/id_ed25519.mingtaoli}"
IMAGE="${IMAGE:-aigc-train:torch213-cu134}"
UID_GID="$(id -u):$(id -g)"

mkdir -p "$ROOT/$REL_OUT"

ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=no root@localhost \
    bash -s -- "$ROOT" "$REL_OUT" "$IMAGE" "$UID_GID" <<'REMOTE' | tee "$ROOT/$REL_OUT/preflight.log"
set -uo pipefail
ROOT="$1"; REL_OUT="$2"; IMAGE="$3"; UID_GID="$4"
echo "# env-smoke $(date -Is)"
echo "# image=$IMAGE gpus=device=2 user=$UID_GID"
docker run --rm --name "ai4ellm-env-smoke-$$" \
    --gpus '"device=2"' --user "$UID_GID" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v /data/models:/models:ro -v "$ROOT":/work -w /work \
    "$IMAGE" bash -c '
set -uo pipefail
OUT="$1"
{
  echo "## date"; date -Is
  echo "## nvidia-smi (container, only GPU2)"; nvidia-smi
  echo "## python / packages"
  /venv/bin/python -c "import sys, torch, transformers, peft, bitsandbytes, accelerate, datasets, trl; print(\"python\", sys.version); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda); print(\"transformers\", transformers.__version__); print(\"peft\", peft.__version__); print(\"bitsandbytes\", bitsandbytes.__version__); print(\"accelerate\", accelerate.__version__); print(\"datasets\", datasets.__version__); print(\"trl\", trl.__version__)"
} > "$OUT/container-env.txt" 2>&1
/venv/bin/pip freeze > "$OUT/pip-freeze.txt" 2>&1
echo "## pip freeze lines: $(wc -l < "$OUT/pip-freeze.txt")"
cat "$OUT/container-env.txt"
echo "## preflight (4bit load + 1 step fwd/bwd)"
/venv/bin/python /work/train/preflight_bnb.py
rc=$?
echo "# preflight exit: $rc"
exit $rc
' _ "$REL_OUT"
REMOTE
rc=${PIPESTATUS[0]}
echo "# env-smoke ssh exit: $rc"
exit $rc
