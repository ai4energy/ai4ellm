#!/usr/bin/env bash
# P0 §1 环境确认（只读，不改配置）
# 用法: bash scripts/00_env_check.sh [YYYY-MM-DD]
# 产出: runs/<日期>-env-check/ 下的原始输出
# 红线: 只读 nvidia-smi / docker ps / ls / df，不 stop 任何容器、不碰 GPU0/1。
set -uo pipefail

DATE="${1:-$(date +%F)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/${DATE}-env-check"
mkdir -p "$OUT"

run() { # run <outfile> <description> -- cmd...
    local file="$1" desc="$2"; shift 2
    { echo "# $desc"; echo "# command: $*"; echo "# date: $(date -Is)"; echo; } > "$OUT/$file"
    "$@" >> "$OUT/$file" 2>&1
    local rc=$?
    echo "# exit: $rc" >> "$OUT/$file"
    return $rc
}

echo "== P0 env check -> $OUT"

run 01_nvidia-smi.txt "nvidia-smi (only read)" nvidia-smi
run 02_docker-images-rootless.txt "rootless docker images (current user)" docker images
run 03_models.txt "existing model assets" ls -lh /data/models/llm
run 04_disk.txt "disk space" df -h /data
run 05_docker-ps-rootless.txt "rootless docker ps" docker ps -a

# 系统 docker（root daemon）只读视图；key 由环境变量 ROOT_SSH_KEY 指定，缺省用本机 key。
ROOT_SSH_KEY="${ROOT_SSH_KEY:-$HOME/.ssh/id_ed25519.mingtaoli}"
if ssh -i "$ROOT_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 \
        root@localhost true 2>/dev/null; then
    run 06_docker-images-system.txt "system(root) docker images" \
        ssh -i "$ROOT_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no root@localhost \
        'docker images --format "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | sort'
    run 07_docker-ps-system.txt "system(root) docker ps (names/images/status/ports only)" \
        ssh -i "$ROOT_SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no root@localhost \
        'docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"'
else
    echo "ssh root@localhost unavailable; skipped system docker view" > "$OUT/06_docker-images-system.txt"
fi

# ---- 判据 ----
GPU_MEM=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
gpu2=$(awk -F', ' '$1==2{print $2}' <<<"$GPU_MEM")
gpu3=$(awk -F', ' '$1==3{print $2}' <<<"$GPU_MEM")
{
    echo "# P0 env check summary"
    echo
    echo "## GPU memory used (MiB)"
    echo '```'
    echo "$GPU_MEM"
    echo '```'
    echo
    echo "- GPU2 used=${gpu2:-?} MiB (require < 100)"
    echo "- GPU3 used=${gpu3:-?} MiB (require < 100)"
    if [[ -n "${gpu2:-}" && -n "${gpu3:-}" ]] && (( gpu2 < 100 && gpu3 < 100 )); then
        echo "- RESULT: GPU2/3 free -> PASS"
    else
        echo "- RESULT: GPU2/3 busy -> FAIL"
    fi
} > "$OUT/08_summary.txt"

cat "$OUT/08_summary.txt"
[[ -n "${gpu2:-}" && -n "${gpu3:-}" ]] && (( gpu2 < 100 && gpu3 < 100 ))
