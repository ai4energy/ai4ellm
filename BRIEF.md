# 启动工单 · ai4ellm · P0 环境与资产

> 给新开的 tmux + Claude Code 会话。先完整读 `PLAN.md`，本文件只定义**第一里程碑**。
> 工作目录：`~/ai4ellm`。

## 当前前提（2026-11-02）

- GPU0（18110）与 GPU1（18111）在跑 ai4e11/ai4e12 的生产 vLLM，**绝对不要动**。
- GPU2（voice）与 GPU3（ComfyUI）已停，显存各 2 MiB，**本次训练只准用 GPU2/3**。
- `/data/models/llm/Qwen3.8-27B` 下只有 GGUF；**没有 bf16 safetensors**。
- 训练栈未装；宿主 Python 没有 torch。

## 第一里程碑：把 27B QLoRA 在 5090 上跑通 100 步

### 1. 环境确认（不改配置）

```bash
nvidia-smi
docker images | grep -E "torch|vllm|cu134"
ls -lh /data/models/llm
df -h /data
```

要求：GPU2/3 显存 < 100 MiB；GPU0/1 不动；记录原始输出到 `runs/<日期>-env-check/`。

### 2. 下载 bf16 基座

- 目标：`/data/models/llm/Qwen3.8-27B-bf16/`，约 54GB。
- 直连被墙：用 `HF_ENDPOINT=https://hf-mirror.com` 或 `ALL_PROXY=socks5h://localhost:1254`。
- 校验：文件数、总大小、`model.safetensors.index.json` 存在；下载日志入 `runs/`。
- 不删 `/data/models/llm` 下任何既有目录。

### 3. 训练环境（优先容器）

- 候选镜像：`aigc-vllm:cu134`（torch 2.13 + CUDA 13.4）或
  `/srv/aigc/aigc-infra/docker/base-torch213-cu134` 构建的基座。
- 容器必须 `--gpus '"device=2,3"'`，挂载 `/data/models`。
- 安装并锁定版本：`transformers`、`peft`、`trl`、`datasets`、`bitsandbytes`、`accelerate`，
  可选 `unsloth`。把 `pip freeze` 写入 `runs/<日期>-env-smoke/`。
- **先验证 sm_120**：`torch.cuda.get_device_capability()` 应为 `(12, 0)`；
  做一次 4-bit 量化加载 + 单步前向/反向，确认 bitsandbytes 可用。
  若 bitsandbytes 不支持，停在这里报告，不要自己换训练方案。

### 4. 100 步 QLoRA 冒烟

- 模型：bf16 基座 → 4bit QLoRA（NF4），r=32，alpha=32，dropout 0.05。
- 序列长度 2048、batch 1、grad accum 4，单卡 GPU2；跑 100 步。
- 数据：先用 `data/` 下的小样本（没有就先造 50 条合成代码/问答样本，标注 synthetic）。
- 记录：显存峰值、tokens/s、loss 曲线、每步耗时；原始日志入 `runs/<日期>-qlora-smoke/`。
- 判据：100 步不 OOM、loss 有限且整体下降、adapter 能保存并由 `peft` 重新加载。

### 5. 产出

```text
scripts/00_env_check.sh
scripts/01_download_base.sh
train/qlora_smoke.yaml
runs/<日期>-env-check/report.md
runs/<日期>-qlora-smoke/report.md
```

报告里必须有**原始命令行与原始输出**，不要转述数字。

## 红线

- 不碰 GPU0/1 的 vLLM（18110/18111）；不 `docker stop` 这两个容器。
- 不 `sudo`（需要 root 时先问 owner）；不 `rm -rf`；不删 `/data/models` 既有模型。
- 不把 `~/.config/vllm.key` 或任何密钥写进脚本、日志、报告。
- 大下载前先看磁盘；下载失败保留日志，不反复重试刷屏。
- 卡住就把卡点原文贴出并停，不要猜。

## 完成后

把 `runs/<日期>-qlora-smoke/report.md` 给 owner 看，再决定下一步：
- 继续 ESML-Bench v0（推荐，与训练并行）；
- 或租 1×A800 80GB 跑 27B QLoRA 正式训练（¥6/h）。
