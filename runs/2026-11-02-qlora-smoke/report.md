# P0 §4 QLoRA 100 步冒烟报告（Qwen3.8-27B / RTX 5090）

日期：2026-11-02 · 训练卡：**GPU2 单卡**（GPU3 空闲，GPU0/1 生产 vLLM 未触碰）

## 1. 判据结果

| 判据 | 结果 |
|---|---|
| 100 步不 OOM | ✅ `exit 0`，峰值 28.638 GiB / 32 GB |
| loss 有限且整体下降 | ✅ 全部有限；2.741072 → 0.024565；线性斜率 -0.011489/step |
| adapter 能保存并由 peft 重新加载 | ✅ 891MB adapter；同进程 reload_loss=0.005863；新进程 verify_loss=0.005918 |

## 2. 环境

- 镜像：`aigc-train:torch213-cu134`（`aigc-base:torch213-cu134` + 钉版依赖）
- torch 2.13.0+cu130 / CUDA runtime 13.4 / transformers 5.12.1
- bitsandbytes 0.50.2 / peft 0.21.2 / trl 1.17.1 / datasets 5.1.0 / accelerate 1.15.0
- GPU：NVIDIA GeForce RTX 5090，capability `(12, 0)`，单卡容器 `--gpus '"device=2"'`
- 完整版本输出与 pip freeze：`runs/2026-11-02-env-smoke/`

## 3. 数据

`data/synthetic-50.jsonl`，50 条合成代码/问答样本，全部 `"synthetic": true`。

```text
[2026-11-02T08:46:42] dataset: 50 usable rows; tokens min/avg/max=243/642/2048
```

其中每 5 条抽 1 条追加长附录，10 条达到 2048 token（truncation 上限），用于覆盖目标序列长度下的显存。

## 4. 配置（`train/qlora_smoke.yaml` 原文）

```yaml
run_name: qlora-smoke
model_path: /models/llm/Qwen3.8-27B-bf16
data_path: /work/data/synthetic-50.jsonl
output_dir: /work/runs/2026-11-02-qlora-smoke
seed: 42

quant:
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16

lora:
  r: 32
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - in_proj_qkv
    - in_proj_z
    - in_proj_b
    - in_proj_a
    - out_proj

train:
  max_length: 2048
  batch_size: 1
  grad_accum: 4
  max_steps: 100
  lr: 1.0e-4
  weight_decay: 0.0
  warmup_steps: 5
  scheduler: cosine
  gradient_checkpointing: true
  max_grad_norm: 1.0
  logging_steps: 1
```

LoRA 实际可训练参数：`trainable=233,455,616 (1.537% of 15,186,957,552)`。

## 5. 原始命令

```bash
bash scripts/04_run_qlora_smoke.sh
```

脚本内实际容器命令：

```bash
docker run --rm --name "ai4ellm-qlora-smoke-$$" \
    --gpus '"device=2"' --user "$(id -u):$(id -g)" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v /data/models:/models:ro -v "$ROOT":/work -w /work \
    aigc-train:torch213-cu134 /venv/bin/python /work/train/qlora_smoke.py \
    --config /work/train/qlora_smoke.yaml
```

全新进程重载校验：

```bash
docker run --rm --gpus '"device=2"' --user 1001:1001 \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v /data/models:/models:ro -v /home/ai4e-hermes/ai4ellm:/work -w /work \
    aigc-train:torch213-cu134 /venv/bin/python /work/train/qlora_smoke.py \
    --config /work/train/qlora_smoke.yaml --verify-only
```

## 6. 原始输出

### 6.1 训练日志（前 10 步与后 10 步）

```text
[2026-11-02T08:48:50] step 1/100 loss=2.7411 lr=4.00e-05 23.35s 126.5 tok/s peak=27.77GiB
[2026-11-02T08:49:03] step 2/100 loss=2.6019 lr=6.00e-05 13.44s 213.4 tok/s peak=28.64GiB
[2026-11-02T08:49:13] step 3/100 loss=2.3679 lr=8.00e-05 10.29s 119.9 tok/s peak=28.64GiB
[2026-11-02T08:49:24] step 4/100 loss=1.9457 lr=1.00e-04 10.49s 105.8 tok/s peak=28.64GiB
[2026-11-02T08:49:44] step 5/100 loss=0.8053 lr=1.00e-04 20.04s 323.3 tok/s peak=28.64GiB
[2026-11-02T08:49:55] step 6/100 loss=1.3940 lr=1.00e-04 10.67s 103.3 tok/s peak=28.64GiB
[2026-11-02T08:50:13] step 7/100 loss=0.8680 lr=9.99e-05 18.01s 262.7 tok/s peak=28.64GiB
[2026-11-02T08:50:26] step 8/100 loss=1.4072 lr=9.98e-05 12.86s 88.0 tok/s peak=28.64GiB
[2026-11-02T08:50:37] step 9/100 loss=1.0167 lr=9.96e-05 11.18s 106.6 tok/s peak=28.64GiB
[2026-11-02T08:50:48] step 10/100 loss=1.3542 lr=9.93e-05 10.93s 101.0 tok/s peak=28.64GiB
...
[2026-11-02T09:09:12] step 91/100 loss=0.0190 lr=2.20e-06 14.79s 199.5 tok/s peak=28.64GiB
[2026-11-02T09:09:23] step 92/100 loss=0.0178 lr=1.74e-06 11.33s 120.6 tok/s peak=28.64GiB
[2026-11-02T09:09:34] step 93/100 loss=0.0195 lr=1.33e-06 10.96s 122.2 tok/s peak=28.64GiB
[2026-11-02T09:09:47] step 94/100 loss=0.0205 lr=9.81e-07 13.25s 210.1 tok/s peak=28.64GiB
[2026-11-02T09:10:04] step 95/100 loss=0.0144 lr=6.82e-07 16.75s 277.5 tok/s peak=28.64GiB
[2026-11-02T09:10:14] step 96/100 loss=0.0223 lr=4.37e-07 10.47s 101.0 tok/s peak=28.64GiB
[2026-11-02T09:10:31] step 97/100 loss=0.0157 lr=2.46e-07 16.70s 277.9 tok/s peak=28.64GiB
[2026-11-02T09:10:45] step 98/100 loss=0.0170 lr=1.09e-07 13.50s 213.9 tok/s peak=28.64GiB
[2026-11-02T09:10:59] step 99/100 loss=0.0175 lr=2.73e-08 13.87s 211.2 tok/s peak=28.64GiB
[2026-11-02T09:11:09] step 100/100 loss=0.0246 lr=0.00e+00 10.51s 108.8 tok/s peak=28.64GiB
```

### 6.2 摘要（`summary.json`）

```json
{
  "max_steps": 100,
  "grad_accum": 4,
  "batch_size": 1,
  "max_length": 2048,
  "train_time_s": 1362.7,
  "first_loss": 2.7410720884799957,
  "last_loss": 0.02456502290442586,
  "min_loss": 0.011696809087879956,
  "loss_decreasing": true,
  "gpu_peak_gib": 28.638,
  "gpu_peak_reserved_gib": 29.783,
  "mean_tokens_per_s": 188.5,
  "reload_loss": 0.005862938240170479,
  "adapter_dir": "/work/runs/2026-11-02-qlora-smoke/adapter"
}
```

### 6.3 指标统计（`stats.json`，由 `scripts/05_analyze_metrics.py` 从 100 行 metrics 计算）

```json
{
  "run_dir": "runs/2026-11-02-qlora-smoke",
  "steps": 100,
  "first_step": 1,
  "last_step": 100,
  "loss_first": 2.741072,
  "loss_last": 0.024565,
  "loss_min": 0.011697,
  "loss_max": 2.741072,
  "loss_mean_first10": 1.650193,
  "loss_mean_last10": 0.01884,
  "loss_slope_per_step": -0.011489036273627363,
  "loss_finite_all": true,
  "step_time_s_mean": 13.627,
  "step_time_s_median": 13.699,
  "step_time_s_p95": 18.01,
  "step_time_s_max": 23.354,
  "tokens_per_s_mean": 179.1,
  "tokens_per_s_median": 199.7,
  "gpu_peak_gib": 28.638,
  "total_tokens": 256912
}
```

每 10 步原始指标（完整 100 步见 `metrics.jsonl`）：

| step | loss | lr | step_time_s | tokens | tokens_per_s | peak_gib |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.741072 | 4.00e-05 | 23.35 | 2954 | 126.5 | 27.771 |
| 10 | 1.354234 | 9.93e-05 | 10.93 | 1104 | 101.0 | 28.638 |
| 20 | 0.262775 | 9.40e-05 | 13.64 | 2951 | 216.3 | 28.638 |
| 30 | 0.082671 | 8.39e-05 | 15.98 | 2923 | 183.0 | 28.638 |
| 40 | 0.039243 | 7.01e-05 | 11.65 | 1218 | 104.6 | 28.638 |
| 50 | 0.025440 | 5.41e-05 | 16.92 | 4588 | 271.2 | 28.638 |
| 60 | 0.020969 | 3.77e-05 | 17.05 | 4694 | 275.3 | 28.638 |
| 70 | 0.022263 | 2.27e-05 | 14.78 | 2902 | 196.4 | 28.638 |
| 80 | 0.027617 | 1.05e-05 | 10.54 | 1012 | 96.0 | 28.638 |
| 90 | 0.019309 | 2.71e-06 | 14.19 | 2877 | 202.8 | 28.638 |
| 100 | 0.024565 | 0.00e+00 | 10.51 | 1143 | 108.8 | 28.638 |

### 6.4 adapter 保存与重载（原始输出）

```text
[2026-11-02T09:11:10] adapter saved -> /work/runs/2026-11-02-qlora-smoke/adapter
[2026-11-02T09:11:15] adapter reload OK: reload_loss=0.005863
```

全新进程：

```text
[2026-11-02T09:11:48] base loaded in 18.7s; max_memory_allocated=16.71 GiB
[2026-11-02T09:12:03] VERIFY {"adapter_dir": "/work/runs/2026-11-02-qlora-smoke/adapter", "lora_tensors": 992, "verify_loss": 0.005918173119425774, "finite": true}
# verify exit: 0
```

`adapter_config.json` 关键字段：

```json
{
  "base_model_name_or_path": "/models/llm/Qwen3.8-27B-bf16",
  "peft_type": "LORA",
  "peft_version": "0.21.2",
  "r": 32,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "task_type": "CAUSAL_LM"
}
```

## 7. 产物

```text
runs/2026-11-02-qlora-smoke/
  train.log          # 原始训练日志
  metrics.jsonl      # 100 行，每步 loss/lr/耗时/tokens/tok/s/显存
  summary.json
  stats.json
  verify.log         # 新进程 peft 重载日志
  verify.json
  adapter/           # 891MB adapter + tokenizer + adapter_config.json
```

## 8. 遇到的问题与处理

1. **默认 HF loss 在 2048 上 OOM**：`2048×248320×4 = 1.89 GiB` 的 fp32 logits 分配失败。
   已改为 `train/loss_utils.py` 的分块 LM head + checkpoint CE，并换
   `PagedAdamW8bit` + `expandable_segments`。详见
   `runs/2026-11-02-env-smoke/report.md`。序列 2048 / bs1 / grad accum 4 未改变。
2. **Qwen3.5 linear-attention fast path 不可用**：
   `[transformers] The fast path is not available ... flash-linear-attention / causal-conv1d`，
   当前回退 torch 实现。功能正常，吞吐受影响；可在正式训练前评估安装 `fla` /
   `causal-conv1d` 是否支持 sm_120。
3. **bitsandbytes 导入时提示 CPU `kernels` 模块缺失**：
   `Failed to load CPU gemm_4bit_forward from kernels-community: No module named 'kernels'`。
   GPU 路径已由预检和 100 步训练验证，不阻塞；若后续用到 CPU 4bit 再补。
4. **adapter 的 `base_model_name_or_path` 是容器路径** `/models/llm/Qwen3.8-27B-bf16`；
   换机器（A800/Slurm）时需保持同路径挂载或修改该字段。

## 9. 红线合规

- 只用 GPU2；GPU3 全程 2 MiB 空闲；未对 GPU0/1 的 vLLM 做任何操作。
- 未使用 `sudo`；系统 docker 通过 owner 提供的 `root@localhost` key 访问，仅创建/运行本项目容器。
- 未删除 `/data/models/llm` 下任何既有模型；新增目录 `Qwen3.8-27B-bf16/`。
- `scripts/`、`train/`、`runs/`、`containers/` 中无 API key / `vllm.key` 内容。

## 10. 结论

P0 里程碑达成：27B bf16 基座 → NF4 4bit QLoRA（r=32, alpha=32, dropout=0.05），
单卡 5090、seq 2048、bs1、grad accum 4，100 步 22.7 分钟，不 OOM，loss 有限且显著下降，
adapter 可保存并由 peft 在新进程中重载。

注意：本报告只证明**训练管线与 sm_120/bnb 兼容性**；loss 下降来自 50 条合成样本的
训练拟合，不能说明模型质量或泛化。下一步应优先做 ESML-Bench v0 与执行验证过的 SFT 数据，
再决定 2×5090 DDP 或 A800 正式训练。
