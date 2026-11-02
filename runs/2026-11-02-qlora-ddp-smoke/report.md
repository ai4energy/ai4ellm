# 双卡 DDP QLoRA 验证报告（2026-11-02）

目的：在 GPU2+GPU3 上验证 DDP 数据并行、NCCL 通信、显存与吞吐；**不是**质量训练。
数据集仍是 50 条合成样本，仅用于管线验证。

## 1. 环境与命令

- 容器：`aigc-train:torch213-cu134`，`--gpus '"device=2,3"'`，`--shm-size=16g`
- 无 NVLink；NCCL 走 PCIe/SHM，设置 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`
- 启动：

```bash
MAX_STEPS=20 bash scripts/06_run_qlora_ddp.sh
```

容器内实际命令：

```bash
torchrun --standalone --nproc_per_node=2 \
  /work/train/qlora_ddp.py --config /work/train/qlora_smoke.yaml \
  --max-steps 20 --output-dir /work/runs/2026-11-02-qlora-ddp-smoke
```

## 2. NCCL 探针（原始输出）

```text
NCCL version 2.29.7+cuda13.2
NCCL_OK world=2 all_reduce_sum=[3.0, 3.0, 3.0, 3.0]
# nccl probe exit: 0
```

## 3. 训练原始输出（首尾步 + summary）

```text
[2026-11-02T09:36:38] step 13/20 loss=0.6172 lr=4.48e-05 15.10s 268.0 tok/s peak=29.51GiB
[2026-11-02T09:36:57] step 14/20 loss=0.4443 lr=3.45e-05 18.22s 321.8 tok/s peak=29.51GiB
[2026-11-02T09:37:14] step 15/20 loss=0.5023 lr=2.50e-05 17.53s 432.8 tok/s peak=29.51GiB
[2026-11-02T09:37:29] step 16/20 loss=0.4327 lr=1.65e-05 14.32s 281.5 tok/s peak=29.51GiB
[2026-11-02T09:37:46] step 17/20 loss=0.4248 lr=9.55e-06 17.46s 336.9 tok/s peak=29.51GiB
[2026-11-02T09:37:57] step 18/20 loss=0.3849 lr=4.32e-06 11.26s 207.4 tok/s peak=29.51GiB
[2026-11-02T09:38:12] step 19/20 loss=0.5035 lr=1.09e-06 14.32s 283.0 tok/s peak=29.51GiB
[2026-11-02T09:38:29] step 20/20 loss=0.3769 lr=0.00e+00 17.74s 336.8 tok/s peak=29.51GiB
[2026-11-02T09:38:31] adapter saved -> /work/runs/2026-11-02-qlora-ddp-smoke/adapter
[2026-11-02T09:38:34] adapter reload OK: reload_loss=0.382522
# qlora-ddp exit: 0
```

```json
{
  "world_size": 2,
  "max_steps": 20,
  "grad_accum": 4,
  "batch_size": 1,
  "max_length": 2048,
  "train_time_s": 328.7,
  "first_loss": 3.767005443572998,
  "last_loss": 0.37693360447883606,
  "min_loss": 0.37693360447883606,
  "loss_decreasing": true,
  "gpu_peak_gib": 29.507,
  "gpu_peak_reserved_gib": 30.338,
  "mean_tokens_per_s": 312.9,
  "reload_loss": 0.3825223743915558,
  "adapter_dir": "/work/runs/2026-11-02-qlora-ddp-smoke/adapter"
}
```

## 4. 单卡 vs 双卡（同一套 50 条合成数据，仅比较吞吐）

| 项目 | 单卡 GPU2（100 步） | 双卡 DDP（20 步） |
|---|---:|---:|
| 每步有效样本 | 1×4=4 | 2×4=8 |
| 全局 tokens/s 均值 | 188.5 | 312.9 |
| 每步耗时中位 | 13.699 s | 17.405 s |
| 显存峰值/卡 | 28.638 GiB | 29.507 GiB |
| 扩展比 | 1.00× | **约 1.66×** |

说明：双卡每步处理 2 倍样本，墙钟只增加约 27%，因此全局吞吐约 1.66×；
不是 2×，原因包括 PCIe/SHM all-reduce、`find_unused_parameters=True` 的图遍历、
每卡独立加载 4bit 基座。对 233M LoRA 参数这个量级，1.66× 在无 NVLink 的 5090 上属正常。

## 5. 结论与限制

- DDP、NCCL、rank 分片数据、rank0 保存/重载均验证通过。
- 每卡峰值 29.507 GiB，距离 32 GB 约 1.9 GiB；正式训练建议保持 seq 2048，
  不要把 seq 直接拉到 4096，并保留 `expandable_segments`。
- 当前脚本**还没有断点续训**；正式训练前需补 checkpoint/resume。
- 本次 loss（3.767 → 0.377）与单卡不可直接比较：全局 batch 和数据分片不同，且只有 20 步。
- 下一步：ESML-Bench v0 + 执行验证 SFT 数据；之后再决定用 2×5090 正式训练还是租 A800。

原始文件：`train.log`、`metrics.jsonl`、`stats.json`、`summary.json`、`adapter/`。
