# P0 §3 训练环境冒烟报告（sm_120 / bitsandbytes）

日期：2026-11-02 · 只使用 GPU2（容器 `--gpus '"device=2"'`）；GPU0/1 未触碰。

## 1. 镜像与版本

构建文件：`containers/train/Dockerfile`

```dockerfile
FROM aigc-base:torch213-cu134
RUN /venv/bin/pip install --no-cache-dir \
        bitsandbytes==0.50.2 peft==0.21.2 trl==1.17.1 \
        datasets==5.1.0 accelerate==1.15.0
```

构建命令：

```bash
docker build --network=host -t aigc-train:torch213-cu134 ~/ai4ellm/containers/train
```

原始环境输出（`container-env.txt`）：

```text
## date
2026-11-02T08:41:41+00:00
## nvidia-smi (container, only GPU2)
...
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:A8:00.0 Off |                  N/A |
| 39%   24C    P8             12W /  540W |       2MiB /  32607MiB |      0%      Default |
...
python 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]
torch 2.13.0+cu130 cuda 13.0
transformers 5.12.1
peft 0.21.2
bitsandbytes 0.50.2
accelerate 1.15.0
datasets 5.1.0
trl 1.17.1
```

完整 `pip freeze`：`pip-freeze.txt`（87 行）。

## 2. 预检命令

```bash
bash scripts/03_env_smoke.sh
# 容器内实际执行:
#   /venv/bin/python /work/train/preflight_bnb.py
```

预检内容：真实 27B bf16 基座 → NF4 4bit 加载 → 挂 r=32 LoRA → 取最长样本
（2048 token）做一次 forward + backward + `PagedAdamW8bit.step()`。

## 3. 第一次尝试：默认 HF loss 在 2048 上 OOM（原始输出）

```text
PREFLIGHT_LOAD {... "load_seconds": 19.6, "linear4bit_modules": 606, "gpu_alloc_after_load_gib": 16.69}
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[W1102 08:42:31.693532430 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2034237440 bytes (free: 277413888, total: 33711521792).
[W1102 08:42:31.693994562 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2034237440 bytes (free: 281608192, total: 33711521792).
Traceback (most recent call last):
  File "/work/train/preflight_bnb.py", line 112, in <module>
    main()
  File "/work/train/preflight_bnb.py", line 91, in main
    loss.backward()
  ...
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB. GPU 0 has a total capacity of 31.40 GiB of which 268.56 MiB is free. Including non-PyTorch memory, this process has 31.12 GiB memory in use. Of the allocated memory 28.68 GiB is allocated by PyTorch, and 1.82 GiB is reserved by PyTorch but unallocated.
```

根因：`2034237440 B = 2048 × 248320 × 4 B`，即 transformers 默认 loss 把完整
logits 转成 fp32。不是 sm_120/bitsandbytes 不兼容。

## 4. 修复（不改变 P0 训练规格）

`train/loss_utils.py` 改为低显存 loss：
1. 只跑 `model.model.language_model` 取 `last_hidden_state`（约 21 MiB），
   不再让 `Qwen3_5ForConditionalGeneration` 物化 `[2048, 248320]` logits；
2. 按 256 token 分块过 `lm_head` + cross-entropy，每块用
   `torch.utils.checkpoint(..., use_reentrant=False)` 重算；
3. 优化器改用 `bitsandbytes.optim.PagedAdamW8bit`；
4. 容器设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

序列长度 2048、batch 1、grad accum 4 的规格不变。

## 5. 修复后预检结果（原始输出）

```text
PREFLIGHT_ENV {"torch": "2.13.0+cu130", "torch_cuda": "13.0", "bitsandbytes": "0.50.2", "device_name": "NVIDIA GeForce RTX 5090", "capability": [12, 0], "capability_is_sm120": true}
PREFLIGHT_LOAD {"torch": "2.13.0+cu130", "torch_cuda": "13.0", "bitsandbytes": "0.50.2", "device_name": "NVIDIA GeForce RTX 5090", "capability": [12, 0], "capability_is_sm120": true, "load_seconds": 19.4, "linear4bit_modules": 606, "gpu_alloc_after_load_gib": 16.67}
PREFLIGHT_RESULT {"torch": "2.13.0+cu130", "torch_cuda": "13.0", "bitsandbytes": "0.50.2", "device_name": "NVIDIA GeForce RTX 5090", "capability": [12, 0], "capability_is_sm120": true, "load_seconds": 19.4, "linear4bit_modules": 606, "gpu_alloc_after_load_gib": 16.67, "sample_tokens": 2048, "loss": 0.906522, "loss_finite": true, "backward_ok": true, "lora_tensors_with_grad": 992, "grad_norm_sum": 12.738162, "optimizer": "PagedAdamW8bit", "optimizer_step_ok": true, "step_seconds": 15.2, "gpu_peak_gib": 27.77}
# preflight exit: 0
```

## 6. 结论

- `torch.cuda.get_device_capability()` = `(12, 0)`，sm_120 确认；
- bitsandbytes 0.50.2 在 CUDA 13 / RTX 5090 上可完成 4bit NF4 加载（606 个
  `Linear4bit` 模块）、单步前向/反向（992 个 LoRA tensor 有有限梯度）和
  8bit paged optimizer step；
- 2048 token、单卡 GPU2 峰值 27.77 GiB，低于 32 GB；
- 可以进入 §4 的 100 步 QLoRA。

原始文件：`container-env.txt`、`pip-freeze.txt`、`preflight.log`、`preflight.json`。
