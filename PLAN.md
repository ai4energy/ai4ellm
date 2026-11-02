# ai4ellm · 能源领域 LLM 总计划

> 建立：2026-11-02 · 目录：`~/ai4ellm`
> 目标：把 ESML / 能源领域的代码、文档、工单和**可执行任务**，变成训练与评测本地编程代理的数据和判据。
> 原始计划留档：`PLAN-original.md`（2026-11-02 之前的版本）。

---

## 0 · 当前状态快照（2026-11-02）

### GPU 与端口

| GPU | 用途 | 端口 | 状态 |
|---|---|---|---|
| 0 | vLLM Qwen3.8-27B **1号**（生产，ai4e12 在用） | 18110 | 运行中，**勿动** |
| 1 | vLLM Qwen3.8-27B **2号**（生产，ai4e11 在用） | 18111 | 运行中，**勿动** |
| 2 | 训练 / 实验 | — | voice 已停，空闲 32GB |
| 3 | 训练 / 实验 | — | ComfyUI 已停，空闲 32GB |

- 已停服务：`aigc-infra-voice`（原 18120）、`aigc-infra-comfyui-gpu3`（原 18141）。
  ComfyUI 网关 `aigc-nginx` 仍在监听 18130/18131，但后端已停，会 502。
- 恢复命令：`sudo aigc up voice comfyui-gpu3`（或 `docker start aigc-infra-voice aigc-infra-comfyui-gpu3`）。
  **训练期间不要恢复**，否则和训练抢 GPU2/3。
- GPU0/1 是 ai4e11/ai4e12 的工作模型；训练脚本只准用 `CUDA_VISIBLE_DEVICES=2,3`。

### 模型资产（`/data/models/llm`）

| 路径 | 内容 | 大小 | 用途 |
|---|---|---:|---|
| `Qwen3.8-27B-AWQ-INT4/` | AWQ INT4 safetensors | 20G | 现有 vLLM 推理 |
| `Qwen3.8-27B/Qwen3.8-27B-UD-Q5_K_M.gguf` | GGUF Q5_K_M | 19.7G | llama.cpp |
| `QuantTrio/Qwen3.6-27B-AWQ/` | 旧版 Qwen3.6 | — | 备选 |
| `google/gemma-4-31B-it/` | bf16 | — | 外部对照 |

- **缺 bf16 基座**：QLoRA / LoRA 需要 `Qwen/Qwen3.8-27B` 官方 bf16 safetensors（约 54GB）。
  第一步下载到 `/data/models/llm/Qwen3.8-27B-bf16/`；HF 直连被墙，用 `hf-mirror.com`
  或本机 `socks5h://localhost:1254` 代理。
- 磁盘：`/data` 余约 8.2T，够放基座 + 多轮 adapter。

### 训练栈现状

- 宿主 `python3` **没有 torch**；不要在系统 Python 里硬装。
- 建议用容器：`aigc-vllm:cu134`（torch 2.13 + CUDA 13.4）或
  `/srv/aigc/aigc-infra/docker/base-torch213-cu134` 构建的基座镜像。
- 需要装/验证：`transformers`、`peft`、`trl`、`datasets`、`bitsandbytes`、`accelerate`，
  可选 `unsloth`。
- **RTX 5090 是 sm_120**：bitsandbytes / FlashAttention / Unsloth 的兼容性必须先做
  100 步冒烟，再谈正式训练。

### 相关位置

- 本目录：`~/ai4ellm/`
- ESML 主仓：`/home/ai4e-hermes/esml-engine-user3`
- 推理服务编排：`/srv/aigc/aigc-infra/docker-compose.yml`（root）
- 服务启动器：`sudo aigc up|down|ps|logs|health`

---

## 1 · 目标与边界

1. **评测先行**：没有 ESML-Bench，不选模型、不训练、不量化。
2. **能源领域垂直**：ESML 语言、四类 DTIL（空气/水/热泵/蒸汽）、物理约束数据集。
3. **可执行验证**：求解收敛/残差、单位覆盖、`esml.dataset/v1` 校验器、测试通过率。
4. **不碰生产**：GPU0/1 的 vLLM 继续服务 ai4e11/ai4e12；训练/评测只在 GPU2/3 或租用 A800。
5. **法律边界**：Claude / GPT / Gemini 等闭源模型的输出**不能**用于商用训练；
   商用数据只用自有代码/文档/测试 + 开源教师 + 执行验证的自产数据。蒸馏权重只做内部评测。

---

## 2 · 目录约定

```text
~/ai4ellm/
  PLAN.md            # 本文件
  PLAN-original.md   # 迁移前的计划留档
  BRIEF.md           # 给新 Claude Code 会话的启动工单
  eval/              # ESML-Bench：任务、夹具、跑分脚本
  data/              # SFT/评测数据（脱敏后）
  train/             # QLoRA/LoRA 配置与训练脚本
  quant/             # AWQ/GPTQ 量化配置与回归
  scripts/           # 环境检查、下载、格式转换、批量评测
  runs/              # 每次训练/评测的原始输出与报告（时间戳命名）
  notes/             # 模型调研、论文、外部经验
```

约定：**所有数字落到 `runs/<日期>-<主题>/`**，不接受只写在聊天里。

---

## 3 · 评测：ESML-Bench

### 3.1 题源

从八个仓的 git 历史抽已合并的 commit / PR，配对应测试，四类题：

1. **Rust 引擎**：编译、求解器、并发、FFI、错误处理。
2. **Julia 客户端**：ESMLKit.jl / MMProps.jl 的协议与边界。
3. **Python 导出器 / 校验器**：`tools/check_dataset.py`、`export_dataset.py`。
4. **ESML 建模与排错**：单位/维度、收敛、PRV/CV、蒸汽/热泵循环。

每类 30–50 题，留 20% 隐藏集；题面只给仓库快照 + 任务描述，判分靠测试。

### 3.2 判分

- 编译通过 + 测试通过（`run == passed + failed`，failed=0）。
- 求解类：`converged=true`、`residual <= residualTolerance`、`unitCoverageComplete=true`。
- 数据集类：`python3 tools/check_dataset.py <dataset>` 退出码 0。
- 记录：`pass@1`、工具调用成功率、一次修好率、平均轮数、token 数、墙钟。

### 3.3 用途

- 选型：base AWQ / Coder390 INT4 / Qwopus3.8-27B-Flash（需量化）/ 自训模型。
- 路由：难任务走强模型，文档/小改走快模型。
- 回归：每次量化、每次微调后必须跑同一套隐藏集。

---

## 4 · 数据

### 4.1 来源（已有）

| 来源 | 内容 | 用法 |
|---|---|---|
| `esml-engine-user3` | Rust 源码、测试、CLI、文档 | 代码 SFT + 评测题 |
| `esml-julia` / `ESMLKit.jl` / `MMProps.jl` | Julia 参考实现与客户端 | 代码 SFT |
| `esml-libraries` | 四个 DTIL 库、scenarios | 领域建模 SFT |
| `esml-books` | 62 卷 / 书系 | 领域知识（优先 RAG，不进权重） |
| 工单/BRIEF/commit/PR | 真实修复过程 | agentic 轨迹 |
| `examples/*-twin/dataset` | `esml.dataset/v1` 样本 | 建模/排错/标定题 |
| Claude Code 会话 | 工具调用序列 | **仅内部评测/风格参考**，法律见 §1.5 |

### 4.2 数据格式

- SFT：`{"messages":[{"role":"system"...},{"role":"user"...},{"role":"assistant"...}], "tools":[...]}`。
- 工具调用轨迹保留：read / edit / bash / test，一步一条 assistant tool_call。
- 每条样本带来源、许可证、是否执行验证、隐藏测试是否通过。

### 4.3 执行验证过滤

- 代码题：在干净 worktree 里跑测试，通过才收。
- 建模题：`esml check` 0 error + `solve` 收敛 + 单位覆盖 100%。
- 数据集题：`check_dataset.py` 通过。
- 通不过的样本进 `data/rejected/`，不删除，便于复查。

---

## 5 · 训练路线

### P0 · 环境与资产（1–2 天）

1. 下载 bf16 基座到 `/data/models/llm/Qwen3.8-27B-bf16/`（约 54GB）。
2. 建训练环境（容器优先），验证 `torch.cuda` 能看到 GPU2/3。
3. 27B QLoRA **100 步冒烟**：记录显存峰值、tokens/s、loss，确认 sm_120 兼容。
4. 产出 `runs/<日期>-env-smoke/` 报告。

### P1 · ESML-Bench v0（2–4 周，可与 P0 并行）

- 先抽 30 题（四类各 7–8 题），跑通判分脚本。
- 用 base AWQ 跑基线，记录分数与失败类型。

### P2 · SFT 数据管线

- 抽取 → 去重 → 许可证过滤 → 执行验证 → 打包。
- 混 20–30% 通用代码/对话数据，防灾难性遗忘。
- 输出 `data/sft-v0/` + 统计报告。

### P3 · QLoRA SFT

- 起点：Qwen3.8-27B bf16，4bit QLoRA，r=32，alpha=32，LR 1e-4 量级。
- 序列长度：先在 2×5090 上 4k–8k；更长上下文租 A800。
- 每轮在 ESML-Bench 隐藏集上回归；掉点就回退。
- 记录：`runs/<日期>-qlora-vN/`（配置、日志、adapter、评测）。

### P4 · 偏好优化（可选）

- 用 review 中「采纳/拒绝」的补丁对做 DPO/ORPO；预算够再考虑 RLOO/GSPO。
- 只在 P3 有稳定收益后做。

---

## 6 · 量化与部署

- 目标格式：AWQ W4A16（与现役 20G 模型一致），校准集用领域文本 + 长上下文样本。
- 量化后必须回 ESML-Bench，掉点 > 2–3% 回退。
- 部署前先起测试端口（如 18112/18113），与现役 18110/18111 做 A/B；
  通过后再切 `docker-compose.yml`，并保留回滚镜像/权重。
- 量化工具：`llm-compressor` / `autoawq` / GPTQ；记录版本与校准集哈希。

---

## 7 · 硬件与成本

### 27B 显存账

| 方法 | 显存需求 | 单卡 5090 32GB | 单卡 A800 80GB |
|---|---|---|---|
| 全参 SFT | ≈430GB | ❌ | ❌（4–8 卡） |
| LoRA，bf16 底座 | ≈60–75GB | ❌ | ✅ seq 4k–8k |
| QLoRA，4bit 底座 | ≈22–28GB | ✅ seq 4k–8k、bs1 | ✅✅ seq 16k–32k |
| 9B QLoRA | ≈6–10GB | ✅ | ✅✅ |

### 选择

- **5090 ×2（GPU2/3）**：环境验证、9B/27B QLoRA 冒烟、短上下文实验、量化校准。
- **租 A800 80GB（¥6/h）**：27B QLoRA 正式训练、长上下文、DPO/RLOO。
  单次 27B QLoRA ¥150–500；全参/RL 4–8 卡 ¥24–48/h。
- 原则：**5090 当实验室，A800 当车间**；数据/配置/评测准备齐再开机，跑完即退。

---

## 8 · 模型调研结论（2026-11-02）

- **AnosVoldigoad11/Qwen3.8-27B-Claude-Opus-Reasoning-Distilled**：LoRA，仅 150 步
  （0.126 epoch），作者自标未收敛；不建议。
- **TeichAI/Qwen3.8-27B-Fable-Distill**：全量 bf16 18 分片≈54GB，ARC/BoolQ 小幅提升，
  无编码/agent 评测；只有 GGUF，vLLM 不便直接用。
- **Jackrong/Qwopus3.8-27B-Flash（V2）**：agent 场景最贴近，1.5M 教师样本筛 10%，
  含 Claude/GPT 重构轨迹；已知 Python 缩进 bug；bf16/GGUF，需自己量化。
- **nerkyor/Qwen3.8-27B-Coder390-…**：自报 GPQA 178/198、MMLU 450/500、LCB 90/100，
  提供 BF16/FP8/NVFP4/INT4，INT4 明确 vLLM 冒烟；工程化最好，但血缘拼装、含 uncensored。
- **barozp/Qwen3.8-27B-Opus-Distill-v2**：修了 v1 的死循环，数据来源诚实，FP8 可用，社区小。
- **khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled**：卡未定稿（占位符），不建议。
- **DavidAU TURBO-Fable-Cold-Fusion / 各类 uncensored merge**：无严格评测，不建议生产。
- **官方没有 Qwen3.8-Coder-27B**，不用等。

判断：多数「Opus 蒸馏」是 TraceInversion 重构轨迹，不是真实思维链；容易学出反复自检。
选型必须用自己的 ESML-Bench，不看模型卡。

---

## 9 · 风险与红线

- **不要碰 GPU0/1**：18110/18111 是 ai4e11/ai4e12 的生产模型；训练脚本显式
  `CUDA_VISIBLE_DEVICES=2,3`。
- **不要 sudo**、不要 `rm -rf`、不要删 `/data/models` 既有模型。
- **大下载先确认磁盘与来源**；HF 直连被墙，用 `hf-mirror.com` 或 `socks5h://localhost:1254`。
- **密钥不落盘/不进聊天**：vLLM key 在 `~/.config/vllm.key`，不要复制到计划或脚本。
- **训练数据许可证**：闭源模型输出只做内部评测；商用训练用自有 + 开源教师 + 执行验证数据。
- **数字实跑**：显存、吞吐、loss、评测分数必须来自 `runs/` 原始输出，不转述。

---

## 10 · 第一周任务清单

1. [ ] 下载 Qwen3.8-27B bf16 基座，校验文件数与大小。
2. [ ] 建训练容器/venv，`nvidia-smi` 确认 GPU2/3 可用，torch 能分配显存。
3. [ ] 27B QLoRA 100 步冒烟（单卡 5090，seq 2k），记录显存峰值/tokens/s/loss。
4. [ ] 跑通 `esml-engine-user3` 的 `scripts/gate.sh`，作为环境基线。
5. [ ] ESML-Bench v0：四类各 7–8 题 + 判分脚本 + base AWQ 基线分数。
6. [ ] 写 `runs/<日期>-env-smoke/report.md` 与 `runs/<日期>-bench-v0/report.md`。
7. [ ] 决定下一批：Coder390 INT4 对比，还是直接 QLoRA。

---

## 11 · 命令速查

```bash
# GPU 状态（只看，不改）
nvidia-smi

# 恢复/停止被训练占用的服务（root）
sudo aigc up voice comfyui-gpu3
docker stop aigc-infra-voice aigc-infra-comfyui-gpu3

# HF 镜像下载（示例）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.8-27B --local-dir /data/models/llm/Qwen3.8-27B-bf16

# 带代理
export ALL_PROXY=socks5h://localhost:1254

# ESML 主仓环境
cd /home/ai4e-hermes/esml-engine-user3
export PATH=$HOME/.cargo/bin:$PATH
bash scripts/gate.sh
```
