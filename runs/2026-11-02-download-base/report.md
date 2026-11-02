# P0 §2 bf16 基座下载报告（2026-11-02）

## 1. 来源与命令

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  .venv/bin/hf download Qwen/Qwen3.8-27B \
  --local-dir /data/models/llm/Qwen3.8-27B-bf16 --max-workers 4
```

- 直连 huggingface.co 超时（`curl` 10s timeout），hf-mirror.com 返回 200。
- 下载日志：`download.log`（同目录）。
- 不删除 `/data/models/llm` 下任何既有目录；目标目录为新建。

## 2. 下载前 API 核对

```text
$ jq -r '[.id, .sha, (.siblings|length|tostring)] | @tsv' <api json>
Qwen/Qwen3.8-27B	1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0	32

$ jq '[.siblings[].size // 0] | add'
55586114863
```

- 18 个 `model-*-of-00018.safetensors`，`model.safetensors.index.json` 存在。
- shard 1 的 safetensors header 抽样：`header_bytes 45392 tensors 392 dtypes {'BF16': 392}`。
- index：`metadata.total_size=55562855904.0`，`weight_map` 1199 个 tensor。

## 3. 下载后原始校验输出（verify.txt）

```text
# verify (date: 2026-11-02T08:41:35+00:00)

## shard count
safetensors_shards=18 (expect 18)

## total bytes of safetensors shards
safetensors_bytes=55563006776

## index
index_exists=yes
index_total_size=55562855904.0 tensors=1199

## disk
52G	/data/models/llm/Qwen3.8-27B-bf16
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1        15T  5.7T  8.1T  42% /data
```

校验结论：
- 分片数 = 18，与仓库清单一致；
- safetensors 实际字节 55,563,006,776 比 index 的 tensor 总量 55,562,855,904 多 150,872 B，即 18 个 safetensors 文件头（JSON header）的开销，属正常；
- `model.safetensors.index.json`、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` 均在；
- 下载退出码 0（`download.log` 末行 `# exit: 0`）。

## 4. 说明

`config.json` 的架构是 `Qwen3_5ForConditionalGeneration`（同时带 `text_config` 与 `vision_config`，
权重键前缀 `model.language_model.*`），训练时必须用该原生类加载，不能用 `Qwen3_5ForCausalLM`
（键前缀不匹配）。
