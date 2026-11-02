# P0 §1 环境确认报告（2026-11-02）

工作目录：`/home/ai4e-hermes/ai4ellm`
脚本：`scripts/00_env_check.sh`（只读；不 stop 容器、不改配置）

## 1. 原始命令行

```bash
bash scripts/00_env_check.sh
```

脚本内部执行的命令：

```bash
nvidia-smi
docker images
ls -lh /data/models/llm
df -h /data
docker ps -a
ssh -i ~/.ssh/id_ed25519.mingtaoli root@localhost \
  'docker images --format "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | sort'
ssh -i ~/.ssh/id_ed25519.mingtaoli root@localhost \
  'docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"'
```

## 2. 原始输出

### 2.1 nvidia-smi

```text
Mon Nov  2 08:29:23 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        On  |   00000000:27:00.0 Off |                  N/A |
| 82%   78C    P1            540W /  540W |   30628MiB /  32607MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5090        On  |   00000000:38:00.0 Off |                  N/A |
| 71%   74C    P1            508W /  540W |   30524MiB /  32607MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 5090        On  |   00000000:A8:00.0 Off |                  N/A |
| 40%   25C    P8             12W /  540W |       2MiB /  32607MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 5090        On  |   00000000:B8:00.0 Off |                  N/A |
| 40%   28C    P8             17W /  540W |       2MiB /  32607MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2278414      C   VLLM::EngineCore                      30618MiB |
|    1   N/A  N/A         2262180      C   VLLM::EngineCore                      30514MiB |
+-----------------------------------------------------------------------------------------+
```

### 2.2 系统 docker 镜像（root daemon，只读）

```text
aigc-base:torch210	c8a694ef7529	18.8GB
aigc-base:torch210-flash	f682d6c6c8ee	21.5GB
aigc-base:torch211	4fca844bd14a	20.7GB
aigc-base:torch211-cu130	385e48ea8ac1	20.9GB
aigc-base:torch211-cu130-flash	c68c6eaf967a	23GB
aigc-base:torch211-flash	f3e064897bba	22.7GB
aigc-base:torch213-cu134	55d1cb978207	22.5GB
aigc-base:torch213-cu134-flash	ab62b4d7c8d4	23.2GB
aigc-comfyui:cu130	7dc161f8ee4c	25.5GB
aigc-comfyui:cu130-nf	14efada8b66d	23.4GB
aigc-comfyui:cu134	20ee613b51eb	25.2GB
aigc-comfyui-fix:interim	d21b1a548b1b	24.1GB
aigc-comfyui:latest	b3f063b95343	24.2GB
aigc-fara:latest	8e901b7e3623	30.8GB
aigc-manim:latest	343bcaa9f94d	3.87GB
aigc-sdwebui:latest	3217077ab1b2	23.7GB
aigc-sglang:cu134	a0e1af09b8b9	28.2GB
aigc-test:cu130	385e48ea8ac1	20.9GB
aigc-vllm:cu130	f2c472cff263	27.3GB
aigc-vllm:cu134	68126dacc214	27GB
aigc-vllm:latest	5c2fa189152b	23.4GB
aigc-voice:cu134	23fb883a5b5a	24.3GB
aigc-voice:latest	ce6b6b76fc7b	24.4GB
alpine:3.20	d9e853e87e55	12.2MB
alpine:latest	28bd5fe8b56d	13MB
ghcr.io/ggml-org/llama.cpp:server-cuda	5268283a8d65	6.99GB
golang:1.23-alpine	383395b794df	370MB
golang:1.24-alpine	8bee1901f1e5	395MB
golang:1.25-alpine	1ae0735f00da	329MB
hello-world:latest	5e2309035332	25.9kB
jupyter/datascience-notebook:latest	476c6e673e7d	8.31GB
jupyter/datascience-notebook:python-3.11	476c6e673e7d	8.31GB
mcr.microsoft.com/playwright/python:v1.53.0	648e2d7cb413	3.58GB
mcr.microsoft.com/playwright:v1.52.0-noble	a021500a801b	3.59GB
nginx:alpine	df221db836e1	94.4MB
node:22-alpine	0a7108bf6c7b	238MB
nvidia/cuda:13.0.0-base-ubuntu22.04	c08986c5ea16	562MB
nvidia/cuda:13.4.2-devel-ubuntu24.04	7005688f5eaa	13.2GB
playwright-test:latest	b4115e8a395a	3.87GB
postgres:16-alpine	57c72fd2a128	420MB
python:3.12-slim	a6e34c598f24	179MB
quay.io/jupyter/datascience-notebook:2026-07-20	6db9d897f50a	9.79GB
quay.io/jupyter/pytorch-notebook:cuda12-2026-07-20	25c9d8c64029	16.4GB
teaching-api:latest	cf3b1c106713	65.2MB
teaching-jupyterhub:latest	3fd4c0f213ad	1.13GB
teaching-user:latest	d52e92243ad1	35GB
```

### 2.3 系统 docker 运行中容器（root daemon，只读）

```text
aigc-infra-vllm-qwen38-gpu0	aigc-vllm:cu134	Up 30 minutes	8121/tcp, 0.0.0.0:18110->8120/tcp, [::]:18110->8120/tcp
aigc-infra-vllm-qwen38	aigc-vllm:cu134	Up 33 minutes	8121/tcp, 0.0.0.0:18111->8120/tcp, [::]:18111->8120/tcp
aigc-nginx	nginx:alpine	Up 2 weeks	
teaching-jupyterhub	teaching-jupyterhub:latest	Up 3 weeks	8000/tcp, 0.0.0.0:9999->9999/tcp, [::]:9999->9999/tcp
teaching-api	teaching-api:latest	Up 2 weeks	0.0.0.0:8100->8100/tcp, [::]:8100->8100/tcp
teaching-nginx	4a73073bd557	Up 3 weeks	0.0.0.0:8899->80/tcp, [::]:8899->80/tcp
teaching-db	postgres:16-alpine	Up 2 weeks (healthy)	0.0.0.0:5433->5432/tcp, [::]:5433->5432/tcp
```

### 2.4 既有模型资产

```text
total 24K
drwxrwxr-x 3 ai4e-user3 ai4energy  4.0K Apr 22  2026 google
drwxrwxr-x 2 ai4e-user3 ai4energy  4.0K Jul  4 08:59 microsoft
drwxrwxr-x 5 ai4e-user3 ai4energy  4.0K Jun 22 00:40 QuantTrio
drwxrwxr-x 2 ai4e-user3 ai4e-user3  4.0K Sep 15 13:58 Qwen3.8-27B
drwxrwxr-x 2 ai4e-user3 ai4e-user3  4.0K Oct 11 08:26 Qwen3.8-27B-AWQ-INT4
drwxrwxr-x 5 ai4e-user3 ai4energy  4.0K Jul  7 08:48 ui-tars-1.5-7b
```

### 2.5 磁盘

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb1        15T  5.6T  8.2T  41% /data
```

## 3. 结论（按判据）

```text
GPU2 used=2 MiB (require < 100)
GPU3 used=2 MiB (require < 100)
RESULT: GPU2/3 free -> PASS
```

- GPU2/3 显存均 2 MiB，满足 <100 MiB；GPU0/1 仍由 `VLLM::EngineCore` 占用，未做任何操作。
- 训练镜像 `aigc-base:torch213-cu134`、`aigc-vllm:cu134` 存在；训练容器将使用 `--gpus '"device=2,3"'`。
- 原始文件：`01_nvidia-smi.txt` … `08_summary.txt`（同目录）。
