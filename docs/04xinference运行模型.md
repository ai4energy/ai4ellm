# xinference 运行模型

## 一、Xinference 简介
Xinference 是一款开源的分布式模型推理框架，由星环科技研发团队开发维护，主要特性包括：

1. **多模态模型支持**
   - 支持大语言模型（LLM）、视觉模型（Vision）、语音模型（Audio）等
   - 兼容主流格式：GGML、PyTorch、HuggingFace 等

2. **分布式推理能力**
   - 支持多节点集群部署
   - 可自动进行负载均衡
   - 支持水平扩展应对高并发场景

3. **高效性能优化**
   - 集成 CUDA/ROCm 加速
   - 支持模型量化（4-bit/8-bit）
   - 提供动态批处理功能

4. **开放接口**
   - 提供 RESTful API
   - 兼容 OpenAI API 协议
   - 支持 Python/Java/Go 客户端

5. **生态集成**
   - 支持 LangChain 等应用框架
   - 可对接 Prometheus 监控系统
   - 提供 Grafana 监控面板

## 二、物理机部署方案

### 1. 环境准备
```bash
# 创建虚拟环境
python -m venv xinference-env
source xinference-env/bin/activate

# 安装核心包（选择对应版本）
pip install xinference[all]  # 全量安装
```

### 2. 服务启动
```bash
# 启动单节点服务
xinference-local --host 0.0.0.0 --port 9997 

# 启用GPU加速示例
XINFERENCE_DEVICE=cuda xinference-local --host 0.0.0.0 --port 9997
```

### 3. 模型管理
```python
from xinference.client import Client

client = Client("http://localhost:9997")

# 部署Llama-2模型
model_uid = client.launch_model(
    model_name="llama-2-chat",
    model_size_in_billions=13,
    quantization="q4_0"
)

# 查看运行中模型
print(client.list_models())
```

### 4. 接口调用
```python
# 通过Python SDK调用
model = client.get_model(model_uid)
response = model.generate("如何做西红柿炒鸡蛋？")

# 兼容OpenAI API
curl http://localhost:9997/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-chat",
    "prompt": "讲一个武侠故事",
    "temperature": 0.7
  }'
```

### 5. 运维管理
```bash
# 监控指标查看
curl http://localhost:9997/metrics

# 服务关闭
pkill -f xinference-local
```

## 三、Docker 部署方案

### 1. 快速启动
```bash
# 拉取官方镜像
docker pull xprobe/xinference:latest

# 启动容器（CPU版）
docker run -d -p 9997:9997 --name xinference xprobe/xinference:latest

# GPU版本启动
docker run -d --gpus all -p 9997:9997 --name xinference-gpu xprobe/xinference:latest
```

### 2. 模型预置
```bash
# 进入容器环境
docker exec -it xinference /bin/bash

# 容器内下载模型
xinference download --model-name llama-2-chat --model-size 13B --quantization q4_0
```

### 3. 持久化部署
```bash
# 创建数据卷
docker volume create xinference-models

# 挂载模型存储
docker run -d -p 9997:9997 \
  -v xinference-models:/root/.xinference \
  --name xinference xprobe/xinference:latest
```

### 4. 集群部署示例
```yaml
# docker-compose.yml 示例
version: '3'
services:
  supervisor:
    image: xprobe/xinference:latest
    command: xinference-supervisor --host 0.0.0.0 --port 9997
    ports:
      - "9997:9997"
  
  worker1:
    image: xprobe/xinference:latest
    command: xinference-worker --supervisor-host supervisor --port 9998
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
```
### 5. docker-compose文件

```shell
services:
  xinference:
    image: xprobe/xinference:v1.3.1
    ports:
      - "9997:9997"
    volumes:
      - /srv/xinference/models:/models
    environment:
      - XINFERENCE_HOME=/models
      - XINFERENCE_MODEL_SRC=modelscope # or huggingface
      - NVIDIA_VISIBLE_DEVICES=all # if using GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["xinference-local", "-H", "0.0.0.0"]
```

把以前按照大语言模型食用指南下载的模型拷贝到models目录。在xinference的网页界面上设置自定义模型，就可以加载了。


## 四、方案对比

| 特性               | 物理机部署          | Docker 部署           |
|--------------------|-------------------|---------------------|
| 启动速度           | 中等              | 快速                |
| 资源隔离           | 无               | 完整命名空间隔离      |
| GPU支持           | 需手动配置驱动     | 需nvidia-container支持 |
| 部署复杂度         | 中               | 低                  |
| 多版本管理         | 依赖虚拟环境       | 镜像隔离            |
| 适用场景           | 生产环境          | 开发/测试环境       |

## 五、最佳实践建议
1. **模型选择策略**
   - 测试环境使用 7B 以下小模型
   - 生产环境推荐 13B 及以上模型
   - 根据硬件选择量化版本（GPU优先q4_0，CPU考虑q8_0）

2. **性能调优技巧**
   - 启用批处理：`max_tokens=512`
   - 设置温度参数：`temperature=0.7`
   - 使用流式响应：`stream=True`

3. **运维监控**
   ```bash
   # Prometheus 配置示例
   - job_name: 'xinference'
     metrics_path: '/metrics'
     static_configs:
       - targets: ['xinference-host:9997']
   ```

4. 安全防护
   - 启用 API Key 认证
   - 配置 Nginx 反向代理
   - 设置防火墙规则

> **注意事项**：生产环境部署建议使用 Kubernetes 进行容器编排，推荐结合 Redis 实现请求队列管理。对于超大规模模型（70B+），需要配置多GPU并行策略。