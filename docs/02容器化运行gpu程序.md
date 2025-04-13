# 容器化运行GPU程序

## 安装 Docker

参考 [Docker 官方文档](https://docs.docker.com/engine/install/ubuntu/) 进行安装。

### 1. **卸载旧版本或不需要的软件包**

在安装 Docker 前，先卸载旧版本或不需要的软件包：

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

### 2. **安装必要的软件**

更新包索引并安装必要的软件包：

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
```

### 3. **添加 Docker APT 仓库**

1. 创建 APT 密钥目录：

   ```bash
   sudo install -m 0755 -d /etc/apt/keyrings
   ```

2. 下载并安装 Docker 官方的 GPG 密钥：

   ```bash
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc
   ```

3. 将 Docker 仓库添加到 APT 源中：

   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
     $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

### 4. **安装 Docker Engine**

更新 APT 包索引并安装 Docker：

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 5. **启动 Docker 服务**

在安装完 Docker 后，启动 Docker 服务：

```bash
sudo systemctl start docker
```

### 6. **验证 Docker 安装**

验证 Docker 是否正确安装并正常运行：

```bash
sudo docker run hello-world
```

如果 Docker 安装成功，你会看到类似如下的信息，表示 Docker 已经启动并正常工作：

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

## 安装 NVIDIA Container Toolkit

参考 [NVIDIA Container Toolkit 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 进行安装。

### 1. **配置生产环境仓库**

首先，添加 NVIDIA Container Toolkit 的 GPG 密钥并配置 APT 仓库：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 2. **更新软件包列表**

更新 APT 软件包索引：

```bash
sudo apt-get update
```

### 3. **安装 NVIDIA Container Toolkit**

安装 NVIDIA Container Toolkit 软件包：

```bash
sudo apt-get install -y nvidia-container-toolkit

### 4. **配置 Docker**

使用如下的命令配置Docker使用nvidia运行时

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 5. **运行 Docker 中的示例工作负载**

安装和配置好 NVIDIA Container Toolkit 以及 NVIDIA GPU 驱动后，可以通过 Docker 运行支持 GPU 的工作负载。

运行一个包含 CUDA 的样例容器：

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

这条命令将启动一个 **Ubuntu** 容器，并运行 `nvidia-smi` 命令，显示当前 GPU 的状态。如果安装成功，你将看到 NVIDIA GPU 驱动的详细信息，表示 Docker 已正确配置为使用 GPU。
