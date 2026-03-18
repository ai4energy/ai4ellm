# 系统安装和cuda配置

## 准备 Ubuntu 24.04 环境

### 1. **下载 Ubuntu 24.04 ISO 文件**

   - 访问 [Ubuntu 官方下载页面](https://ubuntu.com/download/desktop)。
   - 选择 **Ubuntu 24.04 LTS** 版本并下载 ISO 文件。

### 2. **制作启动 U 盘**

   使用工具将 Ubuntu 24.04 的 ISO 文件制作成可启动的 USB 驱动器。推荐使用 **Rufus** 工具来制作启动盘（Windows 系统下）。

   - **下载 Rufus**：[Rufus 官网](https://rufus.ie/)。
   - 插入一个空的 U 盘（建议至少 8GB 容量）。
   - 打开 Rufus 工具，选择下载好的 Ubuntu 24.04 ISO 文件。
   - 在 **"Device"** 中选择你的 U 盘设备。
   - 确保选择 **"Partition Scheme"** 为 **GPT**，**"File System"** 为 **FAT32**。
   - 点击 **"Start"** 开始制作启动盘。
   
   > 注意：制作启动 U 盘会清空 U 盘上的所有数据，务必备份重要文件。

### 3. **从 U 盘启动 Ubuntu 安装程序**

   - 将制作好的启动 U 盘插入目标电脑的 USB 插口。
   - 重启电脑，并进入 BIOS 设置（通常按 **F2**、**F12** 或 **DEL**，具体取决于你的主板和厂商）。
   - 在 BIOS 设置中，找到 **Boot** 或 **Boot Order** 选项，确保 U 盘是首选启动设备。
   - 保存设置并退出 BIOS，电脑将从 U 盘启动，进入 Ubuntu 安装程序。

### 4. **安装 Ubuntu 24.04**

   - 启动时，选择 **"Install Ubuntu"** 选项进入安装界面。
   - 按照安装向导的步骤进行：
     1. **选择语言**：选择你需要的语言。
     2. **网络连接**：如果可以，连接到 Wi-Fi 或有线网络，系统可能会下载更新。
     3. **分区设置**：选择 **"Erase disk and install Ubuntu"**，这会清空硬盘并自动分区。若你有特定的分区需求，可以选择 **"Something else"** 进行自定义分区。
     4. **时区选择**：选择你所在的时区。
     5. **键盘布局**：选择适合你的键盘布局（默认选择一般是合适的）。
     6. **用户设置**：输入你的用户名、计算机名和密码。

   - 安装过程会持续几分钟到十几分钟不等，具体时间取决于硬件配置。
   - 安装完成后，按照提示重新启动系统，并拔掉 U 盘。

### 5. **第一次启动和系统设置**

   - 重启后，进入 Ubuntu 24.04 系统。
   - 使用用户名密码登录进入系统。

## 安装 NVIDIA 驱动和 CUDA Toolkit

访问 [CUDA Toolkit 官方页面](https://developer.nvidia.com/cuda-toolkit)，选择Download。
   - 选择 **Linux** -> **x86_64** -> **Ubuntu** -> **24.04** -> **deb(network)**。
   - 按照网页上的说明进行操作。
```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

下面进行一下解释：

1. **下载并安装 CUDA 存储库的密钥包**

   首先下载 CUDA 的官方密钥包：

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
   ```

2. **安装密钥包**

   使用 `dpkg` 命令安装下载的密钥包：

   ```bash
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   ```

3. **更新 apt 包索引**

   更新 apt 包索引，以便新添加的 CUDA 存储库生效：

   ```bash
   sudo apt-get update
   ```

4. **安装 CUDA Toolkit**

   安装 **CUDA Toolkit 12.8**：

   ```bash
   sudo apt-get -y install cuda-toolkit-12-8
   ```
