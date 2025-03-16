# 接入 Xinference 部署的本地模型

## 部署 Xinference
1. **安装 Xinference：**
   ```bash
   $ pip install "xinference[all]"
   ```

2. **启动 Xinference：**
   ```bash
   $ xinference-local
   ```
   启动后默认端点为：`http://127.0.0.1:9997`，端口号默认为 `9997`。  
   - 仅本机访问：默认设置。
   - 如果非本地客户端访问，请使用参数：`-H 0.0.0.0`。

## 配置 Dify 容器网络
确保 Dify 容器可以访问 Xinference 的端点：
- 使用宿主机 IP 地址替代 `localhost`。

## 创建并部署模型
1. 打开浏览器访问 `http://127.0.0.1:9997`。
2. 选择需要部署的模型和规格。
3. 确认硬件兼容性后完成模型部署。

## 获取模型 UID
部署完成后，从模型页面获取对应 **模型 ID**，例如：`2c886330-8849-11ee-9518-43b0b8f40bea`。

## 在 Dify 中接入模型
1. 进入 **设置 > 模型供应商 > Xinference**。
2. 填写以下信息：
   - **模型名称：** `vicuna-v1.3`
   - **服务器 URL：** `http://<Machine_IP>:9997`  
     替换 `<Machine_IP>` 为你的机器 IP 地址。
   - **模型 UID：** `2c886330-8849-11ee-9518-43b0b8f40bea`

3. 点击 **保存**，完成配置，即可使用该模型。



