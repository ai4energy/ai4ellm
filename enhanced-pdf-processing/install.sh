#!/bin/bash

# 增强版PDF处理库安装脚本

echo "正在安装增强版PDF处理库..."

# 安装所需包
pip install -r requirements.txt

# 检查安装是否成功
if [ $? -eq 0 ]; then
    echo "安装成功完成!"
    echo ""
    echo "使用增强PDF处理库的方法："
    echo "1. 在输入目录中准备PDF文件"
    echo "2. 运行: python main.py -i /path/to/input -o /path/to/output"
    echo "3. 查看更多选项: python main.py --help"
else
    echo "安装失败。请确保已安装pip和Python。"
fi