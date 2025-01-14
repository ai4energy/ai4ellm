import os

# 设置输入文件夹和输出文件路径
input_folder = r"G:\yuliaoku\1-quan\课程\kecheng" # 替换为您的文件夹路径
output_file = r"G:\yuliaoku\1-quan\课程\kecheng/未去重.txt"  # 替换为合并后的文件路径

# 创建或清空输出文件
with open(output_file, "w", encoding="utf-8") as outfile:
    # 遍历文件夹中的所有 .txt 文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_folder, file_name)
            print(f"正在处理文件: {file_path}")
            # 读取每个文件并写入输出文件
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # 添加换行符以分隔文件内容

print(f"所有 .txt 文件已合并到 {output_file}")