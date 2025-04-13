import os
from tqdm import tqdm

def allin(input_folder, output_file):
    # 获取文件夹中所有 .txt 文件
    txt_files = [file for file in os.listdir(input_folder) if file.endswith(".txt")]

    # 创建或清空输出文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        # 使用 tqdm 显示进度条
        for file_name in tqdm(txt_files, desc="txt-merge", unit="file"):
            file_path = os.path.join(input_folder, file_name)
            # 读取每个文件并写入输出文件
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # 添加换行符以分隔文件内容

    print(f"所有 .txt 文件已合并到 {output_file}")


