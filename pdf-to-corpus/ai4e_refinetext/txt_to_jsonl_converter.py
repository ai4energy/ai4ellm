import os
import json
from tqdm import tqdm

def txt_to_jsonl(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    input_folder = os.path.dirname(input_folder)
    # 获取文件夹中的所有txt文件
    txt_files = [file for file in os.listdir(input_folder) if file.endswith(".txt")]

    # 使用tqdm显示处理进度
    for file_name in tqdm(txt_files, desc="Processing files"):
        input_file_path = os.path.join(input_folder, file_name)
        output_file_name = file_name.replace(".txt", ".jsonl")
        output_file_path = os.path.join(output_folder, output_file_name)

        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            jsonl_data = []
            section_title = None
            content = []

            for line in lines:
                line = line.strip()
                if line.startswith("#"):  # 标题行
                    if section_title:  # 如果之前已经有部分内容，保存到jsonl数据
                        jsonl_data.append({"section": section_title, "content": "".join(content).strip()})
                    section_title = line.strip("#").strip()  # 更新标题
                    content = []
                elif line:  # 非空行添加到内容中
                    content.append(line)

            # 添加最后的部分
            if section_title:
                jsonl_data.append({"section": section_title, "content": "".join(content).strip()})

            # 将数据写入jsonl文件
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception as e:
            continue  # 发生错误时跳过当前文件


