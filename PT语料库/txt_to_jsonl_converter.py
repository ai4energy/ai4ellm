import os
import json


def txt_to_jsonl(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有txt文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_name = file_name.replace(".txt", ".jsonl")
            output_file_path = os.path.join(output_folder, output_file_name)

            with open(input_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            jsonl_data = []
            section_title = None
            content = []

            for line in lines:
                if line.startswith("#"):  # 标题行
                    if section_title:
                        jsonl_data.append({"section": section_title, "content": "".join(content).strip()})
                    section_title = line.strip("#").strip()
                    content = []
                else:
                    content.append(line)

            # 添加最后的部分
            if section_title:
                jsonl_data.append({"section": section_title, "content": "".join(content).strip()})

            with open(output_file_path, 'w', encoding='utf-8') as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Converted: {file_name} -> {output_file_name}")


# 设置输入和输出文件夹路径
input_folder = r"G:\yuliaoku\books2\jsonl" # 替换为您的输入文件夹路径
output_folder =r"G:\yuliaoku\books2\jsonl"  # 替换为您的输出文件夹路径

# 执行转换
txt_to_jsonl(input_folder, output_folder)