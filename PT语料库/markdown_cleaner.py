import os
import re

def clean_markdown(content):
    """
    清洗 Markdown 内容
    - 删除图片和链接标记
    - 移除无关元数据（如版权信息、邮箱等）
    - 删除目录部分（通过连续的章节标题判断）
    - 删除多余空行和尾部空格
    - 删除方括号及其内容，以及参考文献
    - 删除类似 Bopp(1984), Brown（1983） 的参考文献
    """
    cleaned_content = []
    is_in_toc = False  # 标记是否处于目录部分
    previous_line = None  # 用于追踪上一行是否为空

    for line in content:
        # 去除行尾空格
        line = line.rstrip()

        # 检查是否进入目录部分
        if line.lower().startswith("contents") or "目录" in line:
            is_in_toc = True
            continue
        # 检查是否结束目录部分（假设目录以空行或某些章节开始）
        if is_in_toc and (not line.strip() or line.startswith("#")):
            is_in_toc = False
        # 跳过目录部分
        if is_in_toc:
            continue

        # 跳过包含图片和链接的行
        if "![" in line or "](" in line:
            continue
        # 跳过无关的元数据行，如版权、邮箱、网址等
        if "copyright" in line.lower() or "email" in line.lower() or "www." in line.lower():
            continue

        # 删除方括号及其内容
        line = re.sub(r"\[.*?\]", "", line)

        # 删除类似 "Bopp(1984)", "Brown（1983）", "Brown和Phillips（1989，1991）" 的内容
        line = re.sub(r"\b[A-Za-z\u4e00-\u9fa5]+(?:和[A-Za-z\u4e00-\u9fa5]+)?\s*[（(]\d{4}(?:，\d{4})*[）)]", "", line)

        # 删除多余的空行（连续多个空行只保留一个）
        if not line.strip():
            if previous_line is None or not previous_line.strip():
                continue  # 当前行和上一行都是空行时跳过
            else:
                cleaned_content.append("")  # 保留一个空行
        else:
            cleaned_content.append(line)  # 添加非空行

        # 更新上一行内容
        previous_line = line

    return cleaned_content

def remove_specific_patterns(content):
    """
    去除特定模式，包括表格、图表引用以及特定标记内容。
    - 删除类似 "表1-1" 和 "图10" 的表图说明
    - 删除类似 "Table 1-1" 和 "Figure 10" 的表图说明
    - 删除类似 "https://chatgpt.com/c/676ebed8-50b4-8000-bf15-c0a1f31cad3f" 的 URL
    - 删除重复性标记，如 "西安交通大学XIANJIAOTONGUNIVERSITY"
    """
    cleaned_content = []
    for line in content:
        # 去除表格和图表引用（中英文）
        line = re.sub(r"表\s?\d+-?\d*", "", line)
        line = re.sub(r"图\s?\d+-?\d*", "", line)
        line = re.sub(r"Table\s?\d+-?\d*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Figure\s?\d+-?\d*", "", line, flags=re.IGNORECASE)

        # 去除 URL
        line = re.sub(r"https?://\S+", "", line)

        # 去除特定重复标记
        line = re.sub(r"西安交通大学XIANJIAOTONGUNIVERSITY", "", line, flags=re.IGNORECASE)

        # 去除多余的空白
        line = line.strip()
        if line:  # 只保留非空行
            cleaned_content.append(line)
    return cleaned_content

def remove_garbled_characters(content):
    """
    删除常见乱码符号和不可见字符
    - 删除不可打印字符
    - 删除类似 "�" 或其他特殊符号
    """
    cleaned_content = []
    for line in content:
        # 删除不可打印字符（Unicode控制字符）
        line = re.sub(r"[^\x20-\x7E\u4E00-\u9FA5]", "", line)

        # 去除常见乱码符号，如"�"
        line = line.replace("�", "")

        # 删除多余空白
        line = line.strip()
        if line:  # 只保留非空行
            cleaned_content.append(line)
    return cleaned_content

def clean_references(content):
    """
    清理引用内容，如文献标记和类似 (Smith et al., 2020) 的模式。
    - 删除文献引用
    """
    cleaned_content = []
    for line in content:
        # 去除学术引用模式，如 "(Smith et al., 2020)"
        line = re.sub(r"\([A-Za-z\u4e00-\u9fa5]+\s+et\s+al\.,\s+\d{4}\)", "", line)
        # 去除中英文混合引用模式 "[10]"
        line = re.sub(r"\[[0-9]+\]", "", line)
        line = line.strip()
        if line:
            cleaned_content.append(line)
    return cleaned_content

def clean_and_extract_markdown(content):
    """
    提取 Markdown 中标题和其下的文本，并清洗标题和正文
    - 提取标题及其下的文本
    - 删除标题及其下的文本小于等于 20 个字符的部分
    - 删除指定标题（如“前言”）和包含关键词（如“思考题”）的段落
    - 将文本拼接成长段格式，不换行
    """
    # 先执行基本清洗
    cleaned_content = clean_markdown(content)
    cleaned_content = remove_specific_patterns(cleaned_content)  # 移除特定模式
    cleaned_content = clean_references(cleaned_content)  # 清理文献引用
    cleaned_content = remove_garbled_characters(cleaned_content)
    result = []
    current_title = None
    current_text = []
    skip_section = False  # 标记是否跳过当前段落

    for line in cleaned_content:
        # 判断是否是标题（假设标题为 Markdown 标题格式，如 # 或 ## 开头）
        if line.startswith("#"):
            # 如果已有标题和文本，进行处理
            if current_title and current_text:
                combined_text = "".join(current_text).strip()
                # 检查标题下的文本长度是否超过 50
                if not skip_section and len(re.findall(r"[\u4e00-\u9fa5\w]+", combined_text)) > 50:
                    result.append(f"{current_title}\n{combined_text}\n")

            # 更新当前标题，清空文本，并检查是否需要跳过
            current_title = line.strip()
            current_text = []
            skip_section = any(keyword in current_title for keyword in ["前言", "思考题", "参考文献", "ACKNOWLEDGEMENTS", "Conference Papers", "TABLE OF CONTENTS", "LIST OF FIGURES", " LIST OF TABLES"
                                                                        , "Fundamental Concepts in Heterogeneous Catalysis", " 图书在版编目CIP数据", "例题","Magnetism"])
        else:
            # 累积当前标题下的文本
            if not skip_section and line.strip():  # 忽略空行
                current_text.append(line.strip())

    # 处理最后一个标题和文本
    if current_title and current_text:
        combined_text = "".join(current_text).strip()
        if not skip_section and len(re.findall(r"[\u4e00-\u9fa5\w]+", combined_text)) > 20:
            result.append(f"{current_title}\n{combined_text}\n")

    return result
def process_specific_file(file_path, output_file):
    """
    处理单个指定文件并保存清洗结果
    - file_path: 输入文件路径
    - output_file: 输出文件路径
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    # 清洗并提取内容
    cleaned_content = clean_and_extract_markdown(content)

    # 输出结果到目标文件
    if cleaned_content:  # 只有当内容非空时才保存文件
        with open(output_file, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write("\n".join(cleaned_content))
        print(f"Processed and saved: {output_file}")
    else:
        print(f"Skipped empty result for: {file_path}")

def process_markdown_files(input_folder, output_folder):
    """
    处理文件夹内的所有 Markdown 和文本文件并保存成 .txt 文件
    - 读取每个 Markdown 或文本文件
    - 清洗并提取标题及其下的文本
    - 清洗后保存为 .txt 文件
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_file_path) and (filename.endswith('.md') or filename.endswith('.txt')):
            with open(input_file_path, 'r', encoding='utf-8') as file:
                content = file.readlines()

            # 清洗并提取内容
            cleaned_content = clean_and_extract_markdown(content)

            # 输出结果到目标文件夹
            if cleaned_content:  # 只有当内容非空时才保存文件
                output_file_path = os.path.join(output_folder, filename.replace('.md', '.txt').replace('.txt', '_cleaned.txt'))
                with open(output_file_path, 'w', encoding='utf-8') as cleaned_file:
                    cleaned_file.write("\n".join(cleaned_content))
                print(f"Processed and saved: {output_file_path}")
            else:
                print(f"Skipped empty result for: {input_file_path}")

# 示例：读取 input_folder 内的文件并输出到 output_folder
#input_folder = r"G:\yuliaoku\books2\markdown"
#output_folder = r"G:\yuliaoku\books2\cleaned_markdown_files_txt"

#process_markdown_files(input_folder, output_folder)
# 示例：处理单个文件
file_path = r"G:\yuliaoku\1-quan\课程\kecheng\未去重.txt"
output_file = r"G:\yuliaoku\1-quan\课程\kecheng\未去重_cleaned.txt"
process_specific_file(file_path, output_file)