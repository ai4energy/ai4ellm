import os
import re
from tqdm import tqdm

def clean_markdown(content):
    # 内容清洗代码保持不变
    cleaned_content = []
    is_in_toc = False
    previous_line = None
    for line in content:
        line = line.rstrip()
        if line.lower().startswith("contents") or "目录" in line:
            is_in_toc = True
            continue
        if is_in_toc and (not line.strip() or line.startswith("#")):
            is_in_toc = False
        if is_in_toc:
            continue
        if "![" in line or "](" in line:
            continue
        if "copyright" in line.lower() or "email" in line.lower() or "www." in line.lower():
            continue
        line = re.sub(r"\[.*?\]", "", line)
        line = re.sub(r"\b[A-Za-z\u4e00-\u9fa5]+(?:和[A-Za-z\u4e00-\u9fa5]+)?\s*[（(]\d{4}(?:，\d{4})*[）)]", "", line)
        if not line.strip():
            if previous_line is None or not previous_line.strip():
                continue
            else:
                cleaned_content.append("")
        else:
            cleaned_content.append(line)
        previous_line = line
    return cleaned_content

def remove_specific_patterns(content):
    # 内容清洗代码保持不变
    cleaned_content = []
    for line in content:
        line = re.sub(r"表\s?\d+-?\d*", "", line)
        line = re.sub(r"图\s?\d+-?\d*", "", line)
        line = re.sub(r"Table\s?\d+-?\d*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"Figure\s?\d+-?\d*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"西安交通大学XIANJIAOTONGUNIVERSITY", "", line, flags=re.IGNORECASE)
        line = line.strip()
        if line:
            cleaned_content.append(line)
    return cleaned_content

def remove_garbled_characters(content):
    # 内容清洗代码保持不变
    cleaned_content = []
    for line in content:
        line = re.sub(r"[^\x20-\x7E\u4E00-\u9FA5]", "", line)
        line = line.replace("�", "")
        line = line.strip()
        if line:
            cleaned_content.append(line)
    return cleaned_content

def clean_references(content):
    # 内容清洗代码保持不变
    cleaned_content = []
    for line in content:
        line = re.sub(r"\([A-Za-z\u4e00-\u9fa5]+\s+et\s+al\.,\s+\d{4}\)", "", line)
        line = re.sub(r"\[[0-9]+\]", "", line)
        line = line.strip()
        if line:
            cleaned_content.append(line)
    return cleaned_content

def clean_and_extract_markdown(content):
    cleaned_content = clean_markdown(content)
    cleaned_content = remove_specific_patterns(cleaned_content)
    cleaned_content = clean_references(cleaned_content)
    cleaned_content = remove_garbled_characters(cleaned_content)
    result = []
    current_title = None
    current_text = []
    skip_section = False
    for line in cleaned_content:
        if line.startswith("#"):
            if current_title and current_text:
                combined_text = "".join(current_text).strip()
                if not skip_section and len(re.findall(r"[\u4e00-\u9fa5\w]+", combined_text)) > 50:
                    result.append(f"{current_title}\n{combined_text}\n")
            current_title = line.strip()
            current_text = []
            skip_section = any(keyword in current_title for keyword in ["前言", "思考题", "参考文献", "ACKNOWLEDGEMENTS", "Conference Papers", "TABLE OF CONTENTS", "LIST OF FIGURES", " LIST OF TABLES", "Fundamental Concepts in Heterogeneous Catalysis", " 图书在版编目CIP数据", "例题", "Magnetism"])
        else:
            if not skip_section and line.strip():
                current_text.append(line.strip())
    if current_title and current_text:
        combined_text = "".join(current_text).strip()
        if not skip_section and len(re.findall(r"[\u4e00-\u9fa5\w]+", combined_text)) > 20:
            result.append(f"{current_title}\n{combined_text}\n")
    return result

def process_specific_file(file_path, output_file):
    if not os.path.exists(file_path):
        return
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    cleaned_content = clean_and_extract_markdown(content)
    if cleaned_content:
        with open(output_file, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write("\n".join(cleaned_content))

def process_markdown_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and (f.endswith('.md') or f.endswith('.txt'))]
    for filename in tqdm(files, desc="Markdown_cleaner"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.replace('.md', '.txt').replace('.txt', '_cleaned.txt'))
        process_specific_file(input_file_path, output_file_path)


