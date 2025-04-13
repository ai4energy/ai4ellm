# pdf语料准备

## 📄 1. PDF 转换（pdf_converter.py）

**功能说明：**  
将指定文件夹内的 `.docx`、`.pptx` 等文件批量转换为 PDF 文件。

**核心代码：**
```python
from comtypes.client import CreateObject
import os

def convert_to_pdf(input_path, output_path, file_type):
    if file_type in ['.docx', '.doc']:
        word = CreateObject('Word.Application')
        doc = word.Documents.Open(input_path)
        doc.SaveAs(output_path, FileFormat=17)
        doc.Close()
        word.Quit()
    elif file_type in ['.pptx', '.ppt']:
        powerpoint = CreateObject('PowerPoint.Application')
        presentation = powerpoint.Presentations.Open(input_path)
        presentation.SaveAs(output_path, 32)
        presentation.Close()
        powerpoint.Quit()

def batch_convert(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.docx', '.doc', '.pptx', '.ppt']:
                input_file = os.path.join(root, file)
                output_file = os.path.splitext(input_file)[0] + '.pdf'
                convert_to_pdf(input_file, output_file, ext)
```

---

## 📥 2. PDF 提取（转 Markdown,pdf_extractor.py）

**功能说明：**  
将 PDF 文件内容提取并保存为 Markdown 文件，支持 OCR 识别。

**核心代码：**
```python
def process_pdf(pdf_file_name, output_dir="output"):
    import torch
    from magic_pdf.data.dataset import PymuDocDataset

    ds = PymuDocDataset(pdf_file_name)
    parse_method = ds.classify()
    infer_result = ds.apply(ocr=(parse_method == "OCR"))
    infer_result.dump_md(output_dir)
```

---
---

## 📝 3. Markdown 内容清洗(markdown_cleaner.py)

**功能说明：**  
清洗 Markdown 内容，去除图片、链接、目录、乱码和参考文献。

**核心代码：**
```python
def clean_markdown(content):
    import re
    cleaned_content = []
    for line in content:
        line = re.sub(r"!\[.*?\]\(.*?\)", "", line)  # 删除图片
        line = re.sub(r"\[.*?\]\(.*?\)", "", line)  # 删除链接
        line = re.sub(r"\[\d+\]", "", line)          # 删除文献引用
        cleaned_content.append(line.strip())
    return cleaned_content
```

---
## 📦 4. txt 文件合并(me.py)

**功能说明：**  
将多个 txt 或文本文件合并为一个文件，方便后续处理。

**核心代码：**
```python
import os

def merge_txt_files(input_folder, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".txt"):
                file_path = os.path.join(input_folder, file_name)
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
```



## 🔍 5. 语义去重(semantic_deduplicator.py)

**功能说明：**  
基于 Sentence-BERT 模型，计算句子相似度并去重。

**核心代码：**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_deduplicate(input_file, output_file, threshold=0.9):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    retained_embeddings = []
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            emb = model.encode(line.strip(), convert_to_numpy=True)
            if not retained_embeddings or max(np.dot(emb, np.vstack(retained_embeddings).T)) < threshold:
                fout.write(line)
                retained_embeddings.append(emb)
```

---

## 🔄 6. TXT 转 JSONL(txt_to_jsonl_converter.py)

**功能说明：**  
将清洗后的 `.txt` 文件转化为结构化的 `.jsonl` 文件，按章节保存。

**核心代码：**
```python
import json
import os

def txt_to_jsonl(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(input_folder, file_name), 'r') as f:
                sections = f.read().split('\n# ')
            with open(os.path.join(output_folder, file_name.replace('.txt', '.jsonl')), 'w') as f:
                for section in sections:
                    title, *content = section.split('\n')
                    json.dump({"section": title, "content": "\n".join(content)}, f, ensure_ascii=False)
                    f.write('\n')
```


## 🎯 **总结**

- **PDF 转换** → **PDF 提取** → **文件合并** → **Markdown 清洗** → **语义去重** → **JSONL 转换**
