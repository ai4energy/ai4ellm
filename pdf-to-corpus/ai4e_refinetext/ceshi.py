import os
import re
import sys
import math
import json
import numpy as np
from comtypes.client import CreateObject
import torch.multiprocessing as mp
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pdf_converter
from  tiqu import ocr_folder_to_markdown
from markdown_cleaner import process_markdown_files
from me import allin
from semantic_deduplicator import semantic_deduplicate
from txt_to_jsonl_converter import txt_to_jsonl
def liucheng(input_folder_path):
    # 子文件夹名称
    subdirs = ['markdown', 'cleaned_markdown', 'txt', 'deduplicate_txt', 'json']
    # 对应子文件夹中要创建的文件名
    file_names = {
        'txt': 'example.txt',
        'deduplicate_txt': 'dedup.txt'
    }

    created_paths = []

    for subdir in subdirs:
        path = os.path.join(input_folder_path, subdir)
        try:
            # 创建子文件夹
            os.makedirs(path, exist_ok=True)

            # 如果子文件夹需要创建文件，则生成文件路径并创建空文件
            if subdir in file_names:
                file_path = os.path.join(path, file_names[subdir])
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('')  # 创建空文件
                created_paths.append(os.path.abspath(file_path))
            else:
                # 否则直接保存子文件夹路径
                created_paths.append(os.path.abspath(path))

        except Exception as e:
            print(f"无法创建目录或文件 {path}: {e}")
    print(created_paths)
    # 1.pdf_converter
    pdf_converter.batch_convert(input_folder_path)
    # 2.pdf_提取
    output_dir = created_paths [0]
    ocr_folder_to_markdown(input_folder_path)
    # 3.markdown清洗
    process_markdown_files(os.path.join(created_paths[0], "markdown"), created_paths[1])
    # 4.merge
    allin(created_paths[1],created_paths[2])
    # 5.去重
    semantic_deduplicate(created_paths[2], created_paths[3], similarity_threshold=0.8)
    # 6.json
    txt_to_jsonl(created_paths[3], created_paths[4])
