#!/usr/bin/env python3
"""
PDF语料清洗 + 格式转换 + 去重流水线
读取已提取的文本 -> 清洗 -> JSONL格式 -> 去重
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

INPUT_DIR = "/data/robotlele/ai4ellm/output/pdf_extracted"
OUTPUT_DIR = "/data/robotlele/ai4ellm/output/pretrain_data"
PROGRESS_FILE = "/data/robotlele/ai4ellm/output/pipeline_progress.json"

def clean_text(text):
    """清洗文本"""
    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 去除 URL
    text = re.sub(r'https?://\S+', '', text)
    
    # 去除连续的空白字符（保留段落换行）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if line:  # 跳过空行
            cleaned_lines.append(line)
    
    # 重新组合，最多保留一个空行
    result = []
    prev_empty = False
    for line in cleaned_lines:
        if not line:
            if not prev_empty:
                result.append('')
                prev_empty = True
        else:
            result.append(line)
            prev_empty = False
    
    return '\n'.join(result)

def compute_dedup_hash(text):
    """计算去重 hash（归一化后）"""
    normalized = text.lower()
    normalized = re.sub(r'\s+', '', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()

def process_file(args):
    """处理单个文本文件"""
    txt_path, cat = args
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text.strip()) < 50:
            return None
        
        # 清洗
        cleaned = clean_text(text)
        
        if len(cleaned.strip()) < 50:
            return None
        
        # 计算 hash
        file_hash = compute_dedup_hash(cleaned)
        
        return {
            "source": cat,
            "filename": os.path.basename(txt_path),
            "text": cleaned,
            "char_count": len(cleaned),
            "hash": file_hash
        }
        
    except Exception as e:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有 txt 文件
    tasks = []
    for cat_dir in sorted(os.listdir(INPUT_DIR)):
        cat_path = os.path.join(INPUT_DIR, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        for fn in sorted(os.listdir(cat_path)):
            if fn.endswith('.txt'):
                tasks.append((os.path.join(cat_path, fn), cat_dir))
    
    total = len(tasks)
    print(f"📂 发现 {total} 个提取的文本文件")
    print()
    
    # 多进程处理
    num_workers = min(mp.cpu_count(), 16)
    print(f"🔧 使用 {num_workers} 个进程")
    print()
    
    start_time = time.time()
    results = []
    
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap(process_file, tasks), 1):
            if result is not None:
                results.append(result)
            
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  进度: {i}/{total} ({i/total*100:.1f}%) | "
                      f"有效:{len(results)} | "
                      f"速率:{rate:.1f}/s | "
                      f"预计剩余:{eta:.0f}s")
                
                # 保存进度
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({
                        "stage": "cleaning",
                        "processed": i,
                        "total": total,
                        "valid": len(results),
                        "timestamp": time.time()
                    }, f)
    
    print(f"\n  清洗完成: {len(results)}/{total} 个有效文本")
    
    # 去重 - 精确去重
    print("\n🔄 开始精确去重...")
    seen_hashes = set()
    unique_docs = []
    dup_count = 0
    
    for doc in results:
        if doc["hash"] not in seen_hashes:
            seen_hashes.add(doc["hash"])
            unique_docs.append(doc)
        else:
            dup_count += 1
    
    print(f"  唯一文档: {len(unique_docs)}")
    print(f"  去重删除: {dup_count}")
    
    # 保存为 JSONL
    print(f"\n💾 保存到 JSONL...")
    jsonl_path = os.path.join(OUTPUT_DIR, "pretrain.jsonl")
    total_chars = 0
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for doc in unique_docs:
            record = {
                "source": doc["source"],
                "filename": doc["filename"],
                "text": doc["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            total_chars += doc["char_count"]
    
    # 按分类统计
    cat_stats = defaultdict(lambda: {"count": 0, "chars": 0})
    for doc in unique_docs:
        cat_stats[doc["source"]]["count"] += 1
        cat_stats[doc["source"]]["chars"] += doc["char_count"]
    
    # 生成报告
    elapsed = time.time() - start_time
    report = {
        "stage": "cleaning_dedup",
        "input_files": total,
        "valid_after_cleaning": len(results),
        "unique_after_dedup": len(unique_docs),
        "duplicates_removed": dup_count,
        "total_chars": total_chars,
        "total_mb": total_chars / 1024 / 1024,
        "output_file": jsonl_path,
        "elapsed_seconds": round(elapsed, 1),
        "categories": {k: {
            "count": v["count"],
            "chars": v["chars"],
            "mb": round(v["chars"] / 1024 / 1024, 1)
        } for k, v in sorted(cat_stats.items())},
        "timestamp": time.time()
    }
    
    report_path = os.path.join(OUTPUT_DIR, "pipeline_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*50}")
    print(f"✅ 清洗 + 去重完成!")
    print(f"  输入: {total} 个文件")
    print(f"  有效: {len(results)} 个（清洗后）")
    print(f"  唯一: {len(unique_docs)} 个（去重后）")
    print(f"  去重删除: {dup_count} 个")
    print(f"  总字符数: {total_chars/1024/1024:.1f} MB")
    print(f"  输出: {jsonl_path}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    
    print(f"\n📊 按分类统计:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: {stats['count']} 篇, {stats['chars']/1024/1024:.1f} MB")
    
    print(f"\n📄 报告: {report_path}")

if __name__ == "__main__":
    main()
