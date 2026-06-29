#!/usr/bin/env python3
"""PDF批量提取脚本 - 直接处理origin-files-organized目录"""

import os
import json
import fitz  # PyMuPDF
import time
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

INPUT_DIR = "/data/robotlele/ai4ellm/origin-files-organized"
OUTPUT_DIR = "/data/robotlele/ai4ellm/output/pdf_extracted"
LOG_FILE = "/data/robotlele/ai4ellm/output/extraction_progress.json"

def extract_single_pdf(pdf_path):
    """提取单个PDF"""
    rel_path = os.path.relpath(pdf_path, INPUT_DIR)
    cat = rel_path.split(os.sep)[0]
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        pages = doc.page_count
        
        for page_num in range(pages):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text())
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        return {
            "file": rel_path,
            "category": cat,
            "pages": pages,
            "text_length": len(full_text),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "file": rel_path,
            "category": cat,
            "pages": 0,
            "text_length": 0,
            "status": "failed",
            "error": str(e)
        }

def process_file(pdf_path):
    """处理单个文件（用于多进程）"""
    rel_path = os.path.relpath(pdf_path, INPUT_DIR)
    cat = rel_path.split(os.sep)[0]
    base_name = Path(pdf_path).stem
    
    output_path = os.path.join(OUTPUT_DIR, cat, f"{base_name}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        pages = doc.page_count
        
        for page_num in range(pages):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text())
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        if len(full_text.strip()) < 50:
            return (rel_path, cat, "empty", pages, 0)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        return (rel_path, cat, "success", pages, len(full_text))
        
    except Exception as e:
        return (rel_path, cat, "failed", 0, str(e))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有PDF
    pdf_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for fn in files:
            if fn.endswith('.pdf'):
                pdf_files.append(os.path.join(root, fn))
    
    total = len(pdf_files)
    print(f"发现 {total} 个PDF文件")
    print(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 多进程处理
    num_workers = min(mp.cpu_count(), 16)
    print(f"使用 {num_workers} 个进程并行处理")
    
    results = []
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap(process_file, pdf_files), 1):
            results.append(result)
            
            # 每处理100个文件输出进度
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                
                successes = sum(1 for r in results if r[2] == "success")
                failed = sum(1 for r in results if r[2] == "failed")
                empty = sum(1 for r in results if r[2] == "empty")
                total_chars = sum(r[4] for r in results if isinstance(r[4], int))
                
                progress = {
                    "total": total,
                    "processed": i,
                    "success": successes,
                    "failed": failed,
                    "empty": empty,
                    "total_chars": total_chars,
                    "elapsed_seconds": round(elapsed, 1),
                    "rate_per_second": round(rate, 1),
                    "eta_seconds": round(eta, 1),
                    "timestamp": time.time()
                }
                
                print(f"进度: {i}/{total} ({i/total*100:.1f}%) | "
                      f"成功:{successes} 失败:{failed} 空:{empty} | "
                      f"速率:{rate:.1f} 文件/s | "
                      f"耗时:{elapsed/60:.1f}min | "
                      f"预计剩余:{eta/60:.1f}min")
                
                # 保存进度
                with open(LOG_FILE, 'w') as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
    
    # 最终统计
    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r[2] == "success")
    failed = sum(1 for r in results if r[2] == "failed")
    empty = sum(1 for r in results if r[2] == "empty")
    
    # 按分类统计
    cat_stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0, "empty": 0})
    for _, cat, status, _, _ in results:
        cat_stats[cat]["total"] += 1
        cat_stats[cat][status] += 1
    
    print(f"\n{'='*50}")
    print(f"提取完成!")
    print(f"总计: {total} 个文件")
    print(f"成功: {successes} | 失败: {failed} | 空文本: {empty}")
    print(f"总耗时: {elapsed/60:.1f} 分钟")
    print(f"\n按分类统计:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: {stats['total']} -> 成功:{stats['success']} 失败:{stats['failed']} 空:{stats['empty']}")
    
    # 保存最终报告
    report = {
        "total": total,
        "success": successes,
        "failed": failed,
        "empty": empty,
        "elapsed_seconds": round(elapsed, 1),
        "categories": {k: dict(v) for k, v in cat_stats.items()},
        "timestamp": time.time()
    }
    
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n报告保存: {report_path}")

if __name__ == "__main__":
    main()
