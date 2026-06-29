#!/usr/bin/env python3
"""
专门针对失败的 PDF 跑 MinerU OCR
"""

import os
import sys
import json
import time
import tempfile
import logging
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

# 抑制 MuPDF 警告
os.environ["MUPDF_LOG_LEVEL"] = "error"

INPUT_DIR = "/data/robotlele/ai4ellm/origin-files-organized"
OUTPUT_DIR = "/data/robotlele/ai4ellm/output/pdf_extracted_v2"
REPORT_FILE = "/data/robotlele/ai4ellm/output/miner_ocr_report.json"

def extract_mineru(pdf_path, min_text=50):
    """MinerU 提取"""
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod
        
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        
        pages = ds.page_count
        parse_method = ds.classify()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            md_writer = FileBasedDataWriter(tmpdir)
            
            if parse_method == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
            
            stem = Path(pdf_path).stem
            pipe_result.dump_md(md_writer, f"{stem}.md", "images")
            
            md_path = os.path.join(tmpdir, f"{stem}.md")
            if os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = ""
            
            text_len = len(text.strip())
            if text_len >= min_text:
                return "success", pages, text_len, text, ""
            else:
                return "empty", pages, 0, "", "OCR提取文本过少"
                
    except Exception as e:
        return "failed", 0, 0, "", str(e)

def process_single(args):
    pdf_path = args
    rel_path = os.path.relpath(pdf_path, INPUT_DIR)
    cat = rel_path.split(os.sep)[0]
    
    status, pages, text_len, text, error = extract_mineru(pdf_path)
    
    if status == "success":
        base_name = Path(pdf_path).stem
        output_path = os.path.join(OUTPUT_DIR, cat, f"{base_name}.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    return (rel_path, cat, status, pages, text_len, error)

def main():
    # 找失败的 PDF
    failed_pdfs = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for fn in files:
            if not fn.endswith('.pdf'):
                continue
            pdf_path = os.path.join(root, fn)
            rel = os.path.relpath(pdf_path, INPUT_DIR)
            cat = rel.split(os.sep)[0]
            base = Path(pdf_path).stem
            out_path = os.path.join(OUTPUT_DIR, cat, f"{base}.txt")
            
            # 没有输出文件的视为失败
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 50:
                failed_pdfs.append(pdf_path)
    
    total = len(failed_pdfs)
    print(f"🔍 发现 {total} 个失败的 PDF，开始 MinerU OCR...")
    print()
    
    if total == 0:
        print("没有需要处理的文件！")
        return
    
    num_workers = 4  # MinerU 很重，用少点
    print(f"🔧 使用 {num_workers} 个进程（MinerU OCR）")
    print()
    
    start_time = time.time()
    results = []
    
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap(process_single, failed_pdfs), 1):
            results.append(result)
            
            if i % 20 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                
                success = sum(1 for r in results if r[2] == "success")
                failed = sum(1 for r in results if r[2] == "failed")
                empty = sum(1 for r in results if r[2] == "empty")
                
                print(f"  进度: {i}/{total} ({i/total*100:.1f}%) | "
                      f"成功:{success} 失败:{failed} 空:{empty} | "
                      f"速率:{rate:.1f}/s | "
                      f"预计:{eta/60:.1f}min")
    
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r[2] == "success")
    failed = sum(1 for r in results if r[2] == "failed")
    empty = sum(1 for r in results if r[2] == "empty")
    total_chars = sum(r[4] for r in results if r[4] > 0)
    
    # 按分类统计
    cat_stats = defaultdict(lambda: {"success": 0, "failed": 0, "empty": 0})
    for _, cat, status, _, chars, _ in results:
        cat_stats[cat][status] += 1
    
    print(f"\n{'='*60}")
    print(f"✅ MinerU OCR 完成!")
    print(f"  总计: {total}")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  空文本: {empty}")
    print(f"  新增字符: {total_chars/1024/1024:.1f} MB")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    
    print(f"\n📊 按分类:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: 成功:{stats['success']} 失败:{stats['failed']} 空:{stats['empty']}")
    
    # 总计全部
    total_txt = 0
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fn in files:
            if fn.endswith('.txt'):
                total_txt += 1
                total_size += os.path.getsize(os.path.join(root, fn))
    
    print(f"\n📈 提取总计:")
    print(f"  文本文件: {total_txt}")
    print(f"  总大小: {total_size/1024/1024:.1f} MB")
    
    report = {
        "mineru_total": total,
        "mineru_success": success,
        "mineru_failed": failed,
        "mineru_empty": empty,
        "mineru_chars_mb": round(total_chars/1024/1024, 1),
        "elapsed_seconds": round(elapsed, 1),
        "total_extracted": total_txt,
        "total_size_mb": round(total_size/1024/1024, 1),
        "categories": {k: dict(v) for k, v in sorted(cat_stats.items())},
        "timestamp": time.time()
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 报告: {REPORT_FILE}")

if __name__ == "__main__":
    main()
