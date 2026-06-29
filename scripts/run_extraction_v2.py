#!/usr/bin/env python3
"""
PDF综合提取脚本 - 多引擎提取 + 智能回退
引擎优先级: PyMuPDF (快速文本) → MinerU (高质量OCR) → pdfplumber (备选)
"""

import os
import sys
import json
import time
import logging
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/data/robotlele/ai4ellm/output/extraction.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

INPUT_DIR = "/data/robotlele/ai4ellm/origin-files-organized"
OUTPUT_DIR = "/data/robotlele/ai4ellm/output/pdf_extracted_v2"
PROGRESS_FILE = "/data/robotlele/ai4ellm/output/extraction_progress.json"
REPORT_FILE = "/data/robotlele/ai4ellm/output/extraction_report_v2.json"

def extract_pymupdf(pdf_path, min_text=50):
    """PyMuPDF 快速提取"""
    try:
        import fitz
        fitz.TOOLS.mupdf_warnings(0)  # 抑制 MuPDF 警告
        
        doc = fitz.open(pdf_path)
        text_parts = []
        pages = doc.page_count
        
        for page_num in range(pages):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text())
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        text_len = len(full_text.strip())
        
        if text_len >= min_text:
            return "pymupdf", pages, full_text
        else:
            return "empty", pages, ""
    except Exception as e:
        return "failed", 0, str(e)

def extract_mineru(pdf_path, min_text=50):
    """MinerU 高质量提取（支持OCR）"""
    try:
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        import tempfile
        
        # 读取 PDF
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        
        pages = ds.page_count
        
        # 判断解析方法
        parse_method = ds.classify()
        
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            md_writer = FileBasedDataWriter(tmpdir)
            
            if parse_method == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
            
            # 获取 Markdown
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
                return "mineru", pages, text
            else:
                return "empty", pages, ""
                
    except Exception as e:
        return "failed", 0, str(e)

def extract_with_fallback(pdf_path, min_text=50):
    """多引擎提取 + 智能回退"""
    
    # 第一优先：PyMuPDF（快）
    method, pages, text = extract_pymupdf(pdf_path, min_text)
    
    if method == "pymupdf":
        return pdf_path, "pymupdf", pages, len(text), text
    
    # 第二优先：MinerU（OCR/高质量）
    method, pages, text = extract_mineru(pdf_path, min_text)
    
    if method == "mineru":
        return pdf_path, "mineru", pages, len(text), text
    
    # 都失败了
    return pdf_path, "failed", pages, 0, ""

def process_single_file(args):
    """处理单个文件（供多进程调用）"""
    pdf_path = args
    
    rel_path = os.path.relpath(pdf_path, INPUT_DIR)
    cat = rel_path.split(os.sep)[0]
    
    try:
        pdf_path_resolved, method, pages, text_len, text = extract_with_fallback(pdf_path)
        
        if method == "failed":
            return (rel_path, cat, "failed", 0, 0, str(text))
        
        if text_len < 50:
            return (rel_path, cat, "empty", pages, 0, "")
        
        # 保存文本
        base_name = Path(pdf_path).stem
        output_path = os.path.join(OUTPUT_DIR, cat, f"{base_name}.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return (rel_path, cat, method, pages, text_len, "")
        
    except Exception as e:
        return (rel_path, cat, "error", 0, 0, str(e))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 收集所有 PDF
    pdf_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for fn in files:
            if fn.endswith('.pdf'):
                pdf_files.append(os.path.join(root, fn))
    
    total = len(pdf_files)
    print(f"📂 发现 {total} 个 PDF 文件")
    print(f"📁 输入目录: {INPUT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print()
    
    # 检查是否有已提取的文本（跳过）
    existing = 0
    to_process = []
    for pdf_path in pdf_files:
        rel = os.path.relpath(pdf_path, INPUT_DIR)
        cat = rel.split(os.sep)[0]
        base = Path(pdf_path).stem
        out_path = os.path.join(OUTPUT_DIR, cat, f"{base}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 50:
            existing += 1
        else:
            to_process.append(pdf_path)
    
    print(f"✅ 已有提取: {existing} 个（跳过）")
    print(f"⏳ 待处理: {len(to_process)} 个")
    print()
    
    if not to_process:
        print("所有文件已提取完毕！")
        return
    
    # 多进程处理
    num_workers = min(mp.cpu_count(), 8)  # MinerU 比较重，用少一点进程
    print(f"🔧 使用 {num_workers} 个进程")
    print(f"📖 提取策略: PyMuPDF → MinerU (自动回退)")
    print()
    
    results = []
    start_time = time.time()
    total_with_existing = len(to_process) + existing
    
    with mp.Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap(process_single_file, to_process), 1):
            results.append(result)
            global_i = i + existing
            
            if global_i % 50 == 0 or global_i == total_with_existing:
                elapsed = time.time() - start_time
                rate = global_i / elapsed if elapsed > 0 else 0
                eta = (total_with_existing - global_i) / rate if rate > 0 else 0
                
                success = sum(1 for r in results if r[2] in ["pymupdf", "mineru"])
                failed = sum(1 for r in results if r[2] == "failed")
                empty = sum(1 for r in results if r[2] == "empty")
                pymupdf_cnt = sum(1 for r in results if r[2] == "pymupdf")
                mineru_cnt = sum(1 for r in results if r[2] == "mineru")
                total_chars = sum(r[4] for r in results if isinstance(r[4], int) and r[4] > 0)
                
                print(f"  进度: {global_i}/{total_with_existing} ({global_i/total_with_existing*100:.1f}%) | "
                      f"成功:{success+existing}(pymupdf:{pymupdf_cnt}, mineru:{mineru_cnt}) "
                      f"失败:{failed} 空:{empty} | "
                      f"字符:{total_chars/1024/1024:.1f}MB | "
                      f"速率:{rate:.1f}/s | "
                      f"预计:{eta/60:.1f}min")
                
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({
                        "processed": global_i,
                        "total": total_with_existing,
                        "success": success + existing,
                        "failed": failed,
                        "empty": empty,
                        "pymupdf": pymupdf_cnt,
                        "mineru": mineru_cnt,
                        "timestamp": time.time()
                    }, f, indent=2)
    
    # 最终统计
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r[2] in ["pymupdf", "mineru"]) + existing
    failed = sum(1 for r in results if r[2] == "failed")
    empty = sum(1 for r in results if r[2] == "empty")
    pymupdf_cnt = sum(1 for r in results if r[2] == "pymupdf") + existing
    mineru_cnt = sum(1 for r in results if r[2] == "mineru")
    total_chars = sum(r[4] for r in results if isinstance(r[4], int) and r[4] > 0)
    
    # 按分类统计
    cat_stats = defaultdict(lambda: {"total": 0, "pymupdf": 0, "mineru": 0, "failed": 0, "empty": 0})
    for rel, cat, method, pages, chars, _ in results:
        cat_stats[cat]["total"] += 1
        if method in ["pymupdf", "mineru"]:
            cat_stats[cat][method] += 1
        elif method == "failed":
            cat_stats[cat]["failed"] += 1
        elif method == "empty":
            cat_stats[cat]["empty"] += 1
    
    # 加上已有的
    for cat_dir in os.listdir(OUTPUT_DIR):
        cat_path = os.path.join(OUTPUT_DIR, cat_dir)
        if not os.path.isdir(cat_path):
            continue
        txt_count = len([f for f in os.listdir(cat_path) if f.endswith('.txt')])
        if cat_dir not in cat_stats and txt_count > 0:
            cat_stats[cat_dir]["total"] = txt_count
            cat_stats[cat_dir]["pymupdf"] = txt_count
    
    print(f"\n{'='*60}")
    print(f"✅ 提取完成!")
    print(f"  总计: {total} 个 PDF")
    print(f"  成功: {success} (PyMuPDF:{pymupdf_cnt}, MinerU:{mineru_cnt})")
    print(f"  失败: {failed}")
    print(f"  空文本: {empty}")
    print(f"  总字符: {total_chars/1024/1024:.1f} MB")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    print(f"\n📊 按分类:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: 总计:{stats['total']} PyMuPDF:{stats['pymupdf']} MinerU:{stats['mineru']} "
              f"失败:{stats['failed']} 空:{stats['empty']}")
    
    # 生成报告
    report = {
        "total": total,
        "success": success,
        "pymupdf": pymupdf_cnt,
        "mineru": mineru_cnt,
        "failed": failed,
        "empty": empty,
        "total_chars": total_chars,
        "total_mb": round(total_chars / 1024 / 1024, 1),
        "elapsed_seconds": round(elapsed, 1),
        "categories": {k: dict(v) for k, v in sorted(cat_stats.items())},
        "timestamp": time.time()
    }
    
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 报告: {REPORT_FILE}")

if __name__ == "__main__":
    main()
