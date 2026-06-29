#!/usr/bin/env python3
"""
扫描版 PDF OCR 提取 - 单文件模式（避免内存累积）
使用 rapidocr-onnxruntime 处理扫描件
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from collections import defaultdict

INPUT_DIR = "/data/robotlele/ai4ellm/origin-files-organized"
OUTPUT_DIR = "/data/robotlele/ai4ellm/output/pdf_extracted_v2"
PROGRESS_FILE = "/data/robotlele/ai4ellm/output/ocr_progress.json"
REPORT_FILE = "/data/robotlele/ai4ellm/output/ocr_report.json"

WORKER_SCRIPT = """
import sys, fitz, json
from rapidocr_onnxruntime import RapidOCR

pdf_path = sys.argv[1]
min_text = int(sys.argv[2])

try:
    ocr = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)
    doc = fitz.open(pdf_path)
    
    all_text = []
    pages = doc.page_count
    
    for page_num in range(pages):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(1.0, 1.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        ocr_result, _ = ocr(img_bytes)
        
        if ocr_result:
            page_texts = [item[1] for item in ocr_result]
            all_text.append('\\n'.join(page_texts))
        else:
            all_text.append('')
        
        pix = None
    
    doc.close()
    
    full_text = '\\n\\n'.join(all_text)
    text_len = len(full_text.strip())
    
    if text_len >= min_text:
        result = {"status": "success", "pages": pages, "text_len": text_len, "text": full_text}
    else:
        result = {"status": "empty", "pages": pages, "text_len": 0, "text": ""}
    
except Exception as e:
    result = {"status": "failed", "pages": 0, "text_len": 0, "text": str(e)}

print(json.dumps(result, ensure_ascii=False))
"""

def extract_ocr_subprocess(pdf_path, min_text=50, timeout=3600):
    """用子进程跑 OCR，跑完释放内存"""
    # 写临时脚本
    tmp_script = "/tmp/ocr_worker_temp.py"
    with open(tmp_script, 'w') as f:
        f.write(WORKER_SCRIPT)
    
    try:
        proc = subprocess.run(
            ['python3', tmp_script, pdf_path, str(min_text)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if proc.returncode != 0:
            return "failed", 0, 0, f"stderr: {proc.stderr[:500]}"
        
        result = json.loads(proc.stdout.strip())
        return result["status"], result["pages"], result["text_len"], result["text"]
        
    except subprocess.TimeoutExpired:
        return "failed", 0, 0, "timeout"
    except Exception as e:
        return "failed", 0, 0, str(e)
    finally:
        if os.path.exists(tmp_script):
            os.remove(tmp_script)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
            
            if not os.path.exists(out_path) or os.path.getsize(out_path) < 50:
                failed_pdfs.append(pdf_path)
    
    total = len(failed_pdfs)
    print(f"🔍 发现 {total} 个失败的 PDF，开始 OCR（子进程模式）...")
    print()
    
    if total == 0:
        print("没有需要处理的文件！")
        return
    
    start_time = time.time()
    results = []
    
    for i, pdf_path in enumerate(failed_pdfs, 1):
        rel_path = os.path.relpath(pdf_path, INPUT_DIR)
        cat = rel_path.split(os.sep)[0]
        
        status, pages, text_len, text = extract_ocr_subprocess(pdf_path)
        results.append((rel_path, cat, status, pages, text_len))
        
        if status == "success":
            base_name = Path(pdf_path).stem
            output_path = os.path.join(OUTPUT_DIR, cat, f"{base_name}.txt")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        if i % 5 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            
            success = sum(1 for r in results if r[2] == "success")
            failed = sum(1 for r in results if r[2] == "failed")
            empty = sum(1 for r in results if r[2] == "empty")
            
            print(f"  进度: {i}/{total} ({i/total*100:.1f}%) | "
                  f"成功:{success} 失败:{failed} 空:{empty} | "
                  f"速率:{rate:.2f}/s | "
                  f"预计:{eta/60:.1f}min")
            
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    "processed": i,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "empty": empty,
                    "timestamp": time.time()
                }, f, indent=2)
    
    elapsed = time.time() - start_time
    success = sum(1 for r in results if r[2] == "success")
    failed = sum(1 for r in results if r[2] == "failed")
    empty = sum(1 for r in results if r[2] == "empty")
    total_chars = sum(r[4] for r in results if r[4] > 0)
    
    cat_stats = defaultdict(lambda: {"success": 0, "failed": 0, "empty": 0})
    for _, cat, status, _, chars in results:
        cat_stats[cat][status] += 1
    
    print(f"\n{'='*60}")
    print(f"✅ OCR 完成!")
    print(f"  总计: {total}")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  空文本: {empty}")
    print(f"  新增字符: {total_chars/1024/1024:.1f} MB")
    print(f"  耗时: {elapsed/60:.1f} 分钟")
    
    print(f"\n📊 按分类:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat}: 成功:{stats['success']} 失败:{stats['failed']} 空:{stats['empty']}")
    
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
        "ocr_total": total,
        "ocr_success": success,
        "ocr_failed": failed,
        "ocr_empty": empty,
        "ocr_chars_mb": round(total_chars/1024/1024, 1),
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
