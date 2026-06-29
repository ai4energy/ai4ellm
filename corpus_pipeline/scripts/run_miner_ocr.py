"""
扫描件 PDF 的 MinerU OCR 重提取脚本

针对主流水线 PyMuPDF 失败（文本过少/扫描件）的 PDF，用 MinerU (magic_pdf 1.3.x)
走 OCR 模式重新提取，输出 .md 到 extracted_text/，与主流水线产物合并。

用法:
    # 测试单个
    python scripts/run_miner_ocr.py --input /tmp/failed_pdfs_unique.txt --limit 1 --dry-run

    # 小批量验证
    python scripts/run_miner_ocr.py --input /tmp/failed_pdfs_unique.txt --limit 5

    # 全量
    python scripts/run_miner_ocr.py --input /tmp/failed_pdfs_unique.txt

环境: conda activate mineru （magic-pdf 1.3.12 + onnxruntime-gpu）
"""

import os
import sys
import time
import argparse

# 项目根
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)


def parse_args():
    ap = argparse.ArgumentParser(description="MinerU OCR 重提取扫描件 PDF")
    ap.add_argument("--input", required=True, help="失败 PDF 路径清单（每行一个绝对路径）")
    ap.add_argument("--output-dir", default=None,
                    help="输出根目录（默认复用主输出 ./output）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只列出待处理文件不跑 OCR")
    ap.add_argument("--min-text-length", type=int, default=50,
                    help="OCR 后文本少于此长度视为失败")
    ap.add_argument("--device", default="cuda", help="cuda | cpu（MinerU OCR 设备）")
    return ap.parse_args()


def load_pdf_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and os.path.exists(ln.strip())]


def ocr_one(pdf_path: str, output_dir: str, min_text_length: int) -> tuple[str | None, str]:
    """
    用 MinerU OCR 单个 PDF，输出 {output_dir}/{name}.md。
    返回 (output_md_path | None, error)
    """
    from pathlib import Path
    import tempfile
    from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    name = Path(pdf_path).stem
    try:
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_path)
        ds = PymuDocDataset(pdf_bytes)
        parse_method = ds.classify()

        with tempfile.TemporaryDirectory() as tmpdir:
            image_writer = FileBasedDataWriter(tmpdir)
            md_writer = FileBasedDataWriter(tmpdir)

            # 扫描件一定走 OCR；classify 为 txt 的也强制 OCR 提高召回
            if parse_method == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                # txt 模式先试，文本少再回退 OCR
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            pipe_result.dump_md(md_writer, f"{name}.md", "images")
            md_path = os.path.join(tmpdir, f"{name}.md")
            text = ""
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read()

            if len(text.strip()) < min_text_length:
                # txt 模式失败，回退 OCR
                if parse_method != SupportedPdfParseMethod.OCR:
                    infer_result2 = ds.apply(doc_analyze, ocr=True)
                    pipe_result2 = infer_result2.pipe_ocr_mode(image_writer)
                    pipe_result2.dump_md(md_writer, f"{name}.md", "images")
                    if os.path.exists(md_path):
                        with open(md_path, "r", encoding="utf-8") as f:
                            text = f.read()
                if len(text.strip()) < min_text_length:
                    return None, f"OCR 文本过少 ({len(text.strip())} chars)"

        # 写入最终输出
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"{name}.md")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        return out_file, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    args = parse_args()

    # 默认输出到主流水线的 extracted_text
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJ_ROOT, "output", "extracted_text")

    pdfs = load_pdf_list(args.input)
    if args.limit > 0:
        pdfs = pdfs[:args.limit]

    print(f"待处理 PDF: {len(pdfs)} 个")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")

    if args.dry_run:
        for p in pdfs:
            print(f"  [DRY] {p}")
        return

    ok, fail = 0, 0
    t0 = time.time()
    errors = []
    for i, pdf in enumerate(pdfs, 1):
        # 跳过已存在的输出（幂等续跑）
        name = os.path.splitext(os.path.basename(pdf))[0]
        out_candidate = os.path.join(args.output_dir, f"{name}.md")
        if os.path.exists(out_candidate):
            ok += 1
            print(f"[{i}/{len(pdfs)}] 跳过(已存在): {os.path.basename(pdf)}")
            continue

        t1 = time.time()
        out, err = ocr_one(pdf, args.output_dir, args.min_text_length)
        dt = time.time() - t1
        if out:
            ok += 1
            print(f"[{i}/{len(pdfs)}] OK ({dt:.1f}s): {os.path.basename(pdf)}")
        else:
            fail += 1
            errors.append((pdf, err))
            print(f"[{i}/{len(pdfs)}] FAIL ({dt:.1f}s): {os.path.basename(pdf)} | {err}")

    total_dt = time.time() - t0
    print(f"\n{'='*50}")
    print(f"完成: {ok} 成功, {fail} 失败, 总耗时 {total_dt:.1f}s")
    if errors:
        err_path = os.path.join(args.output_dir, "..", "reports", "miner_ocr_errors.txt")
        os.makedirs(os.path.dirname(err_path), exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as f:
            for p, e in errors:
                f.write(f"{p} | {e}\n")
        print(f"错误清单: {err_path}")

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
