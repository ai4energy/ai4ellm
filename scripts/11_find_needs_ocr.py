#!/usr/bin/env python3
"""扫描 corpus_pipeline 产物，找出需要 OCR 的 PDF。

判据（任一）:
  - 没有 extracted_text/*.txt
  - 文本 < 50 字符
  - 字符/页 < 100
输出: <output_dir>/needs_ocr.jsonl，并按页数汇总预计 OCR 成本。
"""
import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/data/home/ai4e-hermes/corpus/full-20261102")
    ap.add_argument("--input-dir", default="/home/ai4e-hermes/books")
    ap.add_argument("--min-chars-per-page", type=float, default=100.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    text_dir = out / "extracted_text"
    meta_dir = out / "metadata"
    pages: dict[str, int] = {}
    for m in meta_dir.glob("*_meta.json"):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            pages[m.name[:-len("_meta.json")]] = int(d.get("page_count") or 0)
        except Exception:
            pass

    rows = []
    for pdf in sorted(Path(args.input_dir).glob("*.pdf")):
        stem = pdf.stem
        txt = text_dir / f"{stem}.txt"
        chars = txt.stat().st_size if txt.exists() else 0
        pg = pages.get(stem, 0)
        cpp = chars / pg if pg else 0.0
        reasons = []
        if not txt.exists():
            reasons.append("no_text_file")
        elif chars < 50:
            reasons.append("text_too_short")
        if pg and cpp < args.min_chars_per_page:
            reasons.append(f"chars_per_page<{args.min_chars_per_page:g}")
        if reasons:
            rows.append({
                "path": str(pdf), "name": pdf.name, "pages": pg,
                "text_chars": chars, "chars_per_page": round(cpp, 1),
                "reasons": reasons,
            })

    rows.sort(key=lambda r: (-r["pages"], r["name"]))
    out_file = out / "needs_ocr.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_pages = sum(r["pages"] for r in rows)
    print(json.dumps({
        "output": str(out_file),
        "flagged_books": len(rows),
        "flagged_pages": total_pages,
        "all_books": len(list(Path(args.input_dir).glob("*.pdf"))),
        "est_gpu_hours_at_1_5s_page_1gpu": round(total_pages * 1.5 / 3600, 1),
        "est_gpu_hours_at_1_5s_page_2gpu": round(total_pages * 1.5 / 3600 / 2, 1),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
