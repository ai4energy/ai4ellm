#!/usr/bin/env python3
"""用 PDF 前 3 页文本补全出版社/年份，输出 enrichment.jsonl（只读）。"""
import argparse
import json
import multiprocessing as mp
import re
from pathlib import Path

import pymupdf

PUBLISHERS = [
    "Springer", "Elsevier", "Wiley", "CRC Press", "Taylor & Francis", "McGraw-Hill",
    "Pearson", "Cengage", "ASHRAE", "Academic Press", "Cambridge University Press",
    "Oxford University Press", "IEEE", "Butterworth-Heinemann", "Pergamon",
    "Prentice Hall", "Addison-Wesley", "John Wiley & Sons", "World Scientific",
    "中国电力出版社", "化学工业出版社", "机械工业出版社", "科学出版社",
    "高等教育出版社", "中国建筑工业出版社", "中国石化出版社", "石油工业出版社",
    "清华大学出版社", "北京大学出版社", "人民邮电出版社", "电子工业出版社",
    "国防工业出版社", "中国水利水电出版社", "上海科学技术出版社",
    "华中科技大学出版社", "西安交通大学出版社", "冶金工业出版社",
    "中国轻工业出版社", "中国农业出版社", "中国环境科学出版社",
]
YEAR_RE = re.compile(r"(?:©|copyright|Copyright|出版|印刷|版次)[^\n]{0,60}?((?:19|20)\d{2})")


def work(row: dict) -> dict:
    out = {"sha256": row["sha256"], "publisher": None, "year": None, "text_found": False}
    try:
        doc = pymupdf.open(row["original_path"])
        pages = min(3, doc.page_count)
        text = "\n".join(doc.load_page(i).get_text() for i in range(pages))
        doc.close()
    except Exception:
        return out
    out["text_found"] = bool(text.strip())
    low = text.lower()
    for p in PUBLISHERS:
        if p.lower() in low:
            out["publisher"] = p
            break
    m = YEAR_RE.search(text)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2026:
            out["year"] = y
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/corpus-index/rename_plan.jsonl")
    ap.add_argument("--out", default="data/corpus-index/enrichment.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.plan, encoding="utf-8")]
    uniq = [r for r in rows if not r["is_duplicate"]]
    print(f"enrich {len(uniq)} unique pdfs with {args.workers} workers", flush=True)
    with mp.Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(work, uniq, chunksize=8), 1):
            results.append(res)
            if i % 500 == 0:
                print(f"{i}/{len(uniq)}", flush=True)
    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    pubs = sum(1 for r in results if r["publisher"])
    years = sum(1 for r in results if r["year"])
    print(json.dumps({"rows": len(results), "publisher_found": pubs,
                      "year_found": years, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
