#!/usr/bin/env python3
"""用 PaddleOCR 对单个 PDF 做逐页 OCR（可断点续跑）。

在 .venv-paddle 中运行:
  PADDLE_PDX_CACHE_HOME=/data/models/paddleocr PADDLE_PDX_MODEL_SOURCE=bos \
  CUDA_VISIBLE_DEVICES=2 .venv-paddle/bin/python scripts/12_ocr_pdf.py \
      --pdf /path/book.pdf --out-dir /data/home/ai4e-hermes/corpus/ocr --device gpu --dpi 200

输出:
  <out-dir>/<stem>.jsonl   每页一行 {page, lines, chars, text}
  <out-dir>/<stem>.md      按页拼接的文本
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pymupdf


def collect_texts(result) -> list[str]:
    texts: list[str] = []
    for r in result:
        d = getattr(r, "json", None)
        if callable(d):
            d = d()
        if d is None and isinstance(r, dict):
            d = r
        if not isinstance(d, dict):
            continue
        payload = d.get("res", d)
        texts.extend(payload.get("rec_texts") or [])
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--start-page", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=0, help="0=全部")
    ap.add_argument("--lang", default="ch")
    ap.add_argument("--det-model", default=None)
    ap.add_argument("--rec-model", default=None)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{pdf.stem}.jsonl"
    md_path = out_dir / f"{pdf.stem}.md"

    done: dict[int, dict] = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[int(rec["page"])] = rec

    from paddleocr import PaddleOCR
    kwargs = dict(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=args.device,
    )
    if args.det_model:
        kwargs["text_detection_model_name"] = args.det_model
    if args.rec_model:
        kwargs["text_recognition_model_name"] = args.rec_model
    ocr = PaddleOCR(**kwargs)

    doc = pymupdf.open(pdf)
    end = doc.page_count if args.max_pages <= 0 else min(doc.page_count, args.start_page + args.max_pages)
    pages = range(args.start_page, end)
    todo = [p for p in pages if p not in done]
    print(json.dumps({"pdf": str(pdf), "pages_total": doc.page_count,
                      "todo": len(todo), "done_already": len(done)}), flush=True)

    t0 = time.perf_counter()
    n = 0
    with jsonl_path.open("a", encoding="utf-8") as jf:
        for pno in todo:
            page = doc.load_page(pno)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(args.dpi / 72, args.dpi / 72))
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                arr = arr[:, :, ::-1]  # RGB -> BGR
            result = ocr.predict(arr)
            texts = collect_texts(result)
            rec = {"page": pno, "lines": len(texts), "chars": sum(len(t) for t in texts),
                   "text": "\n".join(texts)}
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            done[pno] = rec
            n += 1
            if n % 20 == 0 or pno == todo[-1]:
                el = time.perf_counter() - t0
                print(json.dumps({"done": n, "last_page": pno,
                                  "pages_per_s": round(n / el, 3)}), flush=True)
    doc.close()

    ordered = [done[p] for p in sorted(done)]
    md_path.write_text(
        "\n\n".join(f"<!-- page {r['page'] + 1} -->\n{r['text']}" for r in ordered),
        encoding="utf-8",
    )
    print(json.dumps({"pdf": str(pdf), "pages_done": len(ordered),
                      "elapsed_s": round(time.perf_counter() - t0, 1),
                      "md": str(md_path)}), flush=True)


if __name__ == "__main__":
    main()
