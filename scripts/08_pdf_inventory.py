#!/usr/bin/env python3
"""PDF 清单：扫描 ~/books，计算 sha256/大小/页数/元数据，输出 manifest + 重复项。

原件只读，不移动、不重命名。用法:
  python3 -I scripts/08_pdf_inventory.py [--input ~/books] [--out data/corpus-inventory]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ai4e-hermes/ai4ellm/.venv/lib/python3.12/site-packages")
import fitz  # noqa: E402


def sha256_file(path: Path, buf: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def pdf_meta(path: Path) -> dict:
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        out = {
            "pages": doc.page_count,
            "encrypted": bool(doc.needs_pass),
            "title": (meta.get("title") or "").strip() or None,
            "author": (meta.get("author") or "").strip() or None,
        }
        doc.close()
        return out
    except Exception as exc:  # noqa: BLE001
        return {"pages": None, "encrypted": None, "title": None, "author": None,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path.home() / "books"))
    ap.add_argument("--out", default="/home/ai4e-hermes/ai4ellm/data/corpus-inventory")
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    dup_path = out / "duplicates.jsonl"

    pdfs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    print(f"scan {src}: {len(pdfs)} pdfs", flush=True)

    seen: dict[str, str] = {}
    dups = []
    t0 = time.time()
    done_bytes = 0
    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, p in enumerate(pdfs, 1):
            st = p.stat()
            done_bytes += st.st_size
            rec = {
                "path": str(p),
                "name": p.name,
                "size": st.st_size,
                "sha256": sha256_file(p),
                "mtime": int(st.st_mtime),
            }
            rec.update(pdf_meta(p))
            rec["duplicate_of"] = seen.get(rec["sha256"])
            if rec["duplicate_of"] is None:
                seen[rec["sha256"]] = str(p)
            else:
                dups.append(rec)
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            mf.flush()
            if i % 100 == 0 or i == len(pdfs):
                el = time.time() - t0
                print(f"{i}/{len(pdfs)}  {el:.0f}s  "
                      f"{done_bytes / 1e9:.1f}GB  dups={len(dups)}", flush=True)

    with dup_path.open("w", encoding="utf-8") as f:
        for rec in dups:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"done: {len(pdfs)} pdfs, {len(dups)} duplicates, "
          f"{time.time() - t0:.0f}s", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"duplicates: {dup_path}", flush=True)


if __name__ == "__main__":
    main()
