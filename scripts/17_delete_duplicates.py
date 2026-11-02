#!/usr/bin/env python3
"""永久删除 books-duplicates/ 中的重复文件。

安全规则:
  1. 只处理 processed_path 位于 books-duplicates/ 的记录；
  2. 必须在 books-processed/ 找到同 sha256 的保留文件；
  3. 重新计算两边 sha256，一致才 unlink；
  4. 写 deleted_duplicates.jsonl + SQLite deleted_at。
默认 dry-run；--apply 执行。
"""
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

DB = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/corpus.sqlite")
LOG = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/deleted_duplicates.jsonl")
DUP_DIR = "/home/ai4e-hermes/books-duplicates/"
KEEP_DIR = "/home/ai4e-hermes/books-processed/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(DB)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(files)")]
    if "deleted_at" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN deleted_at TEXT")
    rows = conn.execute(
        "SELECT sha256, path, processed_path FROM files "
        "WHERE processed_path LIKE ?", (DUP_DIR + "%",)
    ).fetchall()

    planned = []
    skipped = []
    for sha, orig, dup_path in rows:
        dup = Path(dup_path)
        keep = conn.execute(
            "SELECT processed_path FROM files WHERE sha256=? AND processed_path LIKE ? "
            "AND processed_path != ? LIMIT 1",
            (sha, KEEP_DIR + "%", dup_path),
        ).fetchone()
        if not dup.exists() or not keep or not Path(keep[0]).exists():
            skipped.append({"dup": dup_path, "reason": "missing_dup_or_keep"})
            continue
        if sha256(dup) != sha or sha256(Path(keep[0])) != sha:
            skipped.append({"dup": dup_path, "reason": "hash_mismatch"})
            continue
        planned.append({"sha256": sha, "duplicate_path": str(dup),
                        "kept_path": keep[0], "size": dup.stat().st_size})

    freed = sum(p["size"] for p in planned)
    if args.apply:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with LOG.open("a", encoding="utf-8") as f:
            for p in planned:
                Path(p["duplicate_path"]).unlink()
                p["deleted_at"] = ts
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                conn.execute("UPDATE files SET deleted_at=? WHERE processed_path=?",
                             (ts, p["duplicate_path"]))
            conn.commit()
    print(json.dumps({"apply": args.apply, "deleted": len(planned),
                      "skipped": skipped, "freed_bytes": freed,
                      "freed_gb": round(freed / 1e9, 3),
                      "deleted_log": str(LOG)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
