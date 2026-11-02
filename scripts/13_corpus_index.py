#!/usr/bin/env python3
"""语料索引：SHA-256 内容表 + 文件路径表（SQLite）。

子命令:
  build   从 data/corpus-inventory/manifest.jsonl 建索引
  ingest  扫描新目录，判断哪些 PDF 是新文件（size 预筛 + sha256），并入库
  stats   打印索引统计

用法:
  python3 -I scripts/13_corpus_index.py build
  python3 -I scripts/13_corpus_index.py ingest --input ~/uploads
"""
import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path

DB = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/corpus.sqlite")
MANIFEST = Path("/home/ai4e-hermes/ai4ellm/data/corpus-inventory/manifest.jsonl")
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS contents (
    sha256 TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    pages INTEGER,
    title TEXT,
    author TEXT,
    status TEXT NOT NULL DEFAULT 'discovered',
    text_chars INTEGER,
    chars_per_page REAL,
    needs_ocr INTEGER NOT NULL DEFAULT 0,
    extracted_path TEXT,
    ocr_path TEXT,
    first_seen TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime INTEGER,
    duplicate_of TEXT,
    first_seen TEXT,
    last_seen TEXT,
    FOREIGN KEY (sha256) REFERENCES contents(sha256)
);
CREATE INDEX IF NOT EXISTS idx_files_sha ON files(sha256);
CREATE INDEX IF NOT EXISTS idx_files_size ON files(size);
CREATE INDEX IF NOT EXISTS idx_contents_status ON contents(status);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def sha256_file(path: Path, buf: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def db() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)
    return conn


def build() -> None:
    conn = db()
    n_new = n_dup = 0
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ts = now()
            if r.get("duplicate_of"):
                conn.execute(
                    "INSERT OR REPLACE INTO files(path,sha256,name,size,mtime,duplicate_of,first_seen,last_seen) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (r["path"], r["sha256"], r["name"], r["size"], r.get("mtime"),
                     r["duplicate_of"], ts, ts),
                )
                n_dup += 1
                continue
            cur = conn.execute(
                "INSERT INTO contents(sha256,size,pages,title,author,status,first_seen,updated_at) "
                "VALUES(?,?,?,?,?,'discovered',?,?) "
                "ON CONFLICT(sha256) DO NOTHING",
                (r["sha256"], r["size"], r.get("pages"), r.get("title"),
                 r.get("author"), ts, ts),
            )
            n_new += cur.rowcount
            conn.execute(
                "INSERT OR REPLACE INTO files(path,sha256,name,size,mtime,duplicate_of,first_seen,last_seen) "
                "VALUES(?,?,?,?,?,NULL,?,?)",
                (r["path"], r["sha256"], r["name"], r["size"], r.get("mtime"), ts, ts),
            )
    conn.commit()
    print(json.dumps({"db": str(DB), "contents_inserted": n_new,
                      "duplicate_files": n_dup}, indent=2))


def ingest(input_dir: Path) -> None:
    conn = db()
    new_files = []
    dup_files = []
    for p in sorted(input_dir.rglob("*.pdf")):
        st = p.stat()
        row = conn.execute("SELECT sha256,size,mtime FROM files WHERE path=?", (str(p),)).fetchone()
        if row and row[1] == st.st_size:
            continue  # 已知路径且 size 未变
        # size 预筛：没有同 size 的内容，直接判新（仍需算 sha 作为索引键）
        same_size = conn.execute(
            "SELECT 1 FROM contents WHERE size=? LIMIT 1", (st.st_size,)
        ).fetchone()
        sha = sha256_file(p)
        known = conn.execute("SELECT 1 FROM contents WHERE sha256=?", (sha,)).fetchone()
        ts = now()
        if known:
            dup_of = conn.execute(
                "SELECT path FROM files WHERE sha256=? ORDER BY first_seen LIMIT 1", (sha,)
            ).fetchone()
            conn.execute(
                "INSERT OR REPLACE INTO files(path,sha256,name,size,mtime,duplicate_of,first_seen,last_seen) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (str(p), sha, p.name, st.st_size, int(st.st_mtime),
                 dup_of[0] if dup_of else None, ts, ts),
            )
            dup_files.append({"path": str(p), "sha256": sha, "duplicate_of": dup_of[0] if dup_of else None})
        else:
            conn.execute(
                "INSERT INTO contents(sha256,size,status,first_seen,updated_at) "
                "VALUES(?,?,'discovered',?,?)",
                (sha, st.st_size, ts, ts),
            )
            conn.execute(
                "INSERT OR REPLACE INTO files(path,sha256,name,size,mtime,duplicate_of,first_seen,last_seen) "
                "VALUES(?,?,?,?,?,NULL,?,?)",
                (str(p), sha, p.name, st.st_size, int(st.st_mtime), ts, ts),
            )
            new_files.append({"path": str(p), "sha256": sha, "size": st.st_size})
    conn.commit()
    print(json.dumps({"new_unique_files": new_files, "duplicates": dup_files},
                     ensure_ascii=False, indent=2))


def stats() -> None:
    conn = db()
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM contents GROUP BY status ORDER BY status"
    ).fetchall()
    files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    contents = conn.execute("SELECT COUNT(*) FROM contents").fetchone()[0]
    size = conn.execute("SELECT COALESCE(SUM(size),0) FROM contents").fetchone()[0]
    print(json.dumps({"db": str(DB), "files": files, "contents": contents,
                      "content_bytes": size, "by_status": dict(rows)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("ingest")
    p.add_argument("--input", required=True)
    sub.add_parser("stats")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "ingest":
        ingest(Path(args.input).expanduser())
    else:
        stats()


if __name__ == "__main__":
    main()
