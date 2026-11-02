#!/usr/bin/env python3
"""执行重命名/归档：unique → books-processed/，duplicate → books-duplicates/。

默认 dry-run；--apply 才移动。移动前检查全量抽取进程，避免改坏正在处理的路径。
所有移动写入 data/corpus-index/moves.jsonl，可用它回滚。
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

PLAN = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/rename_plan.jsonl")
MOVES = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/moves.jsonl")
DB = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/corpus.sqlite")
PROCESSED = Path("/home/ai4e-hermes/books-processed")
DUPLICATES = Path("/home/ai4e-hermes/books-duplicates")


def extraction_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-af", r"main.py --config .*corpus_full.yaml"],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


def truncate_utf8(name: str, max_bytes: int = 200) -> str:
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    stem, dot, ext = name.rpartition(".")
    suffix = (dot + ext) if dot else ""
    while stem and len((stem + suffix).encode("utf-8")) > max_bytes:
        stem = stem[:-1]
    return stem.rstrip("._-") + suffix


def no_spaces(name: str) -> str:
    name = name.replace("　", " ").replace("'", "").replace('"', "")
    name = re.sub(r"\s+", "_", name)
    return truncate_utf8(re.sub(r"_+", "_", name).strip(" ._-"))


def target_name(name: str, sha: str, dest_dir: Path) -> str:
    p = dest_dir / name
    if p.exists():
        p = p.with_name(f"{p.stem}-{sha[:8]}{p.suffix}")
    return p.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true", help="即使抽取在跑也移动（危险）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if extraction_running() and not args.force:
        raise SystemExit("全量抽取仍在运行；等它结束再加 --apply。")

    plan = [json.loads(l) for l in PLAN.open(encoding="utf-8")]
    if args.limit:
        plan = plan[:args.limit]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    DUPLICATES.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("CREATE TABLE IF NOT EXISTS moves("
                 "sha256 TEXT, original_path TEXT, new_path TEXT, kind TEXT, moved_at TEXT)")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(files)")]
    if "processed_path" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN processed_path TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_processed ON files(processed_path)")

    moves = []
    n_ok = n_skip = 0
    for p in plan:
        src = Path(p["original_path"])
        if not src.exists():
            n_skip += 1
            continue
        if p["is_duplicate"]:
            dest_dir = DUPLICATES
            name = target_name(no_spaces(src.name), p["sha256"], dest_dir)
            kind = "duplicate"
        else:
            dest_dir = PROCESSED
            name = target_name(p["new_name"], p["sha256"], dest_dir)
            kind = "processed"
        dst = dest_dir / name
        if dst.exists():
            n_skip += 1
            continue
        if args.apply:
            try:
                os.rename(src, dst)
            except OSError:
                shutil.move(str(src), str(dst))
            moves.append({"sha256": p["sha256"], "original_path": str(src),
                          "new_path": str(dst), "kind": kind,
                          "moved_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            conn.execute("INSERT INTO moves VALUES(?,?,?,?,?)",
                         (p["sha256"], str(src), str(dst), kind, moves[-1]["moved_at"]))
            conn.execute("UPDATE files SET processed_path=? WHERE path=?",
                         (str(dst), str(src)))
            conn.commit()
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"progress {n_ok}/{len(plan)}", flush=True)

    if args.apply and moves:
        with MOVES.open("a", encoding="utf-8") as f:
            for m in moves:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    conn.close()
    print(json.dumps({"apply": args.apply, "planned": len(plan), "would_move_or_moved": n_ok,
                      "skipped": n_skip, "processed_dir": str(PROCESSED),
                      "duplicates_dir": str(DUPLICATES), "moves_log": str(MOVES)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
