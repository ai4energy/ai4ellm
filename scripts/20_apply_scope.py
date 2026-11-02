#!/usr/bin/env python3
"""应用 scope 决策：keep 留在 books-processed，drop 移到 books-out-of-scope。

review 由 Qwen 结果决定；没有 Qwen 结果的 review 默认 KEEP（保守）。
默认 dry-run；--apply 移动。移动后可另行永久删除 out-of-scope。
"""
import argparse
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

DEC = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_decision.jsonl")
QWEN = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_qwen.jsonl")
OUT = Path("/home/ai4e-hermes/books-out-of-scope")
DB = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/corpus.sqlite")
LOG = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_final.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    dec = [json.loads(l) for l in DEC.open(encoding="utf-8")]
    qwen = {}
    if QWEN.exists():
        for line in QWEN.open(encoding="utf-8"):
            r = json.loads(line)
            qwen[r["path"]] = r["qwen"]

    final = []
    for r in dec:
        if r["decision"] == "keep":
            verdict = "keep"
        elif r["decision"] == "drop":
            verdict = "drop"
        else:
            verdict = "keep" if qwen.get(r["path"], "KEEP") == "KEEP" else "drop"
        final.append({**r, "final": verdict, "qwen": qwen.get(r["path"])})

    keeps = [r for r in final if r["final"] == "keep"]
    drops = [r for r in final if r["final"] == "drop"]
    LOG.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final),
                   encoding="utf-8")

    result = {
        "apply": args.apply, "total": len(final),
        "keep": len(keeps), "drop": len(drops),
        "drop_samples": [r["name"] for r in drops[:30]],
        "keep_samples": [r["name"] for r in keeps[:10]],
        "decision_file": str(LOG), "out_dir": str(OUT),
    }
    if args.apply:
        OUT.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB)
        cols = [x[1] for x in conn.execute("PRAGMA table_info(files)")]
        if "in_scope" not in cols:
            conn.execute("ALTER TABLE files ADD COLUMN in_scope INTEGER")
        if "out_of_scope_path" not in cols:
            conn.execute("ALTER TABLE files ADD COLUMN out_of_scope_path TEXT")
        moved = 0
        for r in final:
            src = Path(r["path"])
            conn.execute("UPDATE files SET in_scope=? WHERE path=?",
                         (1 if r["final"] == "keep" else 0, str(src)))
            if r["final"] != "drop" or not src.exists():
                continue
            dst = OUT / src.name
            if dst.exists():
                dst = OUT / f"{src.stem}-{r['path'][-8:]}{src.suffix}"
            try:
                os.rename(src, dst)
            except OSError:
                shutil.move(str(src), str(dst))
            conn.execute("UPDATE files SET out_of_scope_path=? WHERE path=?",
                         (str(dst), str(src)))
            conn.execute("INSERT INTO moves VALUES(?,?,?,?,?)",
                         (None, str(src), str(dst), "out_of_scope",
                          time.strftime("%Y-%m-%dT%H:%M:%S")))
            moved += 1
        conn.commit()
        conn.close()
        result["moved"] = moved
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
