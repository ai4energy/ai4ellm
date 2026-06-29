#!/usr/bin/env python3
"""
AI4ELLM 文件索引工具
- 构建 SHA256 索引
- 检查重复文件
- 快速查询文件是否已存在
"""

import hashlib
import json
import os
import sys
from pathlib import Path

BASE_DIR = "/data/robotlele/ai4ellm"
INDEX_FILE = os.path.join(BASE_DIR, "file-index.json")
ORIGIN_DIR = os.path.join(BASE_DIR, "origin-files")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_quick(filepath):
    """Fast: hash first 64KB + file size + filename."""
    stat = os.stat(filepath)
    size = stat.st_size
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read(65536))
    # Also hash last 64KB if file is large
    if size > 131072:
        f.seek(max(0, size - 65536))
        h.update(f.read(65536))
    return f"{h.hexdigest()}_{size}"


def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r") as f:
            return json.load(f)
    return {"files": {}, "hashes": {}}


def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def build_index():
    """Scan all files in origin-files and build SHA256 index."""
    index = {"files": {}, "hashes": {}}
    total = 0
    dupes = 0

    print("📚 正在扫描文件并计算 SHA256...")
    for root, dirs, files in os.walk(ORIGIN_DIR):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            filepath = os.path.join(root, fn)
            rel_path = os.path.relpath(filepath, BASE_DIR)
            file_hash = sha256_file(filepath)
            size = os.path.getsize(filepath)

            # Store by file path
            index["files"][rel_path] = {
                "sha256": file_hash,
                "size": size,
                "size_human": _human_size(size),
            }

            # Track by hash for duplicate detection
            if file_hash in index["hashes"]:
                index["hashes"][file_hash].append(rel_path)
                dupes += 1
            else:
                index["hashes"][file_hash] = [rel_path]

            total += 1

    save_index(index)

    print(f"\n✅ 索引构建完成!")
    print(f"  总文件数: {total}")
    print(f"  重复文件: {dupes}")
    print(f"  唯一文件: {total - dupes}")
    print(f"  索引文件: {INDEX_FILE}")
    print(f"  索引大小: {_human_size(os.path.getsize(INDEX_FILE))}")

    # Show duplicates
    dupes_found = {h: paths for h, paths in index["hashes"].items() if len(paths) > 1}
    if dupes_found:
        print(f"\n🔍 发现 {len(dupes_found)} 组重复:")
        for h, paths in sorted(dupes_found.items()):
            print(f"  {h[:16]}...:")
            for p in paths:
                print(f"    - {p}")

    return index


def check_files(filepaths):
    """Check if given files already exist in the index."""
    index = load_index()
    if not index["files"]:
        print("⚠️  索引为空，先运行 build 构建索引。")
        return

    for filepath in filepaths:
        # Try as absolute path or relative to BASE_DIR
        if os.path.isabs(filepath):
            rel_path = os.path.relpath(filepath, BASE_DIR)
        else:
            rel_path = filepath

        if rel_path in index["files"]:
            info = index["files"][rel_path]
            print(f"  ✅ 已存在: {rel_path} ({info['size_human']})")
        else:
            # Check by SHA256
            if os.path.exists(filepath):
                file_hash = sha256_file(filepath)
                if file_hash in index["hashes"]:
                    existing = index["hashes"][file_hash]
                    print(f"  🔄 重复! {filepath}")
                    print(f"     已存在于: {', '.join(existing)}")
                else:
                    print(f"  🆕 新文件: {filepath}")
            else:
                print(f"  ❓ 未找到: {filepath}")


def find_duplicates():
    """Show all duplicate files."""
    index = load_index()
    dupes = {h: paths for h, paths in index["hashes"].items() if len(paths) > 1}

    if not dupes:
        print("✅ 没有发现重复文件。")
        return

    print(f"🔍 发现 {len(dupes)} 组重复文件:\n")
    for h, paths in sorted(dupes.items()):
        print(f"  SHA256: {h[:32]}...")
        for p in paths:
            size = index["files"][p]["size_human"]
            print(f"    - {p} ({size})")
        print()


def scan_new():
    """Scan origin-files and find files not yet in index."""
    index = load_index()
    new_files = []

    for root, dirs, files in os.walk(ORIGIN_DIR):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            filepath = os.path.join(root, fn)
            rel_path = os.path.relpath(filepath, BASE_DIR)

            if rel_path not in index["files"]:
                new_files.append(rel_path)

    if new_files:
        print(f"🆕 发现 {len(new_files)} 个新文件（不在索引中）:")
        for f in new_files:
            print(f"  {f}")
    else:
        print("✅ 所有文件都在索引中，没有新增。")

    return new_files


def _human_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def print_usage():
    print("""
AI4ELLM 文件索引工具

用法:
  python3 file_index.py build          # 构建/更新索引
  python3 file_index.py check <文件>    # 检查文件是否已存在
  python3 file_index.py dupes          # 查找重复文件
  python3 file_index.py new            # 扫描新增文件
  python3 file_index.py stats          # 统计信息

示例:
  # 构建索引
  python3 file_index.py build

  # 检查一个文件
  python3 file_index.py check origin-files/001流体力学/xxx.pdf

  # 检查多个文件
  python3 file_index.py check file1.pdf file2.pdf file3.pdf
""")


def show_stats():
    index = load_index()
    total_files = len(index["files"])
    total_size = sum(f["size"] for f in index["files"].values())
    unique_hashes = len(index["hashes"])
    dupes = total_files - unique_hashes

    # Category stats
    cats = {}
    for path, info in index["files"].items():
        parts = path.split("/")
        if len(parts) >= 2:
            cat = parts[1]
            cats.setdefault(cat, {"count": 0, "size": 0})
            cats[cat]["count"] += 1
            cats[cat]["size"] += info["size"]

    print(f"📊 索引统计")
    print(f"  总文件: {total_files}")
    print(f"  唯一文件: {unique_hashes}")
    print(f"  重复文件: {dupes}")
    print(f"  总大小: {_human_size(total_size)}")
    print()
    print("  分类统计:")
    for cat in sorted(cats.keys()):
        info = cats[cat]
        print(f"    {cat}: {info['count']} 文件, {_human_size(info['size'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "build":
        build_index()
    elif cmd == "check":
        check_files(sys.argv[2:])
    elif cmd == "dupes":
        find_duplicates()
    elif cmd == "new":
        scan_new()
    elif cmd == "stats":
        show_stats()
    else:
        print_usage()
