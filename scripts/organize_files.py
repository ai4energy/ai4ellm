#!/usr/bin/env python3
"""
Deduplicate and organize files in origin-files/.
1. Find and remove exact duplicates (keep one copy)
2. Create organized folder: origin-files-organized/
3. Copy unique files with consistent naming
4. Keep unprocessed files in place, delete processed originals
"""

import os
import shutil
import hashlib
from collections import defaultdict

BASE = "/data/robotlele/ai4ellm/origin-files"
ORGANIZED = "/data/robotlele/ai4ellm/origin-files-organized"

# Category mapping: folder number -> clean category name
CATEGORY_MAP = {
    "01": "01-流体力学",
    "02": "02-工程热力学",
    "03": "03-传热学",
    "04": "04-燃烧学",
    "05": "05-固体与半导体物理",
    "06": "06-材料科学与新能源材料",
    "11": "11-生物质能转化",
    "12": "12-新能源热利用与热发电",
    "13": "13-氢能与新型能源动力系统",
    "14": "14-光电与电化学",
    "15": "15-流体机械能转化",
    "21": "21-新能源发电并网",
    "22": "22-储能原理与技术",
    "23": "23-能源系统数字化与智能化",
    "24": "24-智慧能源",
    "25": "25-自动控制原理",
    "26": "26-能源系统工程与热力发电厂",
    "27": "27-电气与锅炉",
    "28": "28-汽轮机原理",
    "29": "29-热工与核电厂",
    "30": "30-环境",
}

# Sub-category keywords for 26, 27
SUB_KEYWORDS = {
    "26": {"热力发电厂": "热力发电厂", "能源系统工程": "能源系统工程"},
    "27": {"电气": "电气", "锅炉": "锅炉"},
}


def md5_file(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_all_files(base_dir):
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for fn in filenames:
            if fn.startswith(".") or fn == ".DS_Store":
                continue
            files.append(os.path.join(root, fn))
    return files


def deduplicate(all_files):
    """Find duplicates, keep one copy of each."""
    hash_map = defaultdict(list)
    for f in all_files:
        try:
            h = md5_file(f)
            hash_map[h].append(f)
        except Exception as e:
            print(f"  ⚠️  Error reading {f}: {e}")

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    unique_files = set()
    removed_files = set()

    for h, paths in duplicates.items():
        # Keep the one with the best name (longest, most info)
        best = max(paths, key=lambda p: len(os.path.basename(p)))
        unique_files.add(best)
        for p in paths:
            if p != best:
                removed_files.add(p)

    for f in all_files:
        if f not in unique_files and f not in removed_files:
            unique_files.add(f)

    return unique_files, removed_files


def get_category(folder_name):
    """Map folder to category."""
    # Extract leading number
    import re
    m = re.match(r"(\d+)", folder_name)
    if m:
        num = m.group(1)
        if num in CATEGORY_MAP:
            return CATEGORY_MAP[num], num
    # Fallback: use folder name as-is
    return folder_name, None


def normalize_filename(filename):
    """Ensure filename follows: 年份-作者-书名-出版社.后缀"""
    # Already in good format for most files
    return filename


def organize_files(unique_files, removed_files):
    """Copy unique files to organized folder with category structure."""
    os.makedirs(ORGANIZED, exist_ok=True)

    # Track stats
    stats = defaultdict(int)
    total_size = 0

    for filepath in sorted(unique_files):
        rel_path = os.path.relpath(filepath, BASE)
        parts = rel_path.split(os.sep)

        if len(parts) < 2:
            continue

        folder_name = parts[0]
        filename = parts[-1]

        category, cat_num = get_category(folder_name)

        # Create category dir
        cat_dir = os.path.join(ORGANIZED, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Determine sub-category for 26 and 27
        sub_cat = None
        if cat_num in ("26", "27") and len(parts) >= 3:
            parent = parts[-2]
            for keyword, sub_name in SUB_KEYWORDS.get(cat_num, {}).items():
                if keyword in parent or keyword in folder_name:
                    sub_cat = f"{cat_num}-{sub_name}"
                    break

        if sub_cat:
            target_dir = os.path.join(ORGANIZED, sub_cat)
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = cat_dir

        target_path = os.path.join(target_dir, filename)

        # Skip if already exists (duplicate across categories)
        if os.path.exists(target_path):
            continue

        try:
            shutil.copy2(filepath, target_path)
            stats[category] += 1
            total_size += os.path.getsize(filepath)
        except Exception as e:
            print(f"  ⚠️  Error copying {filepath}: {e}")

    return stats, total_size


def delete_originals(removed_files, organized_stats):
    """Delete duplicate originals."""
    for f in removed_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"  ⚠️  Error removing {f}: {e}")
    print(f"  ✅ Deleted {len(removed_files)} duplicate files")


def main():
    print("=" * 60)
    print("📚 AI4ELLM 语料整理: 去重 + 分类 + 组织")
    print("=" * 60)

    # Step 1: Find all files
    print("\n📂 扫描文件...")
    all_files = find_all_files(BASE)
    print(f"  共 {len(all_files)} 个文件")

    # Step 2: Deduplicate
    print("\n🔍 去重中...")
    unique_files, removed_files = deduplicate(all_files)
    print(f"  唯一文件: {len(unique_files)}")
    print(f"  重复文件: {len(removed_files)}")

    # Step 3: Organize
    print("\n📋 整理到 origin-files-organized/ ...")
    stats, total_size = organize_files(unique_files, removed_files)

    print("\n📊 分类统计:")
    for cat, count in sorted(stats.items()):
        print(f"  {cat}: {count} 个文件")
    print(f"\n  总计: {sum(stats.values())} 个文件, {total_size / 1024**3:.1f} GB")

    # Step 4: Delete duplicates
    print("\n🗑️  删除重复文件...")
    delete_originals(removed_files, stats)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("✅ 整理完成!")
    print("=" * 60)
    print(f"  原始目录: {BASE}")
    print(f"  整理目录: {ORGANIZED}")
    print(f"  重复删除: {len(removed_files)} 个")
    print(f"  剩余文件: {len(unique_files)} 个")


if __name__ == "__main__":
    main()
