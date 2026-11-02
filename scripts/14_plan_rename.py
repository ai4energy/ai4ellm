#!/usr/bin/env python3
"""生成 PDF 重命名计划：年-第一作者-题目-出版社.pdf

只读，不移动文件。输出:
  data/corpus-index/rename_plan.jsonl
  data/corpus-index/rename_report.json

元数据来源: PDF 内嵌 metadata（manifest 里的 title/author）+ 文件名启发式。
年份优先取文件名最后 (YYYY)，否则取 PDF creation_date 年份，否则 0000。
出版社从文件名/标题匹配已知出版社，否则 Unknown。
"""
import argparse
import json
import re
from pathlib import Path

MANIFEST = Path("/home/ai4e-hermes/ai4ellm/data/corpus-inventory/manifest.jsonl")
OUT_DIR = Path("/home/ai4e-hermes/ai4ellm/data/corpus-index")

YEAR_PAREN = re.compile(r"\((19\d{2}|20\d{2})\)")
YEAR_ANY = re.compile(r"\b(19\d{2}|20\d{2})\b")
ZLIB = re.compile(r"[-_ ]*z-?library[^.]*$|[-_ ]*1lib[^.]*$|[-_ ]*libgen[^.]*$", re.I)
PUBLISHERS = [
    "Springer", "Elsevier", "Wiley", "CRC Press", "Taylor & Francis", "McGraw-Hill",
    "Pearson", "Cengage", "ASHRAE", "Academic Press", "Cambridge University Press",
    "Oxford University Press", "IEEE", "Butterworth-Heinemann", "Pergamon",
    "中国电力出版社", "化学工业出版社", "机械工业出版社", "科学出版社",
    "高等教育出版社", "中国建筑工业出版社", "中国石化出版社", "石油工业出版社",
]


def clean_text(s: str) -> str:
    s = (s or "").replace("\u0000", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip(" ._-")
    return s


def valid_meta(s: str | None) -> bool:
    if not s:
        return False
    s = s.strip()
    if len(s) < 2 or len(s) > 120:
        return False
    if s.lower() in {"unknown", "untitled", "author", "administrator", "user"}:
        return False
    # 乱码检测：任何控制字符/私用区/替换符都拒绝
    if any(ord(ch) < 32 or 0xE000 <= ord(ch) <= 0xF8FF or ch == "�" for ch in s):
        return False
    # 明显是损坏编码的符号
    if re.search(r"[<>@]{1,}", s):
        return False
    # 字母/CJK 比例过低（乱码串通常符号+数字占多数）
    letters = sum(1 for ch in s if ch.isalpha() or "一" <= ch <= "鿿")
    if letters < max(2, len(s) * 0.4):
        return False
    return True


def pick_year(stem: str, meta: dict) -> tuple[int | None, bool]:
    matches = YEAR_PAREN.findall(stem)
    if matches:
        return int(matches[-1]), True
    m = YEAR_ANY.search(stem)
    if m:
        return int(m.group(1)), True
    cd = meta.get("creation_date") or ""
    m = YEAR_ANY.search(cd)
    if m:
        return int(m.group(1)), False
    return None, False


def pick_author(stem: str, meta: dict) -> tuple[str, bool]:
    author = clean_text(meta.get("author"))
    if valid_meta(author):
        # 去掉 "by "、编辑角色
        author = re.sub(r"^(by|edited by|author[s]?:?)\s+", "", author, flags=re.I)
        author = re.split(r"[;,]", author)[0].strip()
        if author:
            return author[:60], True

    s = YEAR_PAREN.sub("", stem)
    s = ZLIB.sub("", s)
    # "064 - Authors - Title ..."（编号-作者-题目）
    m = re.match(r"^\s*\d{2,5}\s*-\s*([^-]{2,100}?)\s*-\s*(.+)$", s)
    if m:
        cand = m.group(1).strip(" ._")
        if valid_meta(cand):
            return cand[:60], False
    # "Title - Author" / "Title - A, B - Publisher"
    parts = [p.strip() for p in re.split(r"\s+-\s+|\s*_\s*", s) if p.strip()]
    if len(parts) >= 2:
        last = parts[-1]
        last = re.sub(r"\((19|20)\d{2}\)", "", last).strip(" ._-")
        last = re.sub(r"^(by|edited by)\s+", "", last, flags=re.I)
        if 2 <= len(last) <= 80 and not re.match(r"^\d+$", last):
            return last[:60], False

    # "Title-Author" 无空格：取最后一个连字符后的名字片段
    m = re.match(r"^(.+?)[-–—]([^-–—]{2,60})$", s)
    if m:
        cand = m.group(2).strip(" ._")
        if 2 <= len(cand) <= 60:
            return cand[:60], False
    return "Unknown", False


def pick_title(stem: str, meta: dict, author: str) -> tuple[str, bool]:
    title = clean_text(meta.get("title"))
    if valid_meta(title):
        return title[:120], True
    s = YEAR_PAREN.sub("", stem)
    s = ZLIB.sub("", s)
    m = re.match(r"^\s*\d{2,5}\s*-\s*[^-]{2,100}?\s*-\s*(.+)$", s)
    if m:
        cand = clean_text(m.group(1).split(" - ")[0])
        if len(cand) >= 5:
            return cand[:120], False
    s = re.sub(r"^\s*\d{2,5}\s*[-–—]\s*", "", s)  # 去掉开头编号
    s = s.replace("_", " ").strip(" ._-")
    # 去掉末尾作者片段
    if author and author != "Unknown":
        pat = re.escape(author.split()[0]) if author.split() else None
        if pat:
            s = re.split(r"\s+-\s+" + pat, s)[0]
    s = clean_text(s)
    return (s[:120] or "Untitled"), False


def pick_publisher(stem: str, title: str, meta: dict) -> tuple[str, bool]:
    blob = f"{stem} {title}"
    for p in PUBLISHERS:
        if p.lower() in blob.lower():
            return p, True
    producer = clean_text(meta.get("producer"))
    for p in PUBLISHERS:
        if p.lower() in producer.lower():
            return p, True
    return "Unknown", False


def truncate_utf8(name: str, max_bytes: int = 200) -> str:
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    stem, dot, ext = name.rpartition(".")
    suffix = (dot + ext) if dot else ""
    while stem and len((stem + suffix).encode("utf-8")) > max_bytes:
        stem = stem[:-1]
    return stem.rstrip("._-") + suffix


def sanitize(name: str) -> str:
    name = name.replace("　", " ")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", name)
    name = name.replace("'", "").replace('"', "")
    name = re.sub(r"\s+", "_", name)          # 用户要求：文件名不使用空格
    name = re.sub(r"_+", "_", name).strip(" ._-")
    return truncate_utf8(name, 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--enrich", default=str(OUT_DIR / "enrichment.jsonl"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    enrich: dict[str, dict] = {}
    ep = Path(args.enrich)
    if ep.exists():
        for line in ep.open(encoding="utf-8"):
            r = json.loads(line)
            enrich[r["sha256"]] = r

    rows = [json.loads(line) for line in Path(args.manifest).open(encoding="utf-8")]
    # 只保留每个 sha256 的第一个路径（重复文件单独处理）
    seen: dict[str, str] = {}
    plan = []
    for r in rows:
        sha = r["sha256"]
        is_dup = sha in seen
        if not is_dup:
            seen[sha] = r["path"]
        stem = Path(r["name"]).stem
        year, year_from_name = pick_year(stem, r)
        author, author_from_meta = pick_author(stem, r)
        title, title_from_meta = pick_title(stem, r, author)
        publisher, pub_from_name = pick_publisher(stem, title, r)
        enr = enrich.get(sha) or {}
        if publisher == "Unknown" and enr.get("publisher"):
            publisher = enr["publisher"]
            pub_from_name = True
        if year is None and enr.get("year"):
            year = enr["year"]
        base = f"{year or '0000'}-{author}-{title}-{publisher}.pdf"
        plan.append({
            "original_path": r["path"],
            "sha256": sha,
            "size": r["size"],
            "is_duplicate": is_dup,
            "duplicate_of": seen[sha] if is_dup else None,
            "year": year,
            "author": author,
            "title": title,
            "publisher": publisher,
            "new_name": sanitize(base),
            "flags": {
                "year_from_name": year_from_name,
                "author_from_meta": author_from_meta,
                "title_from_meta": title_from_meta,
                "publisher_from_name": pub_from_name,
            },
        })

    # 冲突处理：同目标名追加 sha 短码
    used: dict[str, int] = {}
    for p in plan:
        key = p["new_name"].lower()
        used[key] = used.get(key, 0) + 1
        if used[key] > 1:
            stem, dot, ext = p["new_name"].rpartition(".")
            p["new_name"] = f"{stem}-{p['sha256'][:8]}.{ext}"

    plan_path = out / "rename_plan.jsonl"
    with plan_path.open("w", encoding="utf-8") as f:
        for p in plan:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    unique = [p for p in plan if not p["is_duplicate"]]
    report = {
        "rows": len(plan),
        "unique_files": len(unique),
        "duplicate_rows": len(plan) - len(unique),
        "year_known": sum(1 for p in unique if p["year"]),
        "author_known": sum(1 for p in unique if p["author"] != "Unknown"),
        "publisher_known": sum(1 for p in unique if p["publisher"] != "Unknown"),
        "title_from_meta": sum(1 for p in unique if p["flags"]["title_from_meta"]),
        "plan": str(plan_path),
        "samples": [
            {k: p[k] for k in ("new_name", "year", "author", "title", "publisher")}
            for p in unique[:10]
        ],
    }
    (out / "rename_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
