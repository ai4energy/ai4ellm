"""
预训练格式化模块

将富 chunk（来自结构化切块 + 质量评分）转换为预训练 JSONL。

支持两种格式：
- standard: {"text": "..."}  （TRL/HF 预训练兼容）
- rich:     {"text": "...", "meta": {source_file, source_type, section, category,
           chunk_index, quality_score, quality}}  （带溯源元数据）

category 字段从输入路径推断（如 origin-files-organized/102工程热力学/xxx.pdf → "102工程热力学"）。
"""

import os
import json
import hashlib
from typing import Iterable

from src.logger import get_logger

logger = get_logger()


def infer_category(source_file: str, input_root: str | None = None,
                   category_map: dict | None = None) -> str:
    """
    从 source_file 路径推断类目（输入目录的第一级子目录名）。

    例如 origin-files-organized/102工程热力学/xxx.pdf → "102工程热力学"

    source_file 可能是纯文件名 stem（chunk 里通常只存 stem，无路径），
    此时通过 category_map（stem→category）查表；查不到再回退到路径推断。
    """
    if not source_file:
        return "unknown"

    # 优先：stem → category 查表（chunk 的 source_file 通常是 stem）
    if category_map:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        # 去掉清洗器加的 _cleaned 后缀
        if stem.endswith("_cleaned"):
            stem = stem[:-len("_cleaned")]
        if stem in category_map:
            return category_map[stem]

    norm = os.path.normpath(source_file)
    parts = norm.split(os.sep)
    # 若指定了 input_root，取相对路径的第一级
    if input_root:
        try:
            rel = os.path.relpath(source_file, input_root)
            first = rel.split(os.sep)[0]
            if first and not os.path.isabs(first) and first != "." and not first.startswith(".."):
                return first
        except ValueError:
            pass
    # 否则取倒数第二段（文件名之上的目录）
    if len(parts) >= 2 and parts[-2] not in (".", "..", ""):
        return parts[-2]
    return "unknown"


def build_category_map(input_dir: str) -> dict:
    """
    扫描 input_dir，构建 {文件stem: 类目名} 映射。

    用于 chunk 的 source_file（纯 stem）反查 category。

    类目名来源（按优先级）：
    1. 文件相对 input_dir 的第一级子目录名（多类目批量场景）
    2. input_dir 本身的名字（单类目场景，比如 --input ../102工程热力学）
    3. 文件无相对子目录（直接位于 input_dir 下）也用 input_dir 的名字
    """
    cat_map = {}
    if not input_dir or not os.path.isdir(input_dir):
        return cat_map

    # fallback: 当 input_dir 本身没子目录（单类目场景），
    # 用 input_dir 的最后一段目录名作为类目
    fallback_cat = os.path.basename(os.path.normpath(input_dir))

    for root, dirs, files in os.walk(input_dir):
        # 相对 input_dir 的第一级子目录
        rel = os.path.relpath(root, input_dir)
        first = rel.split(os.sep)[0]
        category = first if first and first != "." else fallback_cat
        for fn in files:
            stem = os.path.splitext(fn)[0]
            cat_map[stem] = category
    return cat_map


def normalize_for_hash(text: str) -> str:
    """归一化文本用于规则去重 hash：小写 + 去所有空白。"""
    import re
    return re.sub(r"\s+", "", text).lower()


def content_hash(text: str) -> str:
    """归一化后的 MD5。"""
    return hashlib.md5(normalize_for_hash(text).encode("utf-8")).hexdigest()


def chunk_to_pretrain_record(
    chunk: dict,
    fmt: str = "rich",
    include_section_prefix: bool = True,
    input_root: str | None = None,
    category_map: dict | None = None,
) -> dict:
    """
    单个富 chunk → 预训练 record。

    chunk 字段（来自 text_splitter + quality_scorer）:
        content, section_title, full_path, source_file, source_type,
        chunk_index, char_count, quality_score?, quality?
    """
    content = (chunk.get("content") or "").strip()
    section = chunk.get("section_title") or chunk.get("section") or ""

    if include_section_prefix and section and not content.lstrip().startswith("#"):
        text = f"# {section}\n\n{content}"
    else:
        text = content

    if fmt == "standard":
        return {"text": text}

    # rich: 带 meta 溯源
    meta = {
        "source_file": chunk.get("source_file", ""),
        "source_type": chunk.get("source_type", "unknown"),
        "section": section,
        "full_path": chunk.get("full_path", section),
        "chunk_index": chunk.get("chunk_index", 0),
        "category": infer_category(chunk.get("source_file", ""), input_root, category_map),
    }
    if chunk.get("quality_score") is not None:
        meta["quality_score"] = chunk["quality_score"]
    if chunk.get("quality") is not None:
        meta["quality"] = chunk["quality"]
    return {"text": text, "meta": meta}


def chunks_to_pretrain_jsonl(
    chunks: Iterable[dict],
    output_path: str,
    fmt: str = "rich",
    min_length: int = 100,
    max_length: int = 1000000,
    include_section_prefix: bool = True,
    dedup: bool = True,
    input_root: str | None = None,
    category_map: dict | None = None,
) -> dict:
    """
    富 chunk 列表 → 预训练 JSONL 文件。

    过滤：content 长度 < min_length 或 > max_length 的跳过。
    去重（可选）：按归一化 content hash 去重，保留首个。

    返回统计 dict: {total, written, skipped_short, skipped_long, duplicates}
    """
    stats = {"total": 0, "written": 0, "skipped_short": 0, "skipped_long": 0, "duplicates": 0}
    seen = set()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            stats["total"] += 1
            content = (chunk.get("content") or "").strip()
            clen = len(content)
            if clen < min_length:
                stats["skipped_short"] += 1
                continue
            if clen > max_length:
                stats["skipped_long"] += 1
                continue
            if dedup:
                h = content_hash(content)
                if h in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(h)

            record = chunk_to_pretrain_record(
                chunk, fmt=fmt, include_section_prefix=include_section_prefix,
                input_root=input_root, category_map=category_map,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

    logger.info(
        f"预训练 JSONL 生成: {output_path} | "
        f"总计 {stats['total']} → 写入 {stats['written']} "
        f"(过短 {stats['skipped_short']}, 过长 {stats['skipped_long']}, 重复 {stats['duplicates']})"
    )
    return stats
