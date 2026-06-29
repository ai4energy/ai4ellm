"""
JSON 导出模块

将结构化 chunk 数据导出为单个 JSON 数组文件。
优先从 chunks/_chunks.json 读取结构化数据，回退到从 txt 文件解析。
每条记录包含 section、full_path、content、source_file、quality 等丰富字段。
"""

import os
import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger()


def export_to_json(
    input_dir: str,
    output_file: str,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> int:
    """
    将 chunk 数据导出为 JSON 数组格式。

    优先策略：读取 chunks/_chunks.json（结构化数据）。
    回退策略：从 txt 文件解析纯文本。

    参数:
        input_dir: 输入的 txt/json 目录
        output_file: 输出的 JSON 文件路径
        ensure_ascii: 是否转义非 ASCII 字符
        indent: JSON 缩进空格数

    返回:
        输出的记录总数
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # 优先尝试结构化数据
    chunks_json = os.path.join(input_dir, "_chunks.json")
    if os.path.exists(chunks_json):
        return _export_structured_json(chunks_json, output_file, ensure_ascii, indent)

    # 回退到旧版纯文本解析
    return _export_legacy_json(input_dir, output_file, ensure_ascii, indent)


def _export_structured_json(
    chunks_path: str,
    output_file: str,
    ensure_ascii: bool,
    indent: int,
) -> int:
    """从 _chunks.json 导出 JSON 数组。"""
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.error(f"读取 _chunks.json 失败: {e}")
        return 0

    all_data = []
    for chunk in chunks:
        record = _build_record(chunk)
        if record["content"]:
            all_data.append(record)

    # 附加源数据信息
    metadata = _collect_source_metadata(chunks)

    output = {
        "metadata": metadata,
        "records": all_data,
        "total_records": len(all_data),
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=ensure_ascii, indent=indent)

    logger.info(f"JSON 导出完成: {output_file} ({len(all_data)} 条记录)")
    return len(all_data)


def _export_legacy_json(
    input_dir: str,
    output_file: str,
    ensure_ascii: bool,
    indent: int,
) -> int:
    """从 txt 文件解析并导出 JSON（旧版兼容）。"""
    input_path = Path(input_dir)
    txt_files = [p for p in input_path.iterdir() if p.suffix.lower() == ".txt"]
    count = 0
    all_data = []

    for txt_file in sorted(txt_files):
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            section_title = None
            content = []

            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    if section_title:
                        all_data.append({
                            "source_file": txt_file.name,
                            "section": section_title,
                            "content": "".join(content).strip(),
                        })
                    section_title = line.lstrip("#").strip()
                    content = []
                elif line:
                    content.append(line)

            if section_title and content:
                all_data.append({
                    "source_file": txt_file.name,
                    "section": section_title,
                    "content": "".join(content).strip(),
                })

            count += 1
        except Exception as e:
            logger.error(f"导出 JSON 失败 {txt_file}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=ensure_ascii, indent=indent)

    logger.info(f"JSON 导出完成（旧版兼容模式）: {count}/{len(txt_files)} 个文件，共 {len(all_data)} 条记录")
    return count


def _build_record(chunk: dict) -> dict:
    """从结构化 chunk 构建 JSON 记录。"""
    record = {
        "section": chunk.get("section_title", ""),
        "full_path": chunk.get("full_path", ""),
        "content": chunk.get("content", ""),
        "source_file": chunk.get("source_file", ""),
        "source_type": chunk.get("source_type", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "char_count": chunk.get("char_count", 0),
    }

    if "quality_score" in chunk:
        record["quality_score"] = chunk["quality_score"]
    if "quality" in chunk:
        record["quality"] = chunk["quality"]

    return record


def _collect_source_metadata(chunks: list[dict]) -> dict:
    """
    从 chunk 列表中收集文档级元数据摘要。

    返回：
    {
        "source_files": ["file1.pdf", "file2.docx", ...],
        "source_types": {"pdf": 3, "docx": 1, ...},
        "total_chars": 123456,
        "avg_quality_score": 0.75,
    }
    """
    source_files = set()
    source_types: dict[str, int] = {}
    total_chars = 0
    quality_scores = []

    for c in chunks:
        sf = c.get("source_file", "")
        if sf:
            source_files.add(sf)
        st = c.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1
        total_chars += c.get("char_count", 0)
        if "quality_score" in c:
            quality_scores.append(c["quality_score"])

    return {
        "source_files": sorted(source_files),
        "source_types": source_types,
        "total_chars": total_chars,
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 4) if quality_scores else None,
    }
