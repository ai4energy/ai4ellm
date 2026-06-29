"""
JSONL 导出模块

将结构化 chunk 数据转换为 JSONL 格式。
优先从 chunks/_chunks.json 读取结构化数据，回退到从 txt 文件解析。
每行输出包含 section、content、source_file、quality 等丰富字段。
"""

import os
import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger()


def export_to_jsonl(
    input_dir: str,
    output_dir: str,
    ensure_ascii: bool = False,
) -> int:
    """
    将 chunk 数据转换为 JSONL 格式。

    优先策略：读取 chunks/_chunks.json（结构化数据，含来源追踪和质量评分）。
    回退策略：从 chunks/*.txt 或 cleaned_text/*.txt 解析纯文本。

    参数:
        input_dir: 输入的 txt/json 目录（通常是 chunks/ 或 cleaned_text/）
        output_dir: 输出的 JSONL 文件目录
        ensure_ascii: 是否转义非 ASCII 字符

    返回:
        输出的记录总数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 优先尝试结构化数据
    chunks_json = os.path.join(input_dir, "_chunks.json")
    if os.path.exists(chunks_json):
        return _export_structured_jsonl(chunks_json, output_dir, ensure_ascii)

    # 回退到旧版纯文本解析
    return _export_legacy_jsonl(input_dir, output_dir, ensure_ascii)


def _export_structured_jsonl(
    chunks_path: str,
    output_dir: str,
    ensure_ascii: bool,
) -> int:
    """从 _chunks.json 导出 JSONL。"""
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        logger.error(f"读取 _chunks.json 失败: {e}")
        return 0

    # 全部输出到单一 JSONL 文件
    output_file = os.path.join(output_dir, "corpus.jsonl")
    total = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = _build_record(chunk)
            if record["content"]:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
                total += 1

    logger.info(f"JSONL 导出完成: {output_file} ({total} 条记录)")
    return total


def _export_legacy_jsonl(
    input_dir: str,
    output_dir: str,
    ensure_ascii: bool,
) -> int:
    """从 txt 文件解析并导出 JSONL（旧版兼容）。"""
    input_path = Path(input_dir)
    txt_files = [p for p in input_path.iterdir() if p.suffix.lower() == ".txt" and p.name != "_chunks.json"]
    count = 0

    for txt_file in sorted(txt_files):
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            jsonl_data = []
            section_title = None
            content = []

            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    if section_title:
                        jsonl_data.append({
                            "section": section_title,
                            "content": "".join(content).strip(),
                        })
                    section_title = line.lstrip("#").strip()
                    content = []
                elif line:
                    content.append(line)

            if section_title and content:
                jsonl_data.append({
                    "section": section_title,
                    "content": "".join(content).strip(),
                })

            if jsonl_data:
                jsonl_file = os.path.join(output_dir, f"{txt_file.stem}.jsonl")
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for entry in jsonl_data:
                        f.write(json.dumps(entry, ensure_ascii=ensure_ascii) + "\n")
                count += len(jsonl_data)
        except Exception as e:
            logger.error(f"导出 JSONL 失败 {txt_file}: {e}")

    logger.info(f"JSONL 导出完成（旧版兼容模式）: {count} 条记录")
    return count


def _build_record(chunk: dict) -> dict:
    """
    从结构化 chunk 构建 JSONL 记录。

    输出字段：
    - section: 当前节标题
    - full_path: 完整层级路径
    - content: 正文内容
    - source_file: 来源文件名
    - source_type: 来源文件类型
    - chunk_index: chunk 序号
    - char_count: 字符数
    - quality_score: 综合质量评分（如有）
    - quality: 各维度评分详情（如有）
    """
    record = {
        "section": chunk.get("section_title", ""),
        "full_path": chunk.get("full_path", ""),
        "content": chunk.get("content", ""),
        "source_file": chunk.get("source_file", ""),
        "source_type": chunk.get("source_type", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "char_count": chunk.get("char_count", 0),
    }

    # 附加质量评分（如已计算）
    if "quality_score" in chunk:
        record["quality_score"] = chunk["quality_score"]
    if "quality" in chunk:
        record["quality"] = chunk["quality"]

    return record
