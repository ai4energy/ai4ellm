"""
纯文本清洗模块

对非 Markdown 格式的纯文本进行行级清洗：
- 去除空行、乱码、水印行
- 修复断行（行末无标点的下一行拼接）
- 去除页码、图表编号、交叉引用等模式
- 输出结构化 _sections.json（用于后续智能切分）
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass

from src.cleaners.rules import (
    INLINE_CITATION_RE, AUTHOR_YEAR_RE, GARBLE_RE,
    PAGE_NUM_TRAILING_RE, PAGE_NUM_TRAILING2_RE,
    TABLE_RE, FIGURE_RE, TABLE_EN_RE, FIGURE_EN_RE,
    CROSS_REF_CN, CROSS_REF_EN, REF_PAREN_RE,
    URL_LINE_RE, WATERMARK_XJTU, CN_WORD_RE,
    DEFAULT_WATERMARK_KEYWORDS,
)
from src.logger import get_logger

logger = get_logger()


def clean_text_lines(
    content: list[str],
    watermark_keywords: list[str] | None = None,
    fix_broken_lines: bool = True,
) -> list[str]:
    """
    对纯文本进行行级清洗。

    1. 去除图片/链接/水印行
    2. 去除页码、图表编号、交叉引用
    3. 修复断行（行末无标点时与下一行拼接）
    4. 去除乱码和空行

    参数:
        content: 原始文本行列表
        watermark_keywords: 水印关键词列表
        fix_broken_lines: 是否修复断行

    返回:
        清洗后的行列表
    """
    if watermark_keywords is None:
        watermark_keywords = DEFAULT_WATERMARK_KEYWORDS

    cleaned = []

    for raw in content:
        line = raw.rstrip()

        # 空行处理
        if not line.strip():
            if cleaned and cleaned[-1] == "":
                continue  # 连续空行只保留一个
            cleaned.append("")
            continue

        # 图片/链接行
        if line.lstrip().startswith("![") or "](" in line:
            continue

        # 水印
        if any(key in line.lower() for key in watermark_keywords):
            continue

        # CIP / 版编目
        stripped = line.strip()
        if "图书在版编目" in stripped or "CIP" in stripped:
            continue

        # URL 整行
        if URL_LINE_RE.match(stripped):
            continue

        # 行内引用
        line = INLINE_CITATION_RE.sub("", line)
        line = AUTHOR_YEAR_RE.sub("", line)

        # 页码
        line = PAGE_NUM_TRAILING_RE.sub("", line)
        line = PAGE_NUM_TRAILING2_RE.sub("", line)

        # 图/表编号
        line = TABLE_RE.sub("", line)
        line = FIGURE_RE.sub("", line)
        line = TABLE_EN_RE.sub("", line)
        line = FIGURE_EN_RE.sub("", line)

        # 交叉引用
        line = CROSS_REF_CN.sub("", line)
        line = CROSS_REF_EN.sub("", line)

        # 乱码
        line = GARBLE_RE.sub("", line).replace("", "")

        # 参考文献标记
        line = REF_PAREN_RE.sub("", line)

        line = line.strip()
        if line:
            cleaned.append(line)

    # 修复断行
    if fix_broken_lines:
        cleaned = _fix_broken_lines(cleaned)

    return cleaned


def _fix_broken_lines(lines: list[str]) -> list[str]:
    """
    修复断行：当一行末尾没有句号等结束标点时，
    将下一行拼接到当前行末尾（加一个空格）。

    参数:
        lines: 清洗后的行列表

    返回:
        修复断行后的行列表
    """
    if not lines:
        return lines

    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 空行直接保留
        if not line:
            result.append(line)
            i += 1
            continue

        # 行末有结束标点或冒号，不拼接
        if line[-1] in ("。", ".", "！", "？", "；", ";", "：", ":", "\n", "—", "…"):
            result.append(line)
            i += 1
            continue

        # 尝试拼接下一行
        if i + 1 < len(lines) and lines[i + 1]:
            next_line = lines[i + 1]
            # 如果下一行以 # 开头（标题），不拼接
            if next_line.startswith("#"):
                result.append(line)
                i += 1
                continue
            # 拼接
            line = line + " " + next_line
            i += 2
            # 继续尝试拼接（可能有多行断行）
            result.append(line)
        else:
            result.append(line)
            i += 1

    return result


def clean_text_file(
    input_file: str,
    output_file: str,
    watermark_keywords: list[str] | None = None,
    fix_broken_lines: bool = True,
) -> str | None:
    """
    读取纯文本文件，清洗后写入输出文件。
    同时输出 _sections.json（结构化数据，供后续智能切分使用）。

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        watermark_keywords: 水印关键词
        fix_broken_lines: 是否修复断行

    返回:
        _sections.json 文件路径（如有），否则 None
    """
    import os
    from pathlib import Path

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.readlines()

        cleaned = clean_text_lines(content, watermark_keywords, fix_broken_lines)

        # 输出 _cleaned.txt（向后兼容）
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned))

        logger.info(f"文本清洗完成: {input_file} → {output_file} ({len(cleaned)} 行)")

        # 输出 _sections.json（结构化数据）
        source_name = os.path.basename(input_file)
        blocks = _extract_sections_from_text(cleaned, source_name)
        if blocks:
            # Path.with_suffix 要求参数以 "." 开头，这里用 stem 拼接 _sections.json
            sections_file = str(Path(output_file).with_name(Path(output_file).stem + "_sections.json"))
            sections_data = [
                {
                    "title": b["title"],
                    "full_path": b["full_path"],
                    "level": b["level"],
                    "body": b["body"],
                    "source_file": b["source_file"],
                    "char_count": len(b["body"]),
                }
                for b in blocks
            ]
            with open(sections_file, "w", encoding="utf-8") as f:
                json.dump(sections_data, f, ensure_ascii=False, indent=2)
            return sections_file
        return None
    except Exception as e:
        logger.error(f"文本清洗失败 {input_file}: {e}")
        return None


def _extract_sections_from_text(lines: list[str], source_file: str) -> list[dict]:
    """
    从纯文本行列表中提取章节结构。

    标题检测模式：
    - ^第.*章 （如 "第一章 概述"）
    - ^\d+[.、] （如 "1.1 引言"、"2、实验方法"）
    - ^# （Markdown 风格标题）

    无标题时，整个文本作为一个区块，使用文件名作为标题。

    参数:
        lines: 清洗后的文本行
        source_file: 源文件名

    返回:
        章节区块列表
    """
    title_patterns = [
        (re.compile(r"^第[一二三四五六七八九十百千万\d]+[章节篇部]"), 1),
        (re.compile(r"^\d+[.、]\s*\S+"), 2),
        (re.compile(r"^#\s*(.+)"), 1),
    ]

    blocks = []
    current_title = None
    current_level = 0
    current_body = []

    def _flush():
        nonlocal current_title, current_level, current_body
        if current_title and current_body:
            body_text = "\n".join(current_body).strip()
            if len(body_text) > 30:  # 最小正文长度
                blocks.append({
                    "title": current_title,
                    "full_path": current_title,
                    "level": current_level,
                    "body": body_text,
                    "source_file": source_file,
                })
        current_body = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = False
        for pattern, level in title_patterns:
            m = pattern.match(stripped)
            if m:
                _flush()
                current_title = m.group(1).strip() if m.lastindex else stripped.lstrip("#").strip()
                current_level = level
                matched = True
                break

        if not matched:
            if not current_title:
                current_title = os.path.splitext(source_file)[0]
                current_level = 1
            current_body.append(stripped)

    # 收尾
    _flush()

    # 如果什么都没匹配到，整个文本作为一个区块
    if not blocks and lines:
        body_text = "\n".join(l.strip() for l in lines if l.strip())
        if body_text:
            blocks.append({
                "title": os.path.splitext(source_file)[0],
                "full_path": os.path.splitext(source_file)[0],
                "level": 1,
                "body": body_text,
                "source_file": source_file,
            })

    return blocks
