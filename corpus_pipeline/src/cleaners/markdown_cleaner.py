"""
Markdown 清洗模块

从 magic-pdf 生成的 Markdown 中提取标题和正文，去除目录、页码、
水印、乱码、参考文献标记等干扰内容。
"""

import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.cleaners.rules import (
    INLINE_CITATION_RE, AUTHOR_YEAR_RE, GARBLE_RE,
    PAGE_NUM_TRAILING_RE, PAGE_NUM_TRAILING2_RE,
    TABLE_RE, FIGURE_RE, TABLE_EN_RE, FIGURE_EN_RE,
    CROSS_REF_CN, CROSS_REF_EN, TOC_CHAPTER_RE, TOC_SECTION_RE,
    REF_PAREN_RE, URL_LINE_RE, WATERMARK_XJTU, CN_WORD_RE,
    DEFAULT_SKIP_SECTIONS, DEFAULT_WATERMARK_KEYWORDS,
)
from src.logger import get_logger

logger = get_logger()


@dataclass
class SectionBlock:
    """Markdown 文档中的一个章节区块，保留标题层级和来源信息。"""
    title: str                          # 当前节标题（如 "1.1 概述"）
    full_path: str                      # 完整层级路径（如 "第一章 > 1.1 概述"）
    level: int                          # 标题层级 1-6
    body: str                           # 正文内容
    source_file: str                    # 原始文件名
    char_count: int = 0                 # 正文字符数

    def __post_init__(self):
        if self.char_count == 0:
            self.char_count = len(self.body)


def clean_markdown(content: list[str], watermark_keywords: Optional[list[str]] = None) -> list[str]:
    """
    提取 Markdown 中标题及正文并做初步清洗。

    1. 跳过目录片段 / 图片 / 链接 / 水印行
    2. 删除连续空行

    参数:
        content: 原始 Markdown 内容行列表
        watermark_keywords: 水印关键词列表

    返回:
        初步清洗后的行列表
    """
    if watermark_keywords is None:
        watermark_keywords = DEFAULT_WATERMARK_KEYWORDS

    cleaned = []
    is_in_toc = False
    prev_line = ""

    for raw in content:
        line = raw.rstrip()

        # 目录段落识别
        if line.lower().startswith("contents") or "目录" in line:
            is_in_toc = True
            continue
        if is_in_toc and (not line.strip() or line.startswith("#")):
            is_in_toc = False
        if is_in_toc:
            continue

        # 图片/链接行
        if line.lstrip().startswith("![") or "](" in line:
            continue

        # 明显水印
        if any(key in line.lower() for key in watermark_keywords):
            continue

        # 行内引用 [xxx]
        line = INLINE_CITATION_RE.sub("", line)
        # 作者年份引用 Zhang (2021)
        line = AUTHOR_YEAR_RE.sub("", line)

        # 连续空行压缩
        if not line.strip():
            if not prev_line.strip():
                continue
            cleaned.append("")
        else:
            cleaned.append(line)
        prev_line = line

    return cleaned


def remove_specific_patterns(lines: list[str]) -> list[str]:
    """
    删除目录行、页码、图表编号、URL、水印、行内交叉引用等。

    参数:
        lines: 清洗后的行列表

    返回:
        剔除特定模式后的行列表
    """
    cleaned = []
    for line in lines:
        stripped = line.strip()

        # CIP 或版编目行
        if "图书在版编目" in stripped or "CIP" in stripped:
            continue

        # 章标题 + 页码
        if TOC_CHAPTER_RE.match(stripped):
            continue

        # 节目录行
        if TOC_SECTION_RE.match(stripped):
            continue

        # 一行出现 ≥2 个编号，判目录汇总
        if len(re.findall(r"\d+\.\d+(?:\.\d+)*", stripped)) >= 2 and re.search(r"[\/\s]\d{1,4}", stripped):
            continue

        # 去掉行尾页码
        line = PAGE_NUM_TRAILING_RE.sub("", line)
        line = PAGE_NUM_TRAILING2_RE.sub("", line)

        # 图/表编号占位
        line = TABLE_RE.sub("", line)
        line = FIGURE_RE.sub("", line)
        line = TABLE_EN_RE.sub("", line)
        line = FIGURE_EN_RE.sub("", line)

        # URL 整行
        if URL_LINE_RE.match(stripped):
            continue

        # 西安交通大学水印
        line = WATERMARK_XJTU.sub("", line)

        # 行内交叉引用
        line = CROSS_REF_CN.sub("", line)
        line = CROSS_REF_EN.sub("", line)

        if line.strip():
            cleaned.append(line.strip())
    return cleaned


def remove_garbled_characters(lines: list[str]) -> list[str]:
    """
    移除乱码字符和替换符号。

    参数:
        lines: 行列表

    返回:
        去除乱码后的行列表
    """
    cleaned = []
    for line in lines:
        line = GARBLE_RE.sub("", line).replace("�", "").strip()
        if line:
            cleaned.append(line)
    return cleaned


def clean_references(lines: list[str]) -> list[str]:
    """
    清理参考文献标记（如 (Author et al., 2024)）。

    参数:
        lines: 行列表

    返回:
        清理参考文献后的行列表
    """
    cleaned = []
    for line in lines:
        line = REF_PAREN_RE.sub("", line)
        if line.strip():
            cleaned.append(line.strip())
    return cleaned


def clean_and_extract_markdown(
    content: list[str],
    source_file: str = "",
    skip_sections: Optional[set[str]] = None,
    min_body_chars: int = 50,
    min_final_body_chars: int = 150,
    watermark_keywords: Optional[list[str]] = None,
) -> list["SectionBlock"]:
    """
    完整清洗管线：目录去除、模式剔除、乱码清理、章节抽取、正文拼接。

    参数:
        content: 原始 Markdown 内容行列表
        source_file: 原始文件名（用于来源追踪）
        skip_sections: 需要跳过的章节名称集合
        min_body_chars: 正文最小字符数（中间章节）
        min_final_body_chars: 正文最小字符数（末尾章节）
        watermark_keywords: 水印关键词列表

    返回:
        SectionBlock 列表，每项包含标题、完整路径、层级和正文
    """
    if skip_sections is None:
        skip_sections = DEFAULT_SKIP_SECTIONS

    # 依次应用各清洗步骤
    lines = clean_markdown(content, watermark_keywords)
    lines = remove_specific_patterns(lines)
    lines = clean_references(lines)
    lines = remove_garbled_characters(lines)

    # 章节抽取与正文拼接
    blocks: list[SectionBlock] = []
    title_stack: list[tuple[str, int]] = []  # (title, level)
    para = []
    skip = False

    def _flush_block():
        """将当前累积的段落输出为一个 SectionBlock。"""
        nonlocal para
        if not title_stack or not para:
            para = []
            return
        current_title, current_level = title_stack[-1]
        full_path = " > ".join(t for t, _ in title_stack)
        body = "".join(para).strip()
        threshold = min_final_body_chars if len(title_stack) == 1 else min_body_chars
        if len(CN_WORD_RE.findall(body)) > threshold:
            blocks.append(SectionBlock(
                title=current_title,
                full_path=full_path,
                level=current_level,
                body=body,
                source_file=source_file,
            ))
        para = []

    def _is_parent_skipped() -> bool:
        """检查当前标题的父级是否在跳过列表中。"""
        if len(title_stack) <= 1:
            return False
        parent_title = title_stack[-2][0].lower()
        return any(k in parent_title for k in skip_sections)

    for line in lines:
        if line.startswith("#"):
            # 结束上一节
            _flush_block()
            # 解析新标题层级
            level = min(len(line) - len(line.lstrip("#")), 6)
            title_text = line.strip("#").strip()
            # 更新标题栈：弹出同级或更深层级，压入新标题
            while title_stack and title_stack[-1][1] >= level:
                title_stack.pop()
            title_stack.append((title_text, level))
            # 判断是否需要跳过：标题本身匹配关键词，或父级在跳过列表中
            skip = any(k in title_text.lower() for k in skip_sections) or _is_parent_skipped()
            continue

        if not skip and line:
            # 智能拼接：句末无标点加空格
            if para and not para[-1].endswith(("。", ".", "！", "？", "；", ";", "：", ":")):
                para[-1] += " "
            para.append(line)

    # 收尾最后一段
    _flush_block()

    return blocks


def process_markdown_files(
    input_dir: str,
    output_dir: str,
    skip_sections: Optional[set[str]] = None,
    min_body_chars: int = 50,
    min_final_body_chars: int = 150,
    remove_garbled: bool = True,
    clean_refs: bool = True,
    watermark_keywords: Optional[list[str]] = None,
) -> list[str]:
    """
    批量处理目录中的所有 .md 文件，返回成功处理的输出文件路径列表。

    对每个文件：
    1. 输出 _sections.json（结构化 SectionBlock 列表）
    2. 输出 _cleaned.txt（向后兼容的纯文本）

    参数:
        input_dir: 输入 .md 文件目录
        output_dir: 输出清洗后 .txt 文件目录
        skip_sections: 跳过章节集合
        min_body_chars: 中间章节最小字符数
        min_final_body_chars: 末尾章节最小字符数
        remove_garbled: 是否去除乱码
        clean_refs: 是否清理参考文献
        watermark_keywords: 水印关键词

    返回:
        成功处理的输出文件路径列表（_cleaned.txt）
    """
    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if skip_sections is None:
        skip_sections = DEFAULT_SKIP_SECTIONS

    output_files = []
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".txt"}]

    for src in files:
        try:
            with src.open("r", encoding="utf-8") as f:
                content = f.readlines()

            blocks = clean_and_extract_markdown(
                content,
                source_file=src.name,
                skip_sections=skip_sections,
                min_body_chars=min_body_chars,
                min_final_body_chars=min_final_body_chars,
                watermark_keywords=watermark_keywords,
            )

            if blocks:
                # 输出 _sections.json（结构化数据）
                sections_json = [
                    {
                        "title": b.title,
                        "full_path": b.full_path,
                        "level": b.level,
                        "body": b.body,
                        "source_file": b.source_file,
                        "char_count": b.char_count,
                    }
                    for b in blocks
                ]
                sections_file = dst_dir / f"{src.stem}_sections.json"
                with sections_file.open("w", encoding="utf-8") as f:
                    json.dump(sections_json, f, ensure_ascii=False, indent=2)

                # 输出 _cleaned.txt（向后兼容）
                dst = dst_dir / f"{src.stem}_cleaned.txt"
                text_lines = []
                for b in blocks:
                    text_lines.append(f"# {b.title}")
                    text_lines.append(b.body)
                    text_lines.append("")
                dst.write_text("\n".join(text_lines), encoding="utf-8")
                output_files.append(str(dst))
        except Exception as e:
            logger.error(f"清洗文件 {src} 时出错: {e}")

    logger.info(f"Markdown 清洗完成，共处理 {len(output_files)}/{len(files)} 个文件")
    return output_files
