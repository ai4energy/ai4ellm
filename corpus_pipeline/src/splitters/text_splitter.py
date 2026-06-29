"""
文本分割模块

按段落或语义块对文本进行切分，支持最大字符数限制。
支持从结构化 _sections.json 读取，保留标题层级和来源信息。
"""

import os
import re
import json
from pathlib import Path

from src.logger import get_logger

logger = get_logger()


def split_by_paragraph(text: str, max_chars: int = 0) -> list[str]:
    """
    按段落分割文本。

    参数:
        text: 输入文本
        max_chars: 每段最大字符数（0 = 不进一步分割）

    返回:
        段落列表
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if max_chars <= 0:
        return paragraphs

    # 对超长段落进一步拆分
    result = []
    for p in paragraphs:
        if len(p) <= max_chars:
            result.append(p)
        else:
            # 按句子边界拆分
            sentences = _split_by_sentences(p)
            current = ""
            for s in sentences:
                if current and len(current) + len(s) > max_chars:
                    result.append(current.strip())
                    current = s
                else:
                    current += s
            if current.strip():
                result.append(current.strip())

    return result


def _split_by_sentences(text: str) -> list[str]:
    """
    按句子边界拆分文本（支持中英文标点）。

    参数:
        text: 输入文本

    返回:
        句子列表
    """
    # 中英文句子结束标点 + 换行符
    pattern = r'(?<=[。！？.!?;；])\s*'
    sentences = re.split(pattern, text)
    return [s for s in sentences if s.strip()]


def _split_long_sentence(sentence: str, max_chars: int = 500) -> list[str]:
    """
    将超长句子按逗号/分号进一步拆分。

    参数:
        sentence: 输入句子
        max_chars: 最大字符数

    返回:
        拆分后的片段列表
    """
    if len(sentence) <= max_chars:
        return [sentence]

    # 按中文逗号、分号、英文逗号、分号拆分
    parts = re.split(r'[，,；;、]', sentence)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [sentence[:max_chars]]

    # 贪心拼接
    result = []
    current = ""
    for p in parts:
        if current and len(current) + len(p) + 1 > max_chars:
            result.append(current.strip())
            current = p
        else:
            if current:
                current += "，" + p
            else:
                current = p
    if current.strip():
        result.append(current.strip())

    return result


def split_sections(
    cleaned_dir: str,
    output_dir: str,
    max_segment_chars: int = 500,
    min_segment_chars: int = 50,
) -> list[dict]:
    """
    从 cleaned_text/ 目录中的 _sections.json 文件智能切分文本块。

    切分策略：
    1. 以 SectionBlock 为单位遍历
    2. 每个 block 的 body 按句子级切分
    3. 贪心拼接句子到 max_segment_chars 以内（保持句子完整）
    4. 单句超长时按逗号/分号进一步拆分
    5. 每个 chunk 携带 full_path、source_file、source_type、chunk_index

    参数:
        cleaned_dir: cleaned_text/ 目录路径
        output_dir: 输出目录（chunks/）
        max_segment_chars: 每段最大字符数
        min_segment_chars: 每段最小字符数

    返回:
        chunk 列表，每个元素为结构化字典
    """
    os.makedirs(output_dir, exist_ok=True)

    sections_files = list(Path(cleaned_dir).glob("*_sections.json"))
    all_chunks = []
    global_index = 0

    for sf in sorted(sections_files):
        try:
            with open(sf, "r", encoding="utf-8") as f:
                sections = json.load(f)
        except Exception as e:
            logger.warning(f"读取 sections 文件失败 {sf}: {e}")
            continue

        source_file = sf.stem.replace("_sections", "")
        # 推断源文件类型
        source_type = _infer_source_type(source_file, cleaned_dir)

        for section in sections:
            body = section.get("body", "")
            title = section.get("title", "")
            full_path = section.get("full_path", title)
            level = section.get("level", 1)

            # 句子级切分
            sentences = _split_by_sentences(body)

            # 贪心拼接
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) > max_segment_chars:
                    # 输出当前 chunk
                    chunk = _make_chunk(
                        content=current.strip(),
                        section_title=title,
                        full_path=full_path,
                        source_file=section.get("source_file", source_file),
                        source_type=source_type,
                        chunk_index=global_index,
                        level=level,
                    )
                    if chunk["char_count"] >= min_segment_chars:
                        all_chunks.append(chunk)
                        global_index += 1
                    current = sent
                else:
                    current += sent

            # 收尾
            if current.strip():
                chunk = _make_chunk(
                    content=current.strip(),
                    section_title=title,
                    full_path=full_path,
                    source_file=section.get("source_file", source_file),
                    source_type=source_type,
                    chunk_index=global_index,
                    level=level,
                )
                if chunk["char_count"] >= min_segment_chars:
                    all_chunks.append(chunk)
                    global_index += 1

    # 输出 _chunks.json（结构化数据）
    chunks_json_path = os.path.join(output_dir, "_chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 输出 .txt 文件（向后兼容）
    for chunk in all_chunks:
        txt_path = os.path.join(output_dir, f"chunk_s{chunk['chunk_index']}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# {chunk['section_title']}\n{chunk['content']}\n")

    logger.info(f"文本分割完成: {len(sections_files)} 个 sections 文件 → {len(all_chunks)} 个 chunk")
    return all_chunks


def split_text_file(
    input_file: str,
    output_dir: str,
    strategy: str = "paragraph",
    max_segment_chars: int = 0,
) -> list[str]:
    """
    读取清洗后的 txt 文件，按策略分割段落，每段写入独立文件。
    （旧版兼容接口，从 all_merged.txt 读取）

    参数:
        input_file: 输入 txt 文件路径
        output_dir: 输出目录
        strategy: 分割策略（"paragraph"）
        max_segment_chars: 每段最大字符数

    返回:
        所有输出文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 # 标题分割成章节
    sections = content.split("\n#")
    base_name = Path(input_file).stem

    for i, section in enumerate(sections):
        section = section.strip()
        if not section.startswith("#"):
            section = "#" + section

        # 分离标题和正文
        lines = section.split("\n", 1)
        title = lines[0].lstrip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""

        if not body:
            continue

        paragraphs = split_by_paragraph(body, max_chars=max_segment_chars)
        for j, para in enumerate(paragraphs):
            if len(para) < 10:
                continue
            out_file = os.path.join(output_dir, f"{base_name}_s{i}_p{j}.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n{para}\n")
            output_files.append(out_file)

    logger.info(f"文本分割完成: {input_file} → {len(output_files)} 段")
    return output_files


# ---- 辅助函数 ----

def _make_chunk(
    content: str,
    section_title: str,
    full_path: str,
    source_file: str,
    source_type: str,
    chunk_index: int,
    level: int = 1,
) -> dict:
    """构建结构化 chunk 字典。"""
    return {
        "content": content,
        "section_title": section_title,
        "full_path": full_path,
        "source_file": source_file,
        "source_type": source_type,
        "chunk_index": chunk_index,
        "level": level,
        "char_count": len(content),
    }


def _infer_source_type(stem: str, cleaned_dir: str) -> str:
    """
    从 cleaned_text/ 中的文件名推断源文件类型。

    查找对应的元数据文件或原始提取文件。
    stem 可能带 _cleaned 后缀（清洗器产出），去掉后再查 metadata（按原始输入名存）。
    """
    base_dir = os.path.dirname(cleaned_dir)
    # 去掉清洗器加的 _cleaned 后缀，得到原始输入文件名 stem
    raw_stem = stem[:-len("_cleaned")] if stem.endswith("_cleaned") else stem

    # 尝试找元数据文件（按原始输入名存：{raw_stem}_meta.json）
    meta_path = os.path.join(base_dir, "metadata", f"{raw_stem}_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ft = meta.get("file_type")
            if ft:
                return ft
        except Exception:
            pass

    # 从 extracted_text/ 中找对应的提取文件
    extracted_dir = os.path.join(base_dir, "extracted_text")
    for ext, ftype in [(".pdf", "pdf"), (".docx", "docx"), (".doc", "doc"),
                        (".pptx", "pptx"), (".ppt", "ppt"), (".txt", "txt"), (".md", "md")]:
        if os.path.exists(os.path.join(extracted_dir, raw_stem + ext)):
            return ftype

    return "unknown"
