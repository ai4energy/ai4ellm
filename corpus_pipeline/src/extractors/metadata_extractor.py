"""
元数据提取模块

从 PDF/DOCX/PPTX/TXT/MD 源文件中提取文档级元数据（标题、作者、创建日期等），
为后续的来源追踪和语料质量评估提供基础信息。
"""

import os
import json
from datetime import datetime
from pathlib import Path

from src.logger import get_logger

logger = get_logger()


def extract_file_metadata(file_path: str) -> dict:
    """
    从源文件提取元数据。

    根据文件扩展名自动选择提取器：
    - PDF: PyPDF2 读取文档信息字典
    - DOCX: python-docx core_properties
    - PPTX: python-pptx core_properties
    - TXT/MD: 文件基本信息

    参数:
        file_path: 源文件绝对路径

    返回:
        元数据字典，包含:
            source_file: 原始文件名
            source_path: 原始文件完整路径
            file_size: 文件大小（字节）
            file_type: 文件类型标识 (pdf/docx/pptx/txt/md)
            title: 文档标题（如有）
            author: 作者（如有）
            creation_date: 创建日期（如有）
            modification_date: 修改日期（如有）
            page_count: 页数（PDF 适用）
            language: 语言推测（None = 未检测）
    """
    ext = os.path.splitext(file_path)[1].lower()
    file_path = os.path.abspath(file_path)

    meta = {
        "source_file": os.path.basename(file_path),
        "source_path": file_path,
        "file_size": os.path.getsize(file_path),
        "file_type": _classify_type(ext),
        "title": None,
        "author": None,
        "creation_date": None,
        "modification_date": None,
        "page_count": None,
        "language": None,
    }

    try:
        if ext == ".pdf":
            meta = _extract_pdf_meta(file_path, meta)
        elif ext in {".docx"}:
            meta = _extract_docx_meta(file_path, meta)
        elif ext in {".pptx"}:
            meta = _extract_pptx_meta(file_path, meta)
        elif ext == ".doc":
            # .doc 旧格式，元数据需 COM 转换后获取
            meta["title"] = os.path.splitext(meta["source_file"])[0]
        elif ext == ".ppt":
            meta["title"] = os.path.splitext(meta["source_file"])[0]
        elif ext in {".txt", ".md"}:
            meta = _extract_text_meta(file_path, meta)
    except Exception as e:
        logger.warning(f"提取元数据失败 {file_path}: {e}")

    # 序列化日期等不可 JSON 化的值
    return _sanitize_for_json(meta)


def save_metadata(meta: dict, output_dir: str) -> str:
    """
    将元数据保存到 JSON 文件。

    参数:
        meta: extract_file_metadata() 返回的元数据
        output_dir: 输出目录（通常为 output/metadata/）

    返回:
        保存的 JSON 文件路径，失败返回 None
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(meta["source_file"])[0]
        output_path = os.path.join(output_dir, f"{stem}_meta.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"元数据已保存: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"保存元数据失败: {e}")
        return None


def load_metadata(meta_path: str) -> dict | None:
    """
    从 JSON 文件加载元数据。

    参数:
        meta_path: 元数据 JSON 文件路径

    返回:
        元数据字典，失败返回 None
    """
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载元数据失败 {meta_path}: {e}")
        return None


# ---- 各格式提取器 ----

def _extract_pdf_meta(file_path: str, meta: dict) -> dict:
    """从 PDF 提取元数据。"""
    try:
        from PyPDF2 import PdfReader
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            info = reader.metadata
            if info:
                meta["title"] = info.get("/Title")
                meta["author"] = info.get("/Author")
                meta["creation_date"] = _parse_pdf_date(info.get("/CreationDate"))
                meta["modification_date"] = _parse_pdf_date(info.get("/ModDate"))
                meta["producer"] = info.get("/Producer")
            meta["page_count"] = len(reader.pages)
    except Exception as e:
        logger.warning(f"PDF 元数据提取失败 {file_path}: {e}")
    return meta


def _extract_docx_meta(file_path: str, meta: dict) -> dict:
    """从 DOCX 提取元数据。"""
    try:
        from docx import Document
        doc = Document(file_path)
        props = doc.core_properties
        meta["title"] = props.title or None
        meta["author"] = props.author or None
        meta["creation_date"] = _format_datetime(props.created)
        meta["modification_date"] = _format_datetime(props.modified)
    except Exception as e:
        logger.warning(f"DOCX 元数据提取失败 {file_path}: {e}")
    return meta


def _extract_pptx_meta(file_path: str, meta: dict) -> dict:
    """从 PPTX 提取元数据。"""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        props = prs.core_properties
        meta["title"] = props.title or None
        meta["author"] = props.author or None
        meta["creation_date"] = _format_datetime(props.created)
        meta["modification_date"] = _format_datetime(props.modified)
    except Exception as e:
        logger.warning(f"PPTX 元数据提取失败 {file_path}: {e}")
    return meta


def _extract_text_meta(file_path: str, meta: dict) -> dict:
    """从 TXT/MD 提取基本信息。"""
    meta["title"] = os.path.splitext(meta["source_file"])[0]
    # 尝试检测编码
    try:
        import chardet
        with open(file_path, "rb") as f:
            raw = f.read(8192)
        result = chardet.detect(raw)
        meta["language"] = result.get("encoding")
    except Exception:
        pass
    # 尝试统计行数/字符数
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            meta["char_count"] = len(content)
            meta["line_count"] = content.count("\n") + 1
    except Exception:
        pass
    return meta


# ---- 辅助函数 ----

def _classify_type(ext: str) -> str:
    """根据扩展名分类文件类型。"""
    mapping = {
        ".pdf": "pdf",
        ".doc": "doc",
        ".docx": "docx",
        ".ppt": "ppt",
        ".pptx": "pptx",
        ".txt": "txt",
        ".md": "md",
    }
    return mapping.get(ext.lower(), "unknown")


def _parse_pdf_date(date_str: str | None) -> str | None:
    """
    解析 PDF 日期字符串（格式: D:20210315120000+08'00'）。

    参数:
        date_str: PDF 日期字符串

    返回:
        ISO 格式日期字符串，解析失败返回原始字符串
    """
    if not date_str:
        return None
    try:
        # PDF 日期格式: D:YYYYMMDDHHmmSSOHH'mm'
        s = date_str.replace("D:", "").replace("'", "")
        # 截取到至少 YYYYMMDD
        if len(s) >= 8:
            dt = datetime(
                year=int(s[0:4]),
                month=int(s[4:6]),
                day=int(s[6:8]),
                hour=int(s[8:10]) if len(s) >= 10 else 0,
                minute=int(s[10:12]) if len(s) >= 12 else 0,
                second=int(s[12:14]) if len(s) >= 14 else 0,
            )
            return dt.isoformat()
    except (ValueError, IndexError):
        pass
    return date_str


def _format_datetime(dt) -> str | None:
    """将 datetime 对象格式化为 ISO 字符串。"""
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _sanitize_for_json(obj: dict) -> dict:
    """确保字典中所有值都可以被 JSON 序列化。"""
    sanitized = {}
    for k, v in obj.items():
        if isinstance(v, datetime):
            sanitized[k] = v.isoformat()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            sanitized[k] = v
        else:
            sanitized[k] = str(v)
    return sanitized
