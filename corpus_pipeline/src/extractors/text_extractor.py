"""
TXT / Markdown 文本提取模块

直接从 .txt 和 .md 文件中读取文本，无需经过 magic-pdf 或 OCR。
自动检测文件编码（UTF-8 / GBK / GB2312），确保中文文件正确读取。
"""

import os
import chardet

from src.extractors.base import BaseExtractor
from src.logger import get_logger

logger = get_logger()


class TextExtractor(BaseExtractor):
    """直接从 .txt 和 .md 文件中读取文本内容。"""

    def __init__(self, config: dict):
        """
        初始化文本提取器。

        参数:
            config: 文本提取配置（当前无特殊参数）
        """
        super().__init__(config)

    def extract(self, file_path: str, output_dir: str) -> str | None:
        """
        提取单个 .txt 或 .md 文件的文本内容。

        流程:
        1. 自动检测文件编码
        2. 读取文本内容
        3. 写入 output_dir 下的 .md 文件（供后续清洗使用）

        参数:
            file_path: 输入文件绝对路径
            output_dir: 输出目录

        返回:
            输出 .md 文件路径，失败返回 None
        """
        ext = os.path.splitext(file_path)[1].lower()
        name_without_suff = os.path.splitext(os.path.basename(file_path))[0]
        md_output = os.path.join(output_dir, f"{name_without_suff}.md")

        try:
            # 自动检测编码
            encoding = self._detect_encoding(file_path)
            if not encoding:
                encoding = "utf-8"

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"文件为空，跳过: {file_path}")
                return None

            os.makedirs(output_dir, exist_ok=True)
            with open(md_output, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"文本提取成功 ({encoding}): {file_path} → {md_output}")
            return md_output

        except Exception as e:
            logger.error(f"文本提取失败 {file_path}: {e}")
            return None

    def supports_extension(self, ext: str) -> bool:
        """判断是否支持 .txt 和 .md 文件。"""
        return ext.lower() in {".txt", ".md"}

    def _detect_encoding(self, file_path: str) -> str | None:
        """
        使用 chardet 自动检测文件编码。

        参数:
            file_path: 文件路径

        返回:
            检测到的编码名称，如 "utf-8"、"gbk"、"gb2312"，失败返回 None
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(8192)  # 读取前 8KB 用于检测
            result = chardet.detect(raw_data)
            encoding = result.get("encoding")
            confidence = result.get("confidence", 0)

            if encoding and confidence > 0.5:
                # 规范化常见中文编码名称
                encoding_lower = encoding.lower()
                if encoding_lower in {"gb2312", "gbk", "gb18030", "cp936", "hz", "euc-kr"}:
                    return "gbk"  # 统一使用 gbk 读取
                return encoding
            return None
        except Exception as e:
            logger.warning(f"编码检测失败 {file_path}: {e}")
            return None
