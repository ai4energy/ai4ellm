"""
PDF 提取器 — 多引擎 + 智能回退（Linux 原生）

引擎优先级:
1. PyMuPDF  - 快速文本提取（首选，文本型 PDF）
2. MinerU   - 高质量 OCR（扫描件/复杂排版，需 magic-pdf）
3. pdfplumber - 表格提取友好（备选）

本模块适配 BaseExtractor 接口，替代原 zip 版强绑 magic-pdf do_parse API 的实现。
多引擎回退逻辑来自原 ai4ellm/pretrain/text-extraction/pdf_extractor.py，已在 907 个
能源领域 PDF 上验证跑通。
"""

import os
import logging
from pathlib import Path

from src.extractors.base import BaseExtractor
from src.logger import get_logger

logger = get_logger()


class _PyMuPDFExtractor:
    """PyMuPDF 提取器 — 快速文本提取。"""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import fitz  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50):
        """返回 (text, pages, success, error)。"""
        if not self.is_available():
            return "", 0, False, "PyMuPDF未安装"
        try:
            import fitz
            fitz.TOOLS.mupdf_warnings(0)
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())
            pages = doc.page_count
            doc.close()
            full_text = "\n\n".join(text_parts)
            if len(full_text.strip()) < min_text_length:
                return "", pages, False, "文本过少（可能是扫描件）"
            return full_text, pages, True, ""
        except Exception as e:
            return "", 0, False, str(e)


class _MinerUExtractor:
    """MinerU 提取器 — 高质量 PDF 提取，支持 OCR（需 magic-pdf）。"""

    def __init__(self, use_gpu: bool = False, formula_enable: bool = False, table_enable: bool = False):
        self.use_gpu = use_gpu
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from magic_pdf.data.dataset import PymuDocDataset  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50):
        """返回 (text, pages, success, error)。"""
        if not self.is_available():
            return "", 0, False, "MinerU未安装"
        try:
            import tempfile
            from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod

            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)
            ds = PymuDocDataset(pdf_bytes)
            parse_method = ds.classify()

            with tempfile.TemporaryDirectory() as tmpdir:
                image_writer = FileBasedDataWriter(tmpdir)
                md_writer = FileBasedDataWriter(tmpdir)

                if parse_method == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                name = Path(pdf_path).stem
                pipe_result.dump_md(md_writer, f"{name}.md", "images")
                md_path = os.path.join(tmpdir, f"{name}.md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    text = ""

                if len(text.strip()) < min_text_length:
                    return "", ds.page_count, False, "OCR提取文本过少"
                return text, ds.page_count, True, ""
        except Exception as e:
            return "", 0, False, str(e)


class _PDFPlumberExtractor:
    """pdfplumber 提取器 — 表格提取友好（备选）。"""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pdfplumber  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50):
        if not self.is_available():
            return "", 0, False, "pdfplumber未安装"
        try:
            import pdfplumber
            text_parts = []
            pages = 0
            with pdfplumber.open(pdf_path) as pdf:
                pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            full_text = "\n\n".join(text_parts)
            if len(full_text.strip()) < min_text_length:
                return "", pages, False, "文本过少"
            return full_text, pages, True, ""
        except Exception as e:
            return "", 0, False, str(e)


class PDFExtractor(BaseExtractor):
    """
    PDF 提取器 — 多引擎智能回退。

    config 键:
        method: "pymupdf" | "mineru" | "pdfplumber" | "auto" (默认 auto)
        min_text_length: int (默认 50)
        num_gpus: int (0=CPU；>0 时 MinerU 可用 GPU)
        formula_enable: bool (MinerU 公式识别)
        table_enable: bool (MinerU 表格识别)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_text_length = config.get("min_text_length", 50)
        method = config.get("method", "auto")
        use_mineru_gpu = config.get("num_gpus", 0) > 0

        self._engines = []
        # 按优先级组装引擎列表
        if method in ("pymupdf", "auto"):
            self._engines.append(("pymupdf", _PyMuPDFExtractor()))
        if method in ("mineru", "auto"):
            self._engines.append(("mineru", _MinerUExtractor(
                use_gpu=use_mineru_gpu,
                formula_enable=config.get("formula_enable", False),
                table_enable=config.get("table_enable", False),
            )))
        if method in ("pdfplumber", "auto"):
            self._engines.append(("pdfplumber", _PDFPlumberExtractor()))

        self.last_error = ""
        available = [name for name, ext in self._engines if ext.is_available()]
        logger.info(f"PDF 提取可用引擎: {available} (method={method})")
        if not available:
            logger.warning("没有可用的 PDF 提取引擎！请至少安装 PyMuPDF (pip install PyMuPDF)")

    def supports_extension(self, ext: str) -> bool:
        return ext.lower() == ".pdf"

    def extract(self, file_path: str, output_dir: str) -> str | None:
        """
        提取单个 PDF，多引擎回退。

        输出扩展名按引擎区分：
        - MinerU 产出真 Markdown（带 # 标题结构）→ {output_dir}/{name}.md
        - PyMuPDF/pdfplumber 产出纯文本 → {output_dir}/{name}.txt
          （pipeline 据扩展名分发：.md 走 markdown_cleaner，.txt 走 text_cleaner）

        返回输出文件绝对路径，全部失败返回 None（self.last_error 记录原因）。
        """
        name = Path(file_path).stem

        for engine_name, engine in self._engines:
            if not engine.is_available():
                continue
            text, pages, success, error = engine.extract(file_path, self.min_text_length)
            if success:
                ext = ".md" if engine_name == "mineru" else ".txt"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{name}{ext}")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(text)
                self.last_error = ""
                logger.info(f"PDF 提取成功 [{engine_name}] {file_path} ({pages}页, {len(text)}字符)")
                return output_file
            self.last_error = f"{engine_name}: {error}"

        logger.warning(f"PDF 提取失败 {file_path}: {self.last_error}")
        return None


def process_folder_parallel(pdf_files, output_dir, config):
    """
    批量提取 PDF（供 pipeline 调用，接口与原 zip 版兼容）。

    返回 list[(pdf_file, output_md|None, error)]
    """
    extractor = PDFExtractor(config)
    results = []
    from tqdm import tqdm
    for pdf_file in tqdm(pdf_files, desc="PDF 提取"):
        output = extractor.extract(pdf_file, output_dir)
        error = "" if output else extractor.last_error
        results.append((pdf_file, output, error))
    return results
