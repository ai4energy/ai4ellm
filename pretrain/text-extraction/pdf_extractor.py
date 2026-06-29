"""
PDF文本提取模块 - 多引擎 + 智能回退

引擎优先级:
1. PyMuPDF - 快速文本提取（首选）
2. MinerU - 高质量 OCR（扫描件/复杂排版）
3. pdfplumber - 表格提取友好（备选）

使用方式:
    from pretrain.text_extraction.pdf_extractor import PDFExtractorPipeline
    
    pipeline = PDFExtractorPipeline()
    pipeline.process_folder("input/pdfs", "output/text")
"""

import os
import json
import tempfile
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """提取结果"""
    text: str
    pages: int
    method: str
    success: bool
    error: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePDFExtractor(ABC):
    """PDF提取器基类"""

    @abstractmethod
    def extract(self, pdf_path: str) -> ExtractionResult:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class PyMuPDFExtractor(BasePDFExtractor):
    """PyMuPDF提取器 - 快速文本提取"""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import fitz
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50) -> ExtractionResult:
        if not self.is_available():
            return ExtractionResult(
                text="", pages=0, method="pymupdf",
                success=False, error="PyMuPDF未安装"
            )

        try:
            import fitz
            # 抑制 MuPDF 警告
            fitz.TOOLS.mupdf_warnings(0)
            
            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())

            doc.close()

            full_text = "\n\n".join(text_parts)

            if len(full_text.strip()) < min_text_length:
                return ExtractionResult(
                    text="", pages=doc.page_count, method="pymupdf",
                    success=False, error="文本过少（可能是扫描件）"
                )

            return ExtractionResult(
                text=full_text,
                pages=doc.page_count,
                method="pymupdf",
                success=True
            )

        except Exception as e:
            return ExtractionResult(
                text="", pages=0, method="pymupdf",
                success=False, error=str(e)
            )


class MinerUExtractor(BasePDFExtractor):
    """MinerU提取器 - 高质量PDF提取，支持OCR"""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from magic_pdf.data.dataset import PymuDocDataset
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50) -> ExtractionResult:
        if not self.is_available():
            return ExtractionResult(
                text="", pages=0, method="mineru",
                success=False, error="MinerU未安装"
            )

        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod

            # 读取PDF
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)
            ds = PymuDocDataset(pdf_bytes)

            # 判断解析方法
            parse_method = ds.classify()

            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as tmpdir:
                image_writer = FileBasedDataWriter(tmpdir)
                md_writer = FileBasedDataWriter(tmpdir)

                # 执行提取
                if parse_method == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)

                # 获取Markdown
                name = Path(pdf_path).stem
                pipe_result.dump_md(md_writer, f"{name}.md", "images")

                md_path = os.path.join(tmpdir, f"{name}.md")
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = ""

                if len(text.strip()) < min_text_length:
                    return ExtractionResult(
                        text="", pages=ds.page_count, method="mineru",
                        success=False, error="OCR提取文本过少"
                    )

                return ExtractionResult(
                    text=text,
                    pages=ds.page_count,
                    method="mineru",
                    success=True,
                    metadata={"parse_method": parse_method.value}
                )

        except Exception as e:
            return ExtractionResult(
                text="", pages=0, method="mineru",
                success=False, error=str(e)
            )


class PDFPlumberExtractor(BasePDFExtractor):
    """pdfplumber提取器 - 表格提取友好"""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pdfplumber
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def extract(self, pdf_path: str, min_text_length: int = 50) -> ExtractionResult:
        if not self.is_available():
            return ExtractionResult(
                text="", pages=0, method="pdfplumber",
                success=False, error="pdfplumber未安装"
            )

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
                return ExtractionResult(
                    text="", pages=pages, method="pdfplumber",
                    success=False, error="文本过少"
                )

            return ExtractionResult(
                text=full_text,
                pages=pages,
                method="pdfplumber",
                success=True
            )

        except Exception as e:
            return ExtractionResult(
                text="", pages=0, method="pdfplumber",
                success=False, error=str(e)
            )


class PDFExtractorPipeline:
    """PDF提取流水线 - 多引擎智能回退"""

    def __init__(
        self,
        use_pymupdf: bool = True,
        use_mineru: bool = True,
        use_pdfplumber: bool = False,
        mineru_use_gpu: bool = False,
    ):
        self.extractors = []
        self.extractor_names = []

        if use_pymupdf:
            self.extractors.append(PyMuPDFExtractor())
            self.extractor_names.append("pymupdf")

        if use_mineru:
            self.extractors.append(MinerUExtractor(use_gpu=mineru_use_gpu))
            self.extractor_names.append("mineru")

        if use_pdfplumber:
            self.extractors.append(PDFPlumberExtractor())
            self.extractor_names.append("pdfplumber")

        # 检查可用引擎
        available = []
        for name, extractor in zip(self.extractor_names, self.extractors):
            if extractor.is_available():
                available.append(name)
        
        logger.info(f"可用提取引擎: {available}")
        if not available:
            logger.warning("没有可用的提取引擎！")

    def extract_with_fallback(self, pdf_path: str, min_text_length: int = 50) -> ExtractionResult:
        """
        多引擎提取 + 智能回退
        
        按优先级尝试每个引擎，直到成功。
        """
        for extractor in self.extractors:
            if not extractor.is_available():
                continue

            result = extractor.extract(pdf_path, min_text_length)

            if result.success:
                return result

        # 所有方法都失败
        return ExtractionResult(
            text="",
            pages=0,
            method="fallback",
            success=False,
            error="所有提取方法都失败"
        )

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        min_text_length: int = 50,
        skip_existing: bool = True,
    ) -> Dict:
        """处理整个文件夹"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "empty": 0,
            "skipped": 0,
            "by_method": {},
            "errors": []
        }

        pdf_files = list(input_path.rglob("*.pdf"))
        stats["total"] = len(pdf_files)

        from tqdm import tqdm
        for pdf_file in tqdm(pdf_files, desc="处理PDF"):
            rel_path = pdf_file.relative_to(input_path)
            cat = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
            output_file = output_path / cat / f"{pdf_file.stem}.txt"

            # 跳过已存在的文件
            if skip_existing and output_file.exists() and output_file.stat().st_size > 50:
                stats["skipped"] += 1
                stats["success"] += 1
                continue

            result = self.extract_with_fallback(str(pdf_file), min_text_length)

            if result.success:
                # 保存结果
                os.makedirs(output_file.parent, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)

                stats["success"] += 1
                method = result.method
                stats["by_method"][method] = stats["by_method"].get(method, 0) + 1
            elif result.error and "文本过少" in result.error:
                stats["empty"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append({
                    "file": str(pdf_file),
                    "error": result.error
                })

        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF文本提取（多引擎）")
    parser.add_argument("--input", "-i", required=True, help="输入PDF文件或文件夹")
    parser.add_argument("--output", "-o", required=True, help="输出路径")
    parser.add_argument("--method", choices=["pymupdf", "mineru", "pdfplumber", "auto"], 
                        default="auto", help="提取方法")
    parser.add_argument("--min-length", type=int, default=50, help="最小文本长度")
    parser.add_argument("--no-gpu", action="store_true", help="禁用GPU")

    args = parser.parse_args()

    pipeline = PDFExtractorPipeline(
        use_pymupdf=args.method in ["pymupdf", "auto"],
        use_mineru=args.method in ["mineru", "auto"],
        use_pdfplumber=args.method in ["pdfplumber", "auto"],
        mineru_use_gpu=not args.no_gpu,
    )

    if os.path.isfile(args.input):
        result = pipeline.extract_with_fallback(args.input, args.min_length)
        if result.success:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"✅ 提取成功 ({result.method}), {result.pages} 页, {len(result.text)} 字符")
        else:
            print(f"❌ 提取失败: {result.error}")
    else:
        stats = pipeline.process_folder(args.input, args.output, args.min_length)
        print(f"\n✅ 处理完成: 成功 {stats['success']}/{stats['total']}")
        print(f"  方法分布: {stats['by_method']}")
        print(f"  空文本: {stats['empty']}")
        print(f"  失败: {stats['failed']}")


if __name__ == "__main__":
    main()
