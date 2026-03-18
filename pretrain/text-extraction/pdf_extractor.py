"""
PDF文本提取模块

支持多种提取方法：
1. MinerU - 高质量PDF提取（支持OCR）
2. OCR API - 调用外部OCR服务（适用于扫描件）
3. PyMuPDF - 快速文本提取
4. pdfplumber - 表格提取友好
"""

import os
import json
import asyncio
import argparse
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
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
        """提取PDF文本"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查提取器是否可用"""
        pass


class MinerUExtractor(BasePDFExtractor):
    """MinerU提取器 - 高质量PDF提取，支持OCR"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from magic_pdf.data.dataset import PymuDocDataset
                self._available = True
            except ImportError:
                logger.warning("MinerU未安装，请运行: pip install magic-pdf[full]")
                self._available = False
        return self._available

    def extract(self, pdf_path: str) -> ExtractionResult:
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

            # 创建输出目录
            output_dir = Path(pdf_path).parent / "output" / Path(pdf_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            image_dir = output_dir / "images"
            image_dir.mkdir(exist_ok=True)

            image_writer = FileBasedDataWriter(str(image_dir))
            md_writer = FileBasedDataWriter(str(output_dir))

            # 执行提取
            if parse_method == SupportedPdfParseMethod.OCR:
                logger.info(f"使用OCR模式处理: {pdf_path}")
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                logger.info(f"使用文本模式处理: {pdf_path}")
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            # 获取文本
            name = Path(pdf_path).stem
            pipe_result.dump_md(md_writer, f"{name}.md", "images")

            # 读取生成的Markdown
            md_path = output_dir / f"{name}.md"
            if md_path.exists():
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = ""

            return ExtractionResult(
                text=text,
                pages=ds.page_count,
                method="mineru",
                success=True,
                metadata={"output_dir": str(output_dir)}
            )

        except Exception as e:
            logger.error(f"MinerU提取失败: {str(e)}")
            return ExtractionResult(
                text="", pages=0, method="mineru",
                success=False, error=str(e)
            )


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
                logger.warning("PyMuPDF未安装，请运行: pip install PyMuPDF")
                self._available = False
        return self._available

    def extract(self, pdf_path: str) -> ExtractionResult:
        if not self.is_available():
            return ExtractionResult(
                text="", pages=0, method="pymupdf",
                success=False, error="PyMuPDF未安装"
            )

        try:
            import fitz

            doc = fitz.open(pdf_path)
            text_parts = []

            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_parts.append(page.get_text())

            doc.close()

            return ExtractionResult(
                text="\n\n".join(text_parts),
                pages=doc.page_count,
                method="pymupdf",
                success=True
            )

        except Exception as e:
            logger.error(f"PyMuPDF提取失败: {str(e)}")
            return ExtractionResult(
                text="", pages=0, method="pymupdf",
                success=False, error=str(e)
            )


class OCRAPIClient:
    """OCR API客户端基类"""

    @abstractmethod
    async def recognize(self, image_path: str) -> str:
        """识别图片中的文字"""
        pass


class DeepSeekOCRClient(OCRAPIClient):
    """DeepSeek Vision OCR客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from openai import AsyncOpenAI
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    async def recognize(self, image_path: str) -> str:
        """使用DeepSeek Vision进行OCR"""
        try:
            from openai import AsyncOpenAI
            import base64

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1"
            )

            # 读取图片并编码
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()

            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请识别图片中的所有文字，保持原有格式输出。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=4096
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek OCR失败: {str(e)}")
            return ""


class QwenOCRClient(OCRAPIClient):
    """通义千问OCR客户端"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def recognize(self, image_path: str) -> str:
        """使用通义千问Vision进行OCR"""
        try:
            import dashscope
            from dashscope import MultiModalConversation

            dashscope.api_key = self.api_key

            with open(image_path, 'rb') as f:
                image_url = f"data:image/jpeg;base64,{f.read().hex()}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": "请识别图片中的所有文字，保持原有格式输出。"}
                    ]
                }
            ]

            response = MultiModalConversation.call(
                model='qwen-vl-max',
                messages=messages
            )

            return response.output.choices[0].message.content[0]['text']

        except Exception as e:
            logger.error(f"通义千问OCR失败: {str(e)}")
            return ""


class PDFExtractorPipeline:
    """PDF提取流水线"""

    def __init__(
        self,
        use_mineru: bool = True,
        use_pymupdf: bool = True,
        ocr_api_key: str = None,
        ocr_api_type: str = None
    ):
        """
        Args:
            use_mineru: 是否使用MinerU
            use_pymupdf: 是否使用PyMuPDF作为备选
            ocr_api_key: OCR API密钥
            ocr_api_type: OCR API类型 (deepseek, qwen)
        """
        self.extractors = []

        if use_mineru:
            self.extractors.append(MinerUExtractor())

        if use_pymupdf:
            self.extractors.append(PyMuPDFExtractor())

        # OCR客户端
        self.ocr_client = None
        if ocr_api_key and ocr_api_type:
            if ocr_api_type == "deepseek":
                self.ocr_client = DeepSeekOCRClient(ocr_api_key)
            elif ocr_api_type == "qwen":
                self.ocr_client = QwenOCRClient(ocr_api_key)

    def extract_with_fallback(self, pdf_path: str, min_text_length: int = 100) -> ExtractionResult:
        """
        使用多个提取器尝试提取，直到成功

        Args:
            pdf_path: PDF文件路径
            min_text_length: 最小有效文本长度
        """
        for extractor in self.extractors:
            if not extractor.is_available():
                continue

            result = extractor.extract(pdf_path)

            if result.success and len(result.text.strip()) >= min_text_length:
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
        min_text_length: int = 100
    ) -> Dict:
        """处理整个文件夹"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

        pdf_files = list(input_path.rglob("*.pdf"))

        from tqdm import tqdm
        for pdf_file in tqdm(pdf_files, desc="处理PDF"):
            stats["total"] += 1

            result = self.extract_with_fallback(str(pdf_file), min_text_length)

            if result.success:
                # 保存结果
                output_file = output_path / f"{pdf_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)

                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["errors"].append({
                    "file": str(pdf_file),
                    "error": result.error
                })

        return stats


def main():
    parser = argparse.ArgumentParser(description="PDF文本提取")
    parser.add_argument("--input", "-i", required=True, help="输入PDF文件或文件夹")
    parser.add_argument("--output", "-o", required=True, help="输出路径")
    parser.add_argument("--method", choices=["mineru", "pymupdf", "auto"], default="auto", help="提取方法")
    parser.add_argument("--min-length", type=int, default=100, help="最小文本长度")
    parser.add_argument("--ocr-api-key", help="OCR API密钥")
    parser.add_argument("--ocr-api-type", choices=["deepseek", "qwen"], help="OCR API类型")

    args = parser.parse_args()

    pipeline = PDFExtractorPipeline(
        use_mineru=args.method in ["mineru", "auto"],
        use_pymupdf=args.method in ["pymupdf", "auto"],
        ocr_api_key=args.ocr_api_key,
        ocr_api_type=args.ocr_api_type
    )

    if os.path.isfile(args.input):
        result = pipeline.extract_with_fallback(args.input, args.min_length)
        if result.success:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"提取成功，共 {result.pages} 页")
        else:
            print(f"提取失败: {result.error}")
    else:
        stats = pipeline.process_folder(args.input, args.output, args.min_length)
        print(f"处理完成: 成功 {stats['success']}/{stats['total']}")


if __name__ == "__main__":
    main()