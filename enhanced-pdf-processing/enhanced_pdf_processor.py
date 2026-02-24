import os
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging
from loguru import logger


class EnhancedPDFProcessor:
    def __init__(self, min_text_length: int = 50, max_chunk_size: int = 1000, overlap: int = 100, dedup_threshold: float = 0.9):
        self.min_text_length = min_text_length
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.dedup_threshold = dedup_threshold
        self.logger = logger

    def extract_with_fallback(self, pdf_path: str) -> str:
        """
        使用多种方法提取PDF文本，如果一种方法失败则使用另一种
        """
        # 优先尝试使用unstructured
        text = self._extract_unstructured(pdf_path)
        if text and len(text.strip()) > self.min_text_length:
            return text

        # 回退到pdfplumber
        try:
            text = self._extract_pdfplumber(pdf_path)
            if text and len(text.strip()) > self.min_text_length:
                return text
        except Exception:
            pass

        # 最后回退到PyMuPDF
        try:
            text = self._extract_pymupdf(pdf_path)
            if text and len(text.strip()) > self.min_text_length:
                return text
        except Exception:
            pass

        # 基础方法
        return self._extract_basic(pdf_path)

    def _extract_unstructured(self, pdf_path: str) -> str:
        """使用unstructured库提取文本"""
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=pdf_path,
                strategy="auto",  # 自动选择最佳策略
            )
            return "\n\n".join([str(el) for el in elements])
        except ImportError:
            self.logger.warning("unstructured库未安装，跳过...")
            return ""
        except Exception as e:
            self.logger.warning(f"Unstructured提取失败: {str(e)}")
            return ""

    def _extract_pdfplumber(self, pdf_path: str) -> str:
        """使用pdfplumber提取文本"""
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            self.logger.warning("pdfplumber库未安装，跳过...")
            return ""
        except Exception as e:
            self.logger.warning(f"Pdfplumber提取失败: {str(e)}")
            return ""

    def _extract_pymupdf(self, pdf_path: str) -> str:
        """使用PyMuPDF提取文本"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_parts.append(text)
            doc.close()
            return "\n\n".join(text_parts)
        except ImportError:
            self.logger.warning("PyMuPDF库未安装，跳过...")
            return ""
        except Exception as e:
            self.logger.warning(f"PyMuPDF提取失败: {str(e)}")
            return ""

    def _extract_basic(self, pdf_path: str) -> str:
        """基础的PyPDF2提取方法作为最后手段"""
        try:
            import PyPDF2
            text_parts = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)
        except ImportError:
            self.logger.warning("PyPDF2库未安装，跳过...")
            return ""
        except Exception as e:
            self.logger.error(f"基础PDF提取失败: {str(e)}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        改进的文本清理方法
        """
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)

        # 移除常见的页眉页脚模式
        patterns_to_remove = [
            r'\b\d+\s*',  # 页码
            r'Copyright\s+.*',  # 版权信息
            r'This material may be.*',  # 版权声明
            r'Retrieved from http.*',  # 检索声明
            r'All rights reserved.*',  # 权限声明
            r'^\s*第\s*\d+\s*页\s*共\s*\d+\s*页\s*$',  # 中文页码
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.MULTILINE)

        # 修复因过度清理造成的字符断裂
        text = re.sub(r'([a-zA-Z])\s([a-zA-Z])', r'\1\2', text)  # 连接被空格分开的单词

        return text.strip()

    def segment_document(self, text: str, max_chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        智能分割文档
        """
        # 使用传入的参数或实例默认值
        max_chunk_size = max_chunk_size or self.max_chunk_size
        overlap = overlap or self.overlap

        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk + para) <= max_chunk_size:
                current_chunk += " " + para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果段落太长，按句子切分
                if len(para) > max_chunk_size:
                    sentences = re.split(r'[.!?。！？]+', para)
                    temp_chunk = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(temp_chunk + sent) <= max_chunk_size:
                            temp_chunk += " " + sent
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sent
                            else:
                                # 单句仍然太长，按字符长度强制分割
                                for i in range(0, len(sent), max_chunk_size):
                                    chunks.append(sent[i:i+max_chunk_size])
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # 应用重叠策略
        if overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0 and len(chunk) < max_chunk_size:
                    # 添加来自前一个块的重叠内容
                    prev_chunk_end = chunks[i-1][-overlap:]
                    chunk = prev_chunk_end + " " + chunk
                overlapped_chunks.append(chunk)
            return overlapped_chunks

        return chunks

    def semantic_deduplicate(self, texts: List[str], threshold: float = None) -> List[str]:
        """
        基于语义相似度的去重
        """
        threshold = threshold or self.dedup_threshold

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer('all-MiniLM-L6-v2')

            # 计算所有文本的嵌入向量
            embeddings = model.encode(texts, convert_to_numpy=True)

            # 计算余弦相似度矩阵
            similarity_matrix = np.dot(embeddings, embeddings.T)

            # 保留与已保留文本相似度低于阈值的文本
            retained_indices = []
            for i in range(len(texts)):
                is_similar = False
                for j in retained_indices:
                    if similarity_matrix[i][j] > threshold:
                        is_similar = True
                        break
                if not is_similar:
                    retained_indices.append(i)

            return [texts[i] for i in retained_indices]
        except ImportError:
            self.logger.warning("sentence-transformers未安装，跳过语义去重...")
            return texts
        except Exception as e:
            self.logger.error(f"语义去重失败: {str(e)}")
            return texts

    def process_single_pdf(self, pdf_path: str, max_chunk_size: int = None, overlap: int = None, dedup_threshold: float = None) -> Dict:
        """
        处理单个PDF文件
        """
        pdf_file = Path(pdf_path)

        # 提取文本
        raw_text = self.extract_with_fallback(str(pdf_file))

        # 清理文本
        cleaned_text = self.clean_text(raw_text)

        # 分割文档
        chunks = self.segment_document(cleaned_text, max_chunk_size, overlap)

        # 语义去重（可选）
        deduplicated_chunks = self.semantic_deduplicate(chunks, dedup_threshold)

        return {
            'filename': pdf_file.name,
            'raw_text_length': len(raw_text),
            'cleaned_text_length': len(cleaned_text),
            'initial_chunks_count': len(chunks),
            'final_chunks_count': len(deduplicated_chunks),
            'chunks': deduplicated_chunks
        }

    def process_folder(self, input_folder: str, output_folder: str,
                      file_extensions: List[str] = ['.pdf'],
                      save_as_jsonl: bool = True,
                      max_chunk_size: int = None,
                      overlap: int = None,
                      dedup_threshold: float = None) -> Dict:
        """
        处理整个文件夹中的PDF文件
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'processed': [],
            'failed': [],
            'stats': {'total_files': 0, 'successful': 0, 'failed': 0}
        }

        pdf_files = []
        for ext in file_extensions:
            pdf_files.extend(list(input_path.rglob(f'*{ext}')))

        results['stats']['total_files'] = len(pdf_files)

        from tqdm import tqdm
        for pdf_file in tqdm(pdf_files, desc="处理PDF"):
            try:
                # 处理单个文件
                file_result = self.process_single_pdf(
                    str(pdf_file),
                    max_chunk_size=max_chunk_size,
                    overlap=overlap,
                    dedup_threshold=dedup_threshold
                )

                # 保存处理结果
                if save_as_jsonl:
                    output_file = output_path / f"{pdf_file.stem}_processed.jsonl"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        import json
                        for i, chunk in enumerate(file_result['chunks']):
                            record = {
                                'id': f"{pdf_file.stem}_chunk_{i}",
                                'source_file': pdf_file.name,
                                'content': chunk,
                                'chunk_index': i
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                else:
                    output_file = output_path / f"{pdf_file.stem}_processed.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for i, chunk in enumerate(file_result['chunks']):
                            f.write(f"<CHUNK_{i}>\n{chunk}\n</CHUNK_{i}>\n\n")

                file_result['output_file'] = str(output_file)
                results['processed'].append(file_result)

                results['stats']['successful'] += 1
                self.logger.info(f"成功处理: {pdf_file.name}")

            except Exception as e:
                results['failed'].append({'file': str(pdf_file), 'error': str(e)})
                results['stats']['failed'] += 1
                self.logger.error(f"处理失败 {pdf_file.name}: {str(e)}")

        # 生成统计报告
        stats_file = output_path / "processing_stats.json"
        import json
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results