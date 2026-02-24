import unittest
from enhanced_pdf_processor import EnhancedPDFProcessor
import tempfile
import os

class TestEnhancedPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = EnhancedPDFProcessor()

    def test_clean_text_removes_page_numbers(self):
        text_with_page_nums = "Some content here 1 \\n\\n More content 2 \\n\\n Final content 3"
        cleaned = self.processor.clean_text(text_with_page_nums)
        # 检查是否移除了简单的页码
        self.assertNotIn("1", cleaned)
        self.assertNotIn("2", cleaned)
        self.assertNotIn("3", cleaned)

    def test_segment_document_splits_correctly(self):
        long_text = "Sentence one. " * 100  # 创建长文本
        chunks = self.processor.segment_document(long_text, max_chunk_size=100, overlap=10)
        # 应该因为长度而分成多个块
        self.assertGreater(len(chunks), 1)

        # 检查块是否在大小限制内（大约）
        for chunk in chunks:
            self.assertLess(len(chunk), 150)  # 允许超过限制的一些缓冲

    def test_minimal_text_handling(self):
        result = self.processor.process_single_pdf("non_existent_file.pdf")
        # 应该优雅地处理
        self.assertIn('chunks', result)
        self.assertIsInstance(result['chunks'], list)

if __name__ == '__main__':
    unittest.main()