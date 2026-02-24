"""
使用示例：增强PDF处理库
"""

from enhanced_pdf_processor import EnhancedPDFProcessor
import json

def example_usage():
    # 创建处理器实例，使用自定义参数
    processor = EnhancedPDFProcessor(
        min_text_length=50,
        max_chunk_size=1000,
        overlap=100,
        dedup_threshold=0.9
    )

    # 处理单个PDF文件
    print("处理单个PDF文件示例...")
    # 注意：此示例需要实际的PDF文件存在才能运行
    # result = processor.process_single_pdf("path/to/your/document.pdf")

    # 以下是一个虚拟演示，展示返回的数据结构
    sample_result = {
        'filename': 'sample.pdf',
        'raw_text_length': 10000,
        'cleaned_text_length': 8000,
        'initial_chunks_count': 15,
        'final_chunks_count': 12,
        'chunks': [
            "这是第一个文本块的内容...",
            "这是第二个文本块的内容...",
            "这是第三个文本块的内容..."
        ]
    }

    print(f"文件名: {sample_result['filename']}")
    print(f"原始文本长度: {sample_result['raw_text_length']}")
    print(f"清理后文本长度: {sample_result['cleaned_text_length']}")
    print(f"初始块数量: {sample_result['initial_chunks_count']}")
    print(f"去重后块数量: {sample_result['final_chunks_count']}")

    # 打印前几个文本块
    for i, chunk in enumerate(sample_result['chunks'][:3]):
        print(f"文本块 {i+1}: {chunk[:100]}...")

def demo_with_config():
    """
    使用配置参数演示
    """
    # 从配置文件加载参数（这里模拟）
    config = {
        "processing_options": {
            "min_text_length": 100,
            "max_chunk_size": 2000,
            "overlap": 200,
            "deduplication_threshold": 0.85,
            "supported_formats": [".pdf"],
            "save_format": "jsonl"
        }
    }

    opts = config["processing_options"]

    # 使用配置参数创建处理器
    processor = EnhancedPDFProcessor(
        min_text_length=opts["min_text_length"],
        max_chunk_size=opts["max_chunk_size"],
        overlap=opts["overlap"],
        dedup_threshold=opts["deduplication_threshold"]
    )

    print(f"处理器已配置：")
    print(f"- 最小文本长度: {processor.min_text_length}")
    print(f"- 最大块大小: {processor.max_chunk_size}")
    print(f"- 重叠大小: {processor.overlap}")
    print(f"- 去重阈值: {processor.dedup_threshold}")


if __name__ == "__main__":
    print("=== 增强PDF处理库使用示例 ===\n")

    example_usage()
    print("\n" + "="*50 + "\n")
    demo_with_config()