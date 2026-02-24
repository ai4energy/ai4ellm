import argparse
import sys
import json
from pathlib import Path
from enhanced_pdf_processor import EnhancedPDFProcessor

def load_config(config_path: str = None):
    """从JSON文件加载配置"""
    default_config = {
        "processing_options": {
            "min_text_length": 50,
            "max_chunk_size": 1000,
            "overlap": 100,
            "deduplication_threshold": 0.9,
            "supported_formats": [".pdf"],
            "save_format": "jsonl"
        }
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            # 合并用户配置与默认配置
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value

    return default_config

def main():
    parser = argparse.ArgumentParser(description="增强版PDF到语料库处理器")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="包含PDF文件的输入文件夹")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="处理后文件的输出文件夹")
    parser.add_argument("--config", "-c", type=str,
                        help="配置文件路径(JSON)")
    parser.add_argument("--chunk-size", type=int,
                        help="文档分段的最大块大小")
    parser.add_argument("--overlap", type=int,
                        help="块之间的重叠量")
    parser.add_argument("--threshold", type=float,
                        help="去重的相似度阈值")
    parser.add_argument("--format", type=str, choices=['jsonl', 'txt'],
                        help="输出格式")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    opts = config["processing_options"]

    # 用命令行参数覆盖配置文件中的设置
    if args.chunk_size is not None:
        opts["max_chunk_size"] = args.chunk_size
    if args.overlap is not None:
        opts["overlap"] = args.overlap
    if args.threshold is not None:
        opts["deduplication_threshold"] = args.threshold
    if args.format is not None:
        opts["save_format"] = args.format

    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径 '{args.input}' 不存在")
        sys.exit(1)

    # 创建处理器实例（当前版本的EnhancedPDFProcessor不接受这些参数，我们会稍后更新类）
    processor = EnhancedPDFProcessor(min_text_length=opts["min_text_length"])

    print(f"正在从 '{args.input}' 处理PDF到 '{args.output}'...")
    print(f"配置: 块大小={opts['max_chunk_size']}, 重叠={opts['overlap']}, "
          f"去重阈值={opts['deduplication_threshold']}")

    # 处理文件夹
    results = processor.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        file_extensions=opts["supported_formats"],
        save_as_jsonl=(opts["save_format"] == 'jsonl')
    )

    print(f"处理完成!")
    print(f"总文件数: {results['stats']['total_files']}")
    print(f"成功: {results['stats']['successful']}")
    print(f"失败: {results['stats']['failed']}")

    if results['failed']:
        print("\n失败的文件:")
        for failed in results['failed']:
            print(f"  - {failed['file']}: {failed['error']}")

if __name__ == "__main__":
    main()