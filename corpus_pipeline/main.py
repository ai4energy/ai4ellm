#!/usr/bin/env python3
"""
语料库构建流水线 — 主入口（预训练语料）

从 PDF/Word/PPT/TXT/Markdown 文档提取文本，章节感知清洗、4 维质量评分、
结构化切块、规则/语义去重，最终生成富格式语料 + 预训练 JSONL。

用法:
    python main.py                                 # 使用默认 config.yaml
    python main.py --config custom.yaml             # 指定配置文件
    python main.py --input /path/to/docs            # 覆盖输入目录
    python main.py --output /path/to/out            # 覆盖输出目录
"""

import argparse
import sys
import os

# 将项目根目录加入 Python 路径，确保 src.* 导入正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.pipeline import CorpusPipeline


def main():
    parser = argparse.ArgumentParser(
        description="语料库构建流水线 — 从文档到富格式语料 + 预训练 JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                                  # 使用 config.yaml 默认配置
    python main.py --config custom.yaml             # 指定配置文件
    python main.py --input ../origin-files-organized/102工程热力学   # 单类目冒烟
    python main.py --input ../origin-files-organized --output ./out  # 全量
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径（默认: config.yaml）")
    parser.add_argument("--input", default=None, help="输入文档目录（覆盖 config.yaml 的 input_dir）")
    parser.add_argument("--output", default=None, help="输出根目录（覆盖 config.yaml 的 output_dir）")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        print("请确保配置文件存在，或使用 --config 指定路径。", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 命令行参数覆盖配置
    if args.input:
        config.raw["paths"]["input_dir"] = os.path.abspath(args.input)
    if args.output:
        abs_output = os.path.abspath(args.output)
        config.raw["paths"]["output_dir"] = abs_output
        # 日志目录跟着输出目录走（避免 CLI --output 写日志到默认 ./output/logs）
        if "logging" in config.raw:
            config.raw["logging"]["log_dir"] = os.path.join(abs_output, "logs")

    pipeline = CorpusPipeline(config)
    stats = pipeline.run()

    # 控制台输出统计报告
    print(stats.generate_report())

    # 有错误则返回非零退出码
    if stats.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
