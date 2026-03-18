#!/usr/bin/env python
"""
预训练语料处理流水线
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretrain.text_extraction.pdf_extractor import PDFExtractorPipeline
from pretrain.cleaning.text_cleaner import TextCleaner, clean_folder
from pretrain.deduplication.semantic_dedup import HybridDeduplicator
from pretrain.format.convert_to_pretrain import PretrainFormatConverter
from tools.dataset.dataset_utils import split_dataset, analyze_dataset


class PretrainPipeline:
    """预训练流水线"""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.output_base = Path("output")
        self.output_base.mkdir(exist_ok=True)

    def run(self):
        """运行完整流水线"""
        print("=" * 60)
        print("预训练语料处理流水线")
        print("=" * 60)
        print(f"开始时间: {datetime.now().isoformat()}")

        results = {}

        # 步骤1: PDF提取
        if self.config.get("extraction", {}).get("pdf", {}).get("enabled", False):
            print("\n[步骤1] PDF文本提取...")
            results["extraction"] = self.run_pdf_extraction()

        # 步骤2: 数据清洗
        if self.config.get("cleaning", {}).get("enabled", True):
            print("\n[步骤2] 数据清洗...")
            results["cleaning"] = self.run_cleaning()

        # 步骤3: 去重
        if self.config.get("deduplication", {}).get("enabled", True):
            print("\n[步骤3] 数据去重...")
            results["deduplication"] = self.run_deduplication()

        # 步骤4: 格式转换
        print("\n[步骤4] 格式转换...")
        results["format"] = self.run_format_conversion()

        # 步骤5: 数据集划分
        if self.config.get("split", {}).get("enabled", True):
            print("\n[步骤5] 数据集划分...")
            results["split"] = self.run_split()

        # 生成报告
        print("\n[完成] 生成报告...")
        self.generate_report(results)

        print(f"\n完成时间: {datetime.now().isoformat()}")
        return results

    def run_pdf_extraction(self) -> dict:
        """运行PDF提取"""
        pdf_config = self.config["extraction"]["pdf"]

        pipeline = PDFExtractorPipeline(
            use_mineru=pdf_config.get("method") in ["mineru", "auto"],
            use_pymupdf=pdf_config.get("method") in ["pymupdf", "auto"],
            ocr_api_key=os.getenv("OCR_API_KEY") if pdf_config.get("use_ocr_api") else None,
            ocr_api_type=pdf_config.get("ocr_api_type")
        )

        input_dir = pdf_config.get("input_dir", "data/pdfs")
        output_dir = pdf_config.get("output_dir", "output/pdf_extracted")

        stats = pipeline.process_folder(input_dir, output_dir)
        print(f"  处理: {stats['total']} 个文件")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")

        return stats

    def run_cleaning(self) -> dict:
        """运行数据清洗"""
        cleaning_config = self.config.get("cleaning", {})

        cleaner = TextCleaner()
        if cleaning_config.get("remove_html", True):
            cleaner.add_cleaner(TextCleaner.remove_html_tags)
        if cleaning_config.get("remove_urls", True):
            cleaner.add_cleaner(TextCleaner.remove_urls)
        if cleaning_config.get("remove_garbled", True):
            cleaner.add_cleaner(TextCleaner.remove_garbled_text)
        if cleaning_config.get("normalize_whitespace", True):
            cleaner.add_cleaner(TextCleaner.normalize_whitespace)

        input_dir = cleaning_config.get("input_dir", "output/extracted")
        output_dir = cleaning_config.get("output_dir", "output/cleaned")

        stats = clean_folder(input_dir, output_dir, cleaner)
        print(f"  处理文件: {stats['files_processed']}")
        print(f"  原始字符: {stats['original_chars']}")
        print(f"  清洗后: {stats['cleaned_chars']}")

        return stats

    def run_deduplication(self) -> dict:
        """运行去重"""
        dedup_config = self.config.get("deduplication", {})

        dedup = HybridDeduplicator(
            semantic_threshold=dedup_config.get("threshold", 0.9)
        )

        input_dir = Path(dedup_config.get("input_dir", "output/cleaned"))
        output_dir = Path(dedup_config.get("output_dir", "output/deduped"))
        output_dir.mkdir(parents=True, exist_ok=True)

        total_stats = {"total": 0, "unique": 0, "duplicates": 0}

        for jsonl_file in input_dir.glob("*.jsonl"):
            output_file = output_dir / jsonl_file.name
            stats = dedup.deduplicate_file(str(jsonl_file), str(output_file), use_semantic=False)

            total_stats["total"] += stats.total
            total_stats["unique"] += stats.unique
            total_stats["duplicates"] += stats.duplicates

        print(f"  总数: {total_stats['total']}")
        print(f"  唯一: {total_stats['unique']}")
        print(f"  重复: {total_stats['duplicates']}")

        return total_stats

    def run_format_conversion(self) -> dict:
        """运行格式转换"""
        format_config = self.config.get("format", {})

        converter = PretrainFormatConverter(
            min_length=format_config.get("min_length", 100),
            max_length=format_config.get("max_length", 1000000)
        )

        input_dir = Path(format_config.get("input_dir", "output/deduped"))
        output_dir = Path(format_config.get("output_dir", "output/pretrain_final"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # 合并所有文件
        input_files = list(input_dir.glob("*.jsonl"))
        merged_output = output_dir / "pretrain.jsonl"

        stats = converter.merge_to_pretrain(
            [str(f) for f in input_files],
            str(merged_output),
            shuffle=True
        )

        print(f"  输入文件: {len(input_files)}")
        print(f"  总样本: {stats['total']}")

        return stats

    def run_split(self) -> dict:
        """运行数据集划分"""
        split_config = self.config.get("split", {})

        input_file = Path("output/pretrain_final/pretrain.jsonl")
        output_dir = Path(split_config.get("output_dir", "output/pretrain_dataset"))

        stats = split_dataset(
            str(input_file),
            str(output_dir),
            test_size=split_config.get("test_size", 0.01),
            val_size=split_config.get("val_size", 0.01),
            seed=split_config.get("seed", 42)
        )

        print(f"  训练集: {stats['train']}")
        print(f"  验证集: {stats['val']}")
        print(f"  测试集: {stats['test']}")

        return stats

    def generate_report(self, results: dict):
        """生成报告"""
        report_path = self.output_base / "pretrain_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"  报告保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="预训练语料处理流水线")
    parser.add_argument("--config", "-c", default="configs/pretrain.yaml", help="配置文件")
    args = parser.parse_args()

    pipeline = PretrainPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()