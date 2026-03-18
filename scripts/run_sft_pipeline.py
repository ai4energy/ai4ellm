#!/usr/bin/env python
"""
SFT语料处理流水线
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sft.data_generation.generate_qa import (
    get_client,
    QAGenerator,
    SFTDataGenerator,
    APIConfig,
)
from sft.format_conversion.convert_to_sft import BatchConverter, SFTFormatConverter
from sft.validation.validate_dataset import SFTValidator
from tools.dataset.dataset_utils import split_dataset, analyze_dataset


class SFTPipeline:
    """SFT流水线"""

    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.output_base = Path("output")
        self.output_base.mkdir(exist_ok=True)

    def run(self):
        """运行完整流水线"""
        print("=" * 60)
        print("SFT语料处理流水线")
        print("=" * 60)
        print(f"开始时间: {datetime.now().isoformat()}")

        results = {}

        # 步骤1: 数据生成
        print("\n[步骤1] API数据生成...")
        results["generation"] = asyncio.run(self.run_generation())

        # 步骤2: 格式转换
        print("\n[步骤2] 格式转换...")
        results["format"] = self.run_format_conversion()

        # 步骤3: 数据验证
        print("\n[步骤3] 数据验证...")
        results["validation"] = self.run_validation()

        # 步骤4: 数据集划分
        if self.config.get("split", {}).get("enabled", True):
            print("\n[步骤4] 数据集划分...")
            results["split"] = self.run_split()

        # 生成报告
        print("\n[完成] 生成报告...")
        self.generate_report(results)

        print(f"\n完成时间: {datetime.now().isoformat()}")
        return results

    async def run_generation(self) -> dict:
        """运行数据生成"""
        api_config = self.config.get("api", {})
        generation_config = self.config.get("generation", {})

        # 获取API密钥
        api_key = os.getenv("API_KEY") or os.getenv(f"{api_config['type'].upper()}_API_KEY")
        if not api_key:
            print("  警告: 未配置API密钥，跳过数据生成")
            return {"skipped": True, "reason": "API密钥未配置"}

        # 创建API客户端
        config = APIConfig(
            api_key=api_key,
            model=api_config.get("model", ""),
            max_tokens=api_config.get("max_tokens", 2048),
            temperature=api_config.get("temperature", 0.7),
            max_retries=api_config.get("max_retries", 3),
        )

        client = get_client(api_config["type"], config)
        qa_generator = QAGenerator(client, generation_config.get("qa", {}).get("domain", "general"))

        results = {}

        # 问答对生成
        qa_config = generation_config.get("qa", {})
        if qa_config.get("enabled", True):
            print("  生成问答对...")
            generator = SFTDataGenerator(client, qa_config.get("output_file", "output/sft_qa.jsonl"))
            generator.qa_generator = qa_generator

            input_file = qa_config.get("input_file")
            if input_file and os.path.exists(input_file):
                count = await generator.generate_from_file(
                    input_file,
                    num_questions_per_text=qa_config.get("num_questions_per_text", 3),
                    concurrency=api_config.get("concurrency", 5)
                )
                results["qa_count"] = count
                print(f"    生成问答对: {count}")
            else:
                print(f"    输入文件不存在: {input_file}")

        return results

    def run_format_conversion(self) -> dict:
        """运行格式转换"""
        format_config = self.config.get("format", {})

        converter = SFTFormatConverter()
        batch_converter = BatchConverter(converter)

        input_dir = Path(format_config.get("input_dir", "output/generated"))
        output_dir = Path(format_config.get("output_dir", "output/sft_formatted"))
        output_dir.mkdir(parents=True, exist_ok=True)

        total_stats = {"total": 0, "converted": 0, "skipped": 0, "errors": 0}

        for jsonl_file in input_dir.glob("*.jsonl"):
            output_file = output_dir / jsonl_file.name

            stats = batch_converter.convert_file(
                str(jsonl_file),
                str(output_file),
                input_format=format_config.get("input_format", "auto"),
                output_format=format_config.get("output_format", "messages"),
                min_length=format_config.get("min_length", 10),
                max_length=format_config.get("max_length", 50000)
            )

            total_stats["total"] += stats["total"]
            total_stats["converted"] += stats["converted"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["errors"] += stats["errors"]

        print(f"  总数: {total_stats['total']}")
        print(f"  转换: {total_stats['converted']}")
        print(f"  跳过: {total_stats['skipped']}")

        return total_stats

    def run_validation(self) -> dict:
        """运行数据验证"""
        validation_config = self.config.get("validation", {})
        format_config = self.config.get("format", {})

        input_dir = Path(format_config.get("output_dir", "output/sft_formatted"))

        validator = SFTValidator(strict=validation_config.get("strict", False))

        total_stats = {"total": 0, "valid": 0, "invalid": 0}

        for jsonl_file in input_dir.glob("*.jsonl"):
            stats = validator.validate_file(str(jsonl_file))

            total_stats["total"] += stats.total_samples
            total_stats["valid"] += stats.valid_samples
            total_stats["invalid"] += stats.invalid_samples

        print(f"  总数: {total_stats['total']}")
        print(f"  有效: {total_stats['valid']}")
        print(f"  无效: {total_stats['invalid']}")

        return total_stats

    def run_split(self) -> dict:
        """运行数据集划分"""
        split_config = self.config.get("split", {})
        format_config = self.config.get("format", {})

        # 合并所有文件
        input_dir = Path(format_config.get("output_dir", "output/sft_formatted"))
        merged_file = self.output_base / "sft_merged.jsonl"

        # 合并文件
        with open(merged_file, 'w', encoding='utf-8') as f_out:
            for jsonl_file in input_dir.glob("*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        if line.strip():
                            f_out.write(line)

        # 划分数据集
        output_dir = Path(split_config.get("output_dir", "output/sft_dataset"))

        stats = split_dataset(
            str(merged_file),
            str(output_dir),
            test_size=split_config.get("test_size", 0.1),
            val_size=split_config.get("val_size", 0.1),
            seed=split_config.get("seed", 42)
        )

        print(f"  训练集: {stats['train']}")
        print(f"  验证集: {stats['val']}")
        print(f"  测试集: {stats['test']}")

        return stats

    def generate_report(self, results: dict):
        """生成报告"""
        report_path = self.output_base / "sft_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "results": results
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"  报告保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT语料处理流水线")
    parser.add_argument("--config", "-c", default="configs/sft.yaml", help="配置文件")
    args = parser.parse_args()

    pipeline = SFTPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()