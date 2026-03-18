"""
SFT数据验证模块

验证数据集是否符合TRL训练要求
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
from collections import Counter
from dataclasses import dataclass, field
import re


@dataclass
class ValidationStats:
    """验证统计信息"""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    errors: List[Dict] = field(default_factory=list)

    # 字段统计
    field_presence: Dict[str, int] = field(default_factory=dict)

    # 内容统计
    text_lengths: List[int] = field(default_factory=list)
    message_counts: List[int] = field(default_factory=list)
    role_distribution: Counter = field(default_factory=Counter)


class SFTValidator:
    """SFT数据验证器"""

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: 是否启用严格模式
        """
        self.strict = strict

    def validate_messages_format(self, sample: Dict) -> Tuple[bool, List[str]]:
        """验证messages格式"""
        errors = []

        if "messages" not in sample:
            errors.append("缺少 'messages' 字段")
            return False, errors

        messages = sample["messages"]
        if not isinstance(messages, list):
            errors.append("'messages' 必须是列表")
            return False, errors

        if len(messages) < 2:
            errors.append("'messages' 至少需要2条消息")
            return False, errors

        # 验证每条消息
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"消息 {i} 必须是字典")
                continue

            # 检查必需字段
            if "role" not in msg:
                errors.append(f"消息 {i} 缺少 'role' 字段")
            elif msg["role"] not in ["user", "assistant", "system", "tool"]:
                errors.append(f"消息 {i} 的 'role' 值无效: {msg['role']}")

            if "content" not in msg:
                errors.append(f"消息 {i} 缺少 'content' 字段")
            elif not isinstance(msg["content"], str):
                errors.append(f"消息 {i} 的 'content' 必须是字符串")

        # 检查角色顺序
        if messages:
            # 最后一条应该是assistant
            if messages[-1].get("role") != "assistant":
                if self.strict:
                    errors.append("最后一条消息应该是 assistant")

        return len(errors) == 0, errors

    def validate_prompt_completion_format(self, sample: Dict) -> Tuple[bool, List[str]]:
        """验证prompt-completion格式"""
        errors = []

        if "prompt" not in sample:
            errors.append("缺少 'prompt' 字段")
        elif not isinstance(sample["prompt"], str):
            errors.append("'prompt' 必须是字符串")

        if "completion" not in sample:
            errors.append("缺少 'completion' 字段")
        elif not isinstance(sample["completion"], str):
            errors.append("'completion' 必须是字符串")

        return len(errors) == 0, errors

    def detect_format(self, sample: Dict) -> str:
        """检测样本格式"""
        if "messages" in sample:
            return "messages"
        elif "prompt" in sample and "completion" in sample:
            return "prompt-completion"
        elif "text" in sample:
            return "text"
        else:
            return "unknown"

    def validate_file(self, input_path: str, max_samples: int = None) -> ValidationStats:
        """验证整个文件"""
        stats = ValidationStats()

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and line_num > max_samples:
                    break

                stats.total_samples += 1

                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    stats.invalid_samples += 1
                    stats.errors.append({
                        "line": line_num,
                        "error": f"JSON解析错误: {str(e)}"
                    })
                    continue

                # 检测格式并验证
                fmt = self.detect_format(sample)

                if fmt == "messages":
                    valid, errs = self.validate_messages_format(sample)
                elif fmt == "prompt-completion":
                    valid, errs = self.validate_prompt_completion_format(sample)
                else:
                    valid = False
                    errs = ["无法识别的数据格式"]

                if valid:
                    stats.valid_samples += 1
                else:
                    stats.invalid_samples += 1
                    for err in errs:
                        stats.errors.append({"line": line_num, "error": err})

                # 统计信息
                for key in sample.keys():
                    stats.field_presence[key] = stats.field_presence.get(key, 0) + 1

                # 文本长度统计
                if fmt == "messages":
                    total_len = sum(
                        len(m.get("content", ""))
                        for m in sample.get("messages", [])
                    )
                    stats.text_lengths.append(total_len)
                    stats.message_counts.append(len(sample.get("messages", [])))
                    for msg in sample.get("messages", []):
                        stats.role_distribution[msg.get("role", "unknown")] += 1

        return stats

    def generate_report(self, stats: ValidationStats) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("SFT数据验证报告")
        report.append("=" * 60)

        # 基本统计
        report.append("\n## 基本统计")
        report.append(f"  总样本数: {stats.total_samples}")
        report.append(f"  有效样本: {stats.valid_samples} ({stats.valid_samples/max(stats.total_samples,1)*100:.1f}%)")
        report.append(f"  无效样本: {stats.invalid_samples}")

        # 字段统计
        if stats.field_presence:
            report.append("\n## 字段出现频率")
            for field, count in sorted(stats.field_presence.items(), key=lambda x: -x[1]):
                report.append(f"  {field}: {count} ({count/stats.total_samples*100:.1f}%)")

        # 文本长度统计
        if stats.text_lengths:
            report.append("\n## 文本长度统计")
            lengths = stats.text_lengths
            report.append(f"  最小: {min(lengths)}")
            report.append(f"  最大: {max(lengths)}")
            report.append(f"  平均: {sum(lengths)/len(lengths):.1f}")

        # 消息统计
        if stats.message_counts:
            report.append("\n## 消息统计")
            report.append(f"  平均消息数: {sum(stats.message_counts)/len(stats.message_counts):.1f}")
            report.append("  角色分布:")
            for role, count in stats.role_distribution.most_common():
                report.append(f"    {role}: {count}")

        # 错误统计
        if stats.errors:
            report.append(f"\n## 错误 ({len(stats.errors)} 条)")
            error_types = Counter(e["error"] for e in stats.errors)
            for err, count in error_types.most_common(10):
                report.append(f"  {err}: {count} 次")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="SFT数据验证")
    parser.add_argument("--input", "-i", required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", help="报告输出路径")
    parser.add_argument("--strict", action="store_true", help="启用严格模式")
    parser.add_argument("--max-samples", type=int, help="最大验证样本数")

    args = parser.parse_args()

    validator = SFTValidator(strict=args.strict)
    stats = validator.validate_file(args.input, args.max_samples)
    report = validator.generate_report(stats)

    print(report)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()