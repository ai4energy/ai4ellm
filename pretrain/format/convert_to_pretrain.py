"""
预训练格式转换模块

将各种格式的数据转换为预训练格式：
- 纯文本格式
- JSONL格式
- Parquet格式
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm


class PretrainFormatConverter:
    """预训练格式转换器"""

    def __init__(self, min_length: int = 100, max_length: int = 1000000):
        """
        Args:
            min_length: 最小文本长度
            max_length: 最大文本长度
        """
        self.min_length = min_length
        self.max_length = max_length

    def text_to_jsonl(self, input_path: str, output_path: str, split_by: str = "line") -> Dict:
        """
        将纯文本转换为JSONL格式

        Args:
            input_path: 输入文本文件
            output_path: 输出JSONL文件
            split_by: 分割方式 (line, paragraph, document)
        """
        stats = {"total": 0, "converted": 0, "skipped": 0}

        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if split_by == "line":
            chunks = [line.strip() for line in content.split('\n') if line.strip()]
        elif split_by == "paragraph":
            chunks = [p.strip() for p in content.split('\n\n') if p.strip()]
        else:
            chunks = [content]

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for chunk in chunks:
                stats["total"] += 1

                if len(chunk) < self.min_length or len(chunk) > self.max_length:
                    stats["skipped"] += 1
                    continue

                record = {"text": chunk}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["converted"] += 1

        return stats

    def code_to_pretrain(
        self,
        input_path: str,
        output_path: str,
        include_metadata: bool = True
    ) -> Dict:
        """
        将代码数据转换为预训练格式

        Args:
            input_path: 输入JSONL文件（来自code-to-corpus）
            output_path: 输出JSONL文件
            include_metadata: 是否包含元数据
        """
        stats = {"total": 0, "converted": 0, "skipped": 0}

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="转换代码"):
                stats["total"] += 1

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue

                # 获取代码文本
                code = data.get("text") or data.get("代码内容") or ""
                if not code:
                    stats["skipped"] += 1
                    continue

                if len(code) < self.min_length:
                    stats["skipped"] += 1
                    continue

                # 构建预训练文本
                if include_metadata:
                    metadata = []
                    if data.get("仓库名"):
                        metadata.append(f"Repository: {data['仓库名']}")
                    if data.get("文件名"):
                        metadata.append(f"File: {data['文件名']}")
                    if data.get("ext"):
                        lang = self._get_language(data["ext"])
                        metadata.append(f"Language: {lang}")

                    if metadata:
                        header = "# " + " | ".join(metadata) + "\n\n"
                        code = header + code

                record = {"text": code}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["converted"] += 1

        return stats

    def pdf_to_pretrain(
        self,
        input_path: str,
        output_path: str,
        split_by_section: bool = True
    ) -> Dict:
        """
        将PDF数据转换为预训练格式

        Args:
            input_path: 输入JSONL文件
            output_path: 输出JSONL文件
            split_by_section: 是否按章节分割
        """
        stats = {"total": 0, "converted": 0, "skipped": 0}

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="转换PDF"):
                stats["total"] += 1

                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue

                if split_by_section:
                    section = data.get("section", "")
                    content = data.get("content", "")

                    if section:
                        text = f"# {section}\n\n{content}"
                    else:
                        text = content
                else:
                    text = data.get("text", "")

                if not text or len(text) < self.min_length:
                    stats["skipped"] += 1
                    continue

                record = {"text": text}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["converted"] += 1

        return stats

    def merge_to_pretrain(
        self,
        input_paths: List[str],
        output_path: str,
        shuffle: bool = True
    ) -> Dict:
        """
        合并多个文件为预训练格式

        Args:
            input_paths: 输入文件列表
            output_path: 输出文件路径
            shuffle: 是否打乱顺序
        """
        import random

        stats = {"total": 0, "sources": {}}

        texts = []

        for input_path in input_paths:
            count = 0
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.strip())
                        text = data.get("text", "")
                        if text:
                            texts.append(text)
                            count += 1
                    except json.JSONDecodeError:
                        continue

            stats["sources"][input_path] = count
            stats["total"] += count

        if shuffle:
            random.seed(42)
            random.shuffle(texts)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for text in texts:
                record = {"text": text}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        return stats

    def _get_language(self, ext: str) -> str:
        """根据扩展名获取语言"""
        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".go": "Go",
            ".rs": "Rust",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".r": "R",
            ".jl": "Julia",
            ".sh": "Shell",
            ".sql": "SQL",
            ".html": "HTML",
            ".css": "CSS",
            ".json": "JSON",
            ".xml": "XML",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".md": "Markdown",
        }
        return lang_map.get(ext.lower(), ext)


def main():
    parser = argparse.ArgumentParser(description="预训练格式转换")
    parser.add_argument("--input", "-i", required=True, nargs='+', help="输入文件")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument("--type", choices=["text", "code", "pdf", "merge"], default="text", help="输入类型")
    parser.add_argument("--min-length", type=int, default=100, help="最小文本长度")
    parser.add_argument("--split-by", choices=["line", "paragraph", "document"], default="paragraph", help="分割方式")

    args = parser.parse_args()

    converter = PretrainFormatConverter(min_length=args.min_length)

    if args.type == "text":
        stats = converter.text_to_jsonl(args.input[0], args.output, args.split_by)
    elif args.type == "code":
        stats = converter.code_to_pretrain(args.input[0], args.output)
    elif args.type == "pdf":
        stats = converter.pdf_to_pretrain(args.input[0], args.output)
    else:
        stats = converter.merge_to_pretrain(args.input, args.output)

    print(f"转换完成:")
    print(f"  总数: {stats['total']}")
    print(f"  转换: {stats.get('converted', stats['total'])}")
    print(f"  跳过: {stats.get('skipped', 0)}")


if __name__ == "__main__":
    main()