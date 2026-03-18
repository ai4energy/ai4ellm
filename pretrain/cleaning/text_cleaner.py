"""
文本清洗模块

支持多种清洗功能：
- 去除HTML标签
- 去除特殊字符
- 去除乱码
- 去除重复行
- 去除页眉页脚
- 修复编码问题
"""

import os
import re
import json
import argparse
from typing import List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class CleaningStats:
    """清洗统计"""
    original_chars: int = 0
    cleaned_chars: int = 0
    original_lines: int = 0
    cleaned_lines: int = 0
    removed_lines: int = 0


class TextCleaner:
    """文本清洗器"""

    def __init__(self):
        self.cleaners: List[Callable] = []

    def add_cleaner(self, cleaner: Callable):
        """添加清洗函数"""
        self.cleaners.append(cleaner)
        return self

    def clean(self, text: str) -> str:
        """应用所有清洗函数"""
        for cleaner in self.cleaners:
            text = cleaner(text)
        return text

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """去除HTML标签"""
        return re.sub(r'<[^>]+>', '', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """去除URL"""
        return re.sub(r'https?://\S+', '', text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """去除邮箱"""
        return re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)

    @staticmethod
    def remove_special_chars(text: str, keep_newlines: bool = True) -> str:
        """去除特殊字符"""
        if keep_newlines:
            # 保留换行符，移除其他控制字符
            return re.sub(r'[^\S\n]+', ' ', text)
        else:
            return re.sub(r'[^\S]+', ' ', text)

    @staticmethod
    def remove_garbled_text(text: str) -> str:
        """去除乱码"""
        # 移除替换字符
        text = text.replace('', '')

        # 移除不可打印字符（保留中文、英文、数字、标点）
        text = re.sub(r'[^\x20-\x7E\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF\n\r\t]', '', text)

        return text

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """去除页码"""
        # 移除常见页码格式
        patterns = [
            r'^\s*第\s*\d+\s*页\s*$',  # 第N页
            r'^\s*\d+\s*/\s*\d+\s*$',  # N/M
            r'^\s*-?\s*\d+\s*-?\s*$',  # -N- 或 N
            r'^\s*Page\s*\d+\s*$',     # Page N
        ]
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            is_page_num = any(re.match(p, line) for p in patterns)
            if not is_page_num:
                cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def remove_headers_footers(text: str, header_patterns: List[str] = None) -> str:
        """去除页眉页脚"""
        if header_patterns is None:
            header_patterns = [
                r'Copyright\s+',
                r'All rights reserved',
                r'版权所有',
            ]

        lines = text.split('\n')
        cleaned = []
        for line in lines:
            should_remove = any(re.search(p, line, re.IGNORECASE) for p in header_patterns)
            if not should_remove:
                cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """规范化空白字符"""
        # 将多个空格合并为一个
        text = re.sub(r'[^\S\n]+', ' ', text)
        # 将多个换行合并为两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """去除空行"""
        lines = text.split('\n')
        return '\n'.join(line for line in lines if line.strip())

    @staticmethod
    def remove_short_lines(text: str, min_length: int = 5) -> str:
        """去除过短的行"""
        lines = text.split('\n')
        return '\n'.join(line for line in lines if len(line.strip()) >= min_length)

    @staticmethod
    def fix_encoding(text: str) -> str:
        """修复编码问题"""
        # 常见编码错误修复
        replacements = {
            '锟斤拷': '',
            '烫烫烫': '',
            '屯屯屯': '',
            '\ufffd': '',  # 替换字符
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text


class MarkdownCleaner(TextCleaner):
    """Markdown清洗器"""

    @staticmethod
    def remove_images(text: str) -> str:
        """去除图片"""
        return re.sub(r'!\[.*?\]\(.*?\)', '', text)

    @staticmethod
    def remove_links(text: str, keep_text: bool = True) -> str:
        """去除链接"""
        if keep_text:
            # 保留链接文本
            return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        else:
            return re.sub(r'\[.*?\]\(.*?\)', '', text)

    @staticmethod
    def remove_code_blocks(text: str) -> str:
        """去除代码块"""
        return re.sub(r'```[\s\S]*?```', '', text)

    @staticmethod
    def remove_toc(text: str) -> str:
        """去除目录"""
        # 简单的目录检测
        lines = text.split('\n')
        in_toc = False
        cleaned = []

        for line in lines:
            if re.match(r'^#\s*(目录|Contents|Table of Contents)', line, re.IGNORECASE):
                in_toc = True
                continue

            if in_toc:
                # 目录通常以链接或数字开头
                if re.match(r'^\s*[\d\.\-]+\s+', line) or re.match(r'^\s*\[', line):
                    continue
                else:
                    in_toc = False

            cleaned.append(line)

        return '\n'.join(cleaned)


class CodeCleaner(TextCleaner):
    """代码清洗器"""

    @staticmethod
    def remove_comments(text: str, language: str = "python") -> str:
        """去除注释"""
        if language == "python":
            # 去除单行注释
            text = re.sub(r'#.*$', '', text, flags=re.MULTILINE)
            # 去除多行注释
            text = re.sub(r'"""[\s\S]*?"""', '', text)
            text = re.sub(r"'''[\s\S]*?'''", '', text)
        elif language in ["java", "javascript", "c", "cpp"]:
            # 去除单行注释
            text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
            # 去除多行注释
            text = re.sub(r'/\*[\s\S]*?\*/', '', text)
        return text

    @staticmethod
    def remove_imports(text: str, language: str = "python") -> str:
        """去除导入语句"""
        if language == "python":
            text = re.sub(r'^import .+$', '', text, flags=re.MULTILINE)
            text = re.sub(r'^from .+ import .+$', '', text, flags=re.MULTILINE)
        return text


def clean_file(
    input_path: str,
    output_path: str,
    cleaner: TextCleaner = None,
    min_length: int = 50
) -> CleaningStats:
    """清洗单个文件"""
    if cleaner is None:
        cleaner = TextCleaner()
        cleaner.add_cleaner(TextCleaner.fix_encoding)
        cleaner.add_cleaner(TextCleaner.remove_garbled_text)
        cleaner.add_cleaner(TextCleaner.remove_html_tags)
        cleaner.add_cleaner(TextCleaner.remove_urls)
        cleaner.add_cleaner(TextCleaner.normalize_whitespace)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    stats = CleaningStats()
    stats.original_chars = len(text)
    stats.original_lines = len(text.split('\n'))

    cleaned = cleaner.clean(text)
    stats.cleaned_chars = len(cleaned)
    stats.cleaned_lines = len(cleaned.split('\n'))
    stats.removed_lines = stats.original_lines - stats.cleaned_lines

    # 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    return stats


def clean_folder(
    input_folder: str,
    output_folder: str,
    cleaner: TextCleaner = None,
    extensions: List[str] = None
) -> Dict:
    """清洗整个文件夹"""
    if extensions is None:
        extensions = [".txt", ".md"]

    if cleaner is None:
        cleaner = TextCleaner()
        cleaner.add_cleaner(TextCleaner.fix_encoding)
        cleaner.add_cleaner(TextCleaner.remove_garbled_text)
        cleaner.add_cleaner(TextCleaner.normalize_whitespace)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    total_stats = CleaningStats()
    files_processed = 0

    for ext in extensions:
        for file in tqdm(list(input_path.rglob(f"*{ext}")), desc=f"清洗 {ext} 文件"):
            rel_path = file.relative_to(input_path)
            out_file = output_path / rel_path

            stats = clean_file(str(file), str(out_file), cleaner)
            total_stats.original_chars += stats.original_chars
            total_stats.cleaned_chars += stats.cleaned_chars
            total_stats.original_lines += stats.original_lines
            total_stats.cleaned_lines += stats.cleaned_lines
            total_stats.removed_lines += stats.removed_lines
            files_processed += 1

    return {
        "files_processed": files_processed,
        "original_chars": total_stats.original_chars,
        "cleaned_chars": total_stats.cleaned_chars,
        "removed_chars": total_stats.original_chars - total_stats.cleaned_chars,
        "removed_lines": total_stats.removed_lines
    }


def main():
    parser = argparse.ArgumentParser(description="文本清洗")
    parser.add_argument("--input", "-i", required=True, help="输入文件或文件夹")
    parser.add_argument("--output", "-o", required=True, help="输出路径")
    parser.add_argument("--type", choices=["text", "markdown", "code"], default="text", help="清洗类型")
    parser.add_argument("--min-length", type=int, default=50, help="最小行长度")

    args = parser.parse_args()

    # 选择清洗器
    if args.type == "markdown":
        cleaner = MarkdownCleaner()
        cleaner.add_cleaner(MarkdownCleaner.remove_images)
        cleaner.add_cleaner(MarkdownCleaner.remove_links)
        cleaner.add_cleaner(MarkdownCleaner.remove_toc)
        cleaner.add_cleaner(TextCleaner.normalize_whitespace)
    elif args.type == "code":
        cleaner = CodeCleaner()
    else:
        cleaner = TextCleaner()
        cleaner.add_cleaner(TextCleaner.fix_encoding)
        cleaner.add_cleaner(TextCleaner.remove_garbled_text)
        cleaner.add_cleaner(TextCleaner.normalize_whitespace)

    if os.path.isfile(args.input):
        stats = clean_file(args.input, args.output, cleaner)
        print(f"清洗完成:")
        print(f"  原始字符: {stats.original_chars}")
        print(f"  清洗后: {stats.cleaned_chars}")
        print(f"  移除行数: {stats.removed_lines}")
    else:
        stats = clean_folder(args.input, args.output, cleaner)
        print(f"清洗完成:")
        print(f"  处理文件: {stats['files_processed']}")
        print(f"  原始字符: {stats['original_chars']}")
        print(f"  清洗后: {stats['cleaned_chars']}")


if __name__ == "__main__":
    main()