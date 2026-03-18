"""
数据集工具模块

提供数据集划分、合并、采样等功能
"""

import os
import json
import random
import argparse
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm


def split_dataset(
    input_path: str,
    output_dir: str,
    test_size: float = 0.1,
    val_size: float = 0.0,
    seed: int = 42
) -> Dict:
    """
    划分数据集

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        val_size: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)

    # 读取数据
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取数据"):
            if line.strip():
                samples.append(json.loads(line.strip()))

    total = len(samples)
    print(f"总样本数: {total}")

    # 打乱顺序
    random.shuffle(samples)

    # 计算划分点
    n_test = int(total * test_size)
    n_val = int(total * val_size)
    n_train = total - n_test - n_val

    # 划分
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 写入文件
    stats = {"total": total, "train": n_train, "val": n_val, "test": n_test}

    with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for s in tqdm(train_samples, desc="写入训练集"):
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if val_samples:
        with open(os.path.join(output_dir, "val.jsonl"), 'w', encoding='utf-8') as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "test.jsonl"), 'w', encoding='utf-8') as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 写入划分信息
    with open(os.path.join(output_dir, "split_info.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "source": input_path,
            "seed": seed,
            "test_size": test_size,
            "val_size": val_size,
            "stats": stats
        }, f, indent=2)

    return stats


def merge_datasets(
    input_paths: List[str],
    output_path: str,
    deduplicate: bool = False,
    shuffle: bool = True,
    seed: int = 42
) -> Dict:
    """
    合并多个数据集

    Args:
        input_paths: 输入文件路径列表
        output_path: 输出文件路径
        deduplicate: 是否去重
        shuffle: 是否打乱
        seed: 随机种子
    """
    samples = []
    seen_hashes = set()
    stats = {"total": 0, "unique": 0, "duplicates": 0, "sources": {}}

    for input_path in input_paths:
        count = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"读取 {input_path}"):
                if not line.strip():
                    continue

                stats["total"] += 1

                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                if deduplicate:
                    text = sample.get("text", "")
                    h = hashlib.md5(text.encode()).hexdigest()
                    if h in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    seen_hashes.add(h)

                samples.append(sample)
                count += 1

        stats["sources"][input_path] = count

    stats["unique"] = len(samples)

    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return stats


def sample_dataset(
    input_path: str,
    output_path: str,
    n: Optional[int] = None,
    ratio: Optional[float] = None,
    seed: int = 42
) -> Dict:
    """
    采样数据集

    Args:
        input_path: 输入文件
        output_path: 输出文件
        n: 采样数量
        ratio: 采样比例
        seed: 随机种子
    """
    random.seed(seed)

    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))

    total = len(samples)

    if n is not None:
        sample_size = min(n, total)
    elif ratio is not None:
        sample_size = int(total * ratio)
    else:
        sample_size = total

    sampled = random.sample(samples, sample_size)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sampled:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return {"total": total, "sampled": sample_size}


def analyze_dataset(input_path: str) -> Dict:
    """分析数据集统计信息"""
    stats = {
        "total_samples": 0,
        "total_chars": 0,
        "total_words": 0,
        "lengths": [],
        "min_length": float('inf'),
        "max_length": 0,
    }

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            stats["total_samples"] += 1

            try:
                data = json.loads(line.strip())
                text = data.get("text", "")
                if not text and "messages" in data:
                    text = " ".join(m.get("content", "") for m in data["messages"])

                length = len(text)
                stats["lengths"].append(length)
                stats["total_chars"] += length
                stats["total_words"] += len(text.split())
                stats["min_length"] = min(stats["min_length"], length)
                stats["max_length"] = max(stats["max_length"], length)

            except json.JSONDecodeError:
                continue

    if stats["total_samples"] > 0:
        stats["avg_length"] = stats["total_chars"] / stats["total_samples"]
        stats["median_length"] = sorted(stats["lengths"])[len(stats["lengths"]) // 2]
    else:
        stats["min_length"] = 0
        stats["avg_length"] = 0
        stats["median_length"] = 0

    del stats["lengths"]
    return stats


def main():
    parser = argparse.ArgumentParser(description="数据集工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # split
    split_parser = subparsers.add_parser("split", help="划分数据集")
    split_parser.add_argument("--input", "-i", required=True)
    split_parser.add_argument("--output", "-o", required=True)
    split_parser.add_argument("--test-size", type=float, default=0.1)
    split_parser.add_argument("--val-size", type=float, default=0.0)
    split_parser.add_argument("--seed", type=int, default=42)

    # merge
    merge_parser = subparsers.add_parser("merge", help="合并数据集")
    merge_parser.add_argument("--input", "-i", nargs='+', required=True)
    merge_parser.add_argument("--output", "-o", required=True)
    merge_parser.add_argument("--deduplicate", action="store_true")
    merge_parser.add_argument("--seed", type=int, default=42)

    # sample
    sample_parser = subparsers.add_parser("sample", help="采样数据集")
    sample_parser.add_argument("--input", "-i", required=True)
    sample_parser.add_argument("--output", "-o", required=True)
    sample_parser.add_argument("--n", type=int)
    sample_parser.add_argument("--ratio", type=float)
    sample_parser.add_argument("--seed", type=int, default=42)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="分析数据集")
    analyze_parser.add_argument("--input", "-i", required=True)

    args = parser.parse_args()

    if args.command == "split":
        stats = split_dataset(args.input, args.output, args.test_size, args.val_size, args.seed)
        print(f"划分完成: 训练 {stats['train']}, 验证 {stats['val']}, 测试 {stats['test']}")

    elif args.command == "merge":
        stats = merge_datasets(args.input, args.output, args.deduplicate, seed=args.seed)
        print(f"合并完成: {stats['unique']}/{stats['total']} 条记录")

    elif args.command == "sample":
        stats = sample_dataset(args.input, args.output, args.n, args.ratio, args.seed)
        print(f"采样完成: {stats['sampled']}/{stats['total']} 条记录")

    elif args.command == "analyze":
        stats = analyze_dataset(args.input)
        print(f"数据集分析:")
        print(f"  样本数: {stats['total_samples']}")
        print(f"  总字符: {stats['total_chars']}")
        print(f"  平均长度: {stats['avg_length']:.1f}")
        print(f"  中位数长度: {stats['median_length']}")
        print(f"  最小长度: {stats['min_length']}")
        print(f"  最大长度: {stats['max_length']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()