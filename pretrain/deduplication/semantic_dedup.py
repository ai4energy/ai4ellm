"""
语义去重模块

支持多种去重方法：
- 精确去重（MD5哈希）
- 近似去重（MinHash LSH）
- 语义去重（Sentence-BERT）
"""

import os
import json
import hashlib
import argparse
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DedupStats:
    """去重统计"""
    total: int = 0
    unique: int = 0
    duplicates: int = 0


class ExactDeduplicator:
    """精确去重器（基于哈希）"""

    def __init__(self, hash_field: str = "text"):
        """
        Args:
            hash_field: 用于计算哈希的字段
        """
        self.hash_field = hash_field
        self.seen_hashes: Set[str] = set()

    def compute_hash(self, text: str) -> str:
        """计算文本哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """检查是否重复"""
        h = self.compute_hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def deduplicate_file(self, input_path: str, output_path: str) -> DedupStats:
        """去重文件"""
        stats = DedupStats()

        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:

            for line in tqdm(f_in, desc="精确去重"):
                stats.total += 1

                try:
                    data = json.loads(line.strip())
                    text = data.get(self.hash_field, "")
                except json.JSONDecodeError:
                    continue

                if not self.is_duplicate(text):
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    stats.unique += 1
                else:
                    stats.duplicates += 1

        return stats


class MinHashDeduplicator:
    """近似去重器（MinHash LSH）"""

    def __init__(self, num_perm: int = 128, threshold: float = 0.9, ngram: int = 3):
        """
        Args:
            num_perm: 排列数量
            threshold: 相似度阈值
            ngram: n-gram大小
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.ngram = ngram
        self._available = None

    def _check_availability(self) -> bool:
        if self._available is None:
            try:
                from datasketch import MinHash, MinHashLSH
                self._available = True
            except ImportError:
                logger.warning("datasketch未安装，请运行: pip install datasketch")
                self._available = False
        return self._available

    def _get_tokens(self, text: str) -> List[str]:
        """获取n-gram tokens"""
        tokens = []
        for i in range(len(text) - self.ngram + 1):
            tokens.append(text[i:i + self.ngram])
        return tokens

    def deduplicate_file(self, input_path: str, output_path: str) -> DedupStats:
        """去重文件"""
        if not self._check_availability():
            logger.error("MinHash去重不可用")
            return DedupStats()

        from datasketch import MinHash, MinHashLSH

        stats = DedupStats()

        # 创建LSH索引
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # 第一遍：建立索引
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        minhashes = []
        for i, line in enumerate(tqdm(lines, desc="建立索引")):
            try:
                data = json.loads(line.strip())
                text = data.get("text", "")
            except json.JSONDecodeError:
                minhashes.append(None)
                continue

            mh = MinHash(num_perm=self.num_perm)
            for token in self._get_tokens(text):
                mh.update(token.encode('utf-8'))

            minhashes.append(mh)
            lsh.insert(f"doc_{i}", mh)

        # 第二遍：检查重复
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, (line, mh) in enumerate(zip(lines, minhashes)):
                stats.total += 1

                if mh is None:
                    continue

                # 查询相似文档
                result = lsh.query(mh)

                # 如果第一个匹配的是自己，说明是唯一的
                if result and result[0] == f"doc_{i}":
                    f_out.write(json.dumps(json.loads(line.strip()), ensure_ascii=False) + "\n")
                    stats.unique += 1
                else:
                    stats.duplicates += 1

        return stats


class SemanticDeduplicator:
    """语义去重器（基于Sentence-BERT）"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.9,
        batch_size: int = 256
    ):
        """
        Args:
            model_name: 模型名称
            threshold: 相似度阈值
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model

    def deduplicate_file(
        self,
        input_path: str,
        output_path: str,
        text_field: str = "text"
    ) -> DedupStats:
        """去重文件"""
        import numpy as np

        model = self._load_model()
        stats = DedupStats()

        # 读取所有数据
        texts = []
        data_list = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line.strip())
                    text = data.get(text_field, "")
                    if text:
                        texts.append(text)
                        data_list.append(data)
                except json.JSONDecodeError:
                    continue

        stats.total = len(texts)

        # 计算嵌入
        logger.info("计算文本嵌入...")
        embeddings = model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)

        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        # 去重
        logger.info("执行语义去重...")
        retained_indices = []

        for i in tqdm(range(len(texts)), desc="去重中"):
            is_duplicate = False

            for j in retained_indices:
                similarity = np.dot(embeddings[i], embeddings[j])
                if similarity > self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                retained_indices.append(i)
                stats.unique += 1
            else:
                stats.duplicates += 1

        # 写入结果
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for idx in retained_indices:
                f_out.write(json.dumps(data_list[idx], ensure_ascii=False) + "\n")

        return stats


class HybridDeduplicator:
    """混合去重器（先精确去重，再语义去重）"""

    def __init__(self, semantic_threshold: float = 0.9):
        self.exact_dedup = ExactDeduplicator()
        self.semantic_dedup = SemanticDeduplicator(threshold=semantic_threshold)

    def deduplicate_file(
        self,
        input_path: str,
        output_path: str,
        use_semantic: bool = True
    ) -> DedupStats:
        """混合去重"""
        # 第一步：精确去重
        temp_path = output_path + ".temp"
        stats1 = self.exact_dedup.deduplicate_file(input_path, temp_path)

        if not use_semantic:
            os.rename(temp_path, output_path)
            return stats1

        # 第二步：语义去重
        stats2 = self.semantic_dedup.deduplicate_file(temp_path, output_path)

        # 清理临时文件
        os.remove(temp_path)

        # 合并统计
        return DedupStats(
            total=stats1.total,
            unique=stats2.unique,
            duplicates=stats1.total - stats2.unique
        )


def main():
    parser = argparse.ArgumentParser(description="文本去重")
    parser.add_argument("--input", "-i", required=True, help="输入文件")
    parser.add_argument("--output", "-o", required=True, help="输出文件")
    parser.add_argument(
        "--method", "-m",
        choices=["exact", "minhash", "semantic", "hybrid"],
        default="hybrid",
        help="去重方法"
    )
    parser.add_argument("--threshold", type=float, default=0.9, help="相似度阈值")
    parser.add_argument("--text-field", default="text", help="文本字段名")

    args = parser.parse_args()

    if args.method == "exact":
        dedup = ExactDeduplicator(hash_field=args.text_field)
    elif args.method == "minhash":
        dedup = MinHashDeduplicator(threshold=args.threshold)
    elif args.method == "semantic":
        dedup = SemanticDeduplicator(threshold=args.threshold)
    else:
        dedup = HybridDeduplicator(semantic_threshold=args.threshold)

    stats = dedup.deduplicate_file(args.input, args.output)

    print(f"去重完成:")
    print(f"  总数: {stats.total}")
    print(f"  唯一: {stats.unique}")
    print(f"  重复: {stats.duplicates}")
    print(f"  去重率: {stats.duplicates/stats.total*100:.1f}%")


if __name__ == "__main__":
    main()