"""
语义去重模块

使用 Sentence-BERT 模型对文本进行语义去重。
支持中英文 embedding 模型，自动检测 GPU 可用性。
"""

import sys
import numpy as np
from tqdm import tqdm
import json

from src.logger import get_logger

logger = get_logger()


def semantic_deduplicate(
    input_file: str,
    output_file: str,
    model_name: str = "shibing624/text2vec-base-chinese",
    similarity_threshold: float = 0.9,
    batch_size: int = 1024,
    device: str = "auto",
    encoding: str = "utf-8",
) -> tuple[int, int]:
    """
    使用 Sentence-BERT 模型对文本进行语义去重。

    参数:
        input_file: 输入文本文件路径（每行一条文本）
        output_file: 输出去重后的文本文件路径
        model_name: SentenceTransformer 模型名称
        similarity_threshold: 相似度阈值（0~1 之间的余弦相似度）
        batch_size: 批处理大小
        device: 设备选择，"auto" 自动检测，"cuda" 或 "cpu"
        encoding: 文件编码

    返回:
        (输入行数, 去重后行数)
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        logger.error(
            "语义去重需要 sentence-transformers / torch，当前环境未安装。"
            "请安装相关依赖或关闭语义去重开关。"
        )
        return 0, 0

    # 自动检测设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"加载语义去重模型时出错: {e}")
        return 0, 0

    retained_embeddings = []
    line_count = 0
    unique_count = 0

    with open(input_file, "r", encoding=encoding) as fin, \
            open(output_file, "w", encoding=encoding) as fout:

        buffer_lines = []

        for line in tqdm(fin, desc="语义去重", unit="line"):
            text = line.strip()
            if not text:
                continue
            buffer_lines.append(text)

            # 批处理
            if len(buffer_lines) >= batch_size:
                unique_lines = _process_batch(
                    buffer_lines, model, retained_embeddings, similarity_threshold
                )
                for t, emb in unique_lines:
                    fout.write(t + "\n")
                    retained_embeddings.append(emb)
                unique_count += len(unique_lines)
                line_count += len(buffer_lines)
                buffer_lines = []

        # 处理剩余行
        if buffer_lines:
            unique_lines = _process_batch(
                buffer_lines, model, retained_embeddings, similarity_threshold
            )
            for t, emb in unique_lines:
                fout.write(t + "\n")
                retained_embeddings.append(emb)
            unique_count += len(unique_lines)
            line_count += len(buffer_lines)

    logger.info(f"语义去重完成: {line_count} 行 → {unique_count} 行")
    return line_count, unique_count


def _process_batch(
    lines: list[str],
    model,
    retained_embeddings: list,
    similarity_threshold: float,
) -> list[tuple[str, np.ndarray]]:
    """
    对一批文本计算嵌入，与已保留向量比较余弦相似度。

    参数:
        lines: 文本行列表
        model: SentenceTransformer 模型实例
        retained_embeddings: 已保留文本的嵌入列表
        similarity_threshold: 相似度阈值

    返回:
        保留的 (text, embedding) 列表
    """
    if not lines:
        return []

    embs = model.encode(lines, convert_to_numpy=True)
    embs = _normalize_embeddings(embs)

    results = []
    for i, line in enumerate(lines):
        emb = embs[i]
        candidates = retained_embeddings + [item[1] for item in results]
        if not candidates:
            results.append((line, emb))
            continue

        retained_matrix = np.vstack(candidates)
        # 余弦相似度（已归一化，点积即相似度）
        scores = np.dot(retained_matrix, emb)
        max_score = np.max(scores) if len(scores) > 0 else 0
        if max_score < similarity_threshold:
            results.append((line, emb))

    return results


def semantic_deduplicate_chunks(
    chunks: list[dict],
    model_name: str = "shibing624/text2vec-base-chinese",
    similarity_threshold: float = 0.9,
    batch_size: int = 128,
    device: str = "auto",
    report_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    对结构化 chunks 做语义去重，返回保留和过滤的 chunk。

    这个函数用于最终 JSON/JSONL 导出前，确保语义去重真正影响语料成品。
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        logger.error(
            "语义去重需要 sentence-transformers / torch，当前环境未安装。"
            "请安装相关依赖或关闭语义去重开关。"
        )
        return chunks, []

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"语义去重加载模型: {model_name} (device={device})")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"加载语义去重模型时出错: {e}")
        return chunks, []

    kept: list[dict] = []
    filtered: list[dict] = []
    retained_embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(chunks), batch_size), desc="Chunk 语义去重", unit="batch"):
        batch = chunks[start:start + batch_size]
        texts = [str(chunk.get("content", "")).strip() for chunk in batch]
        if not texts:
            continue

        embs = model.encode(texts, convert_to_numpy=True)
        embs = _normalize_embeddings(embs)

        for chunk, emb in zip(batch, embs):
            content = str(chunk.get("content", "")).strip()
            if not content:
                filtered.append(_with_semantic_reason(chunk, 1.0, "empty_content"))
                continue

            candidates = retained_embeddings
            if not candidates:
                kept.append(chunk)
                retained_embeddings.append(emb)
                continue

            scores = np.dot(np.vstack(candidates), emb)
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            if max_score < similarity_threshold:
                kept.append(chunk)
                retained_embeddings.append(emb)
            else:
                filtered.append(_with_semantic_reason(chunk, max_score, "semantic_duplicate"))

    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"语义去重报告保存失败 {report_path}: {e}")

    logger.info(f"Chunk 语义去重完成: {len(chunks)} → {len(kept)}（过滤 {len(filtered)}）")
    return kept, filtered


def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    """归一化 embedding，避免零向量导致 NaN。"""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def _with_semantic_reason(chunk: dict, score: float, reason: str) -> dict:
    item = dict(chunk)
    item["semantic_duplicate_score"] = round(score, 4)
    item["semantic_filter_reason"] = reason
    return item
