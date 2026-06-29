"""
语料质量评分模块

对文本 chunk 进行规则-based 质量评分，无需外部模型。
评分维度：可读性、连贯性、信息密度、噪声比。
支持按综合评分阈值过滤低质 chunk。
"""

import re
import json
import os
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger()


# ---- 停用词表 ----

# 中文常见停用词
CN_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "他", "她", "它", "们", "这", "这么", "那么", "为什么",
    "啊", "哦", "嗯", "呀", "吧", "呢", "嘛", "哈", "唉", "喂",
    "而", "之", "与", "或", "但", "如果", "因为", "所以", "虽然", "然而",
    "从", "向", "往", "到", "以", "用", "把", "被", "让", "给", "对", "对于",
    "关于", "通过", "根据", "由于", "为了", "对于", "至于",
    "其", "此", "该", "本", "该", "该", "该",
    "等", "等等", "各", "各个", "每", "各个",
    "之", "所", "之", "之",
    "如", "如果", "如果", "若", "如",
    "可", "可以", "能", "能够", "可能", "应该", "应当", "必须",
    "只", "仅", "仅仅", "才", "刚", "刚刚", "正在", "已经",
    "多", "少", "多少", "几", "几", "多少",
    "大", "小", "多", "少", "新", "旧", "长", "短", "高", "低",
    "来", "去", "回", "过", "进", "出", "上", "下", "前", "后",
    "时", "时候", "年", "月", "日", "天", "点", "分", "秒",
    "第", "初", "末", "底", "底", "底",
    "做", "作", "干", "弄", "搞", "打", "发", "起",
    "些", "这个", "那个", "哪些", "什么", "哪", "谁",
}

# 英文常见停用词
EN_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "it's", "this", "that", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "we", "us", "our", "ours", "they", "them",
    "their", "theirs", "what", "which", "who", "whom", "whose",
    "not", "no", "nor", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "into", "over", "after", "before",
    "as", "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "also", "here", "there", "again", "further", "once",
}

ALL_STOPWORDS = CN_STOPWORDS | EN_STOPWORDS


@dataclass
class QualityScore:
    """文本 chunk 的质量评分。"""
    overall: float            # 0-1 综合评分
    readability: float        # 可读性（句长分布、标点多样性）
    coherence: float          # 连贯性（相邻句词汇重叠）
    info_density: float       # 信息密度（非停用词占比）
    noise_ratio: float        # 噪声比（乱码/URL/特殊符号占比）


def score_chunk(content: str) -> QualityScore:
    """
    对单个文本 chunk 进行质量评分。

    评分维度：
    - 可读性：平均句长（30-80 字最优）、标点多样性
    - 连贯性：相邻句子间的实词重叠率
    - 信息密度：非停用词字符数 / 总字符数
    - 噪声比：乱码/URL/邮箱/特殊符号占比

    参数:
        content: 文本内容

    返回:
        QualityScore 实例
    """
    if not content or not content.strip():
        return QualityScore(overall=0.0, readability=0.0, coherence=0.0,
                            info_density=0.0, noise_ratio=1.0)

    readability = _score_readability(content)
    coherence = _score_coherence(content)
    info_density = _score_info_density(content)
    noise_ratio = _score_noise(content)

    # 综合评分：加权平均（噪声比取反）
    overall = (
        readability * 0.30
        + coherence * 0.25
        + info_density * 0.25
        + (1.0 - noise_ratio) * 0.20
    )

    return QualityScore(
        overall=round(overall, 4),
        readability=round(readability, 4),
        coherence=round(coherence, 4),
        info_density=round(info_density, 4),
        noise_ratio=round(noise_ratio, 4),
    )


def filter_chunks(
    chunks: list[dict],
    min_score: float = 0.3,
    max_noise_ratio: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    """
    按质量评分过滤 chunk。

    参数:
        chunks: chunk 列表（每个需包含 "content" 字段）
        min_score: 最低综合评分阈值
        max_noise_ratio: 最大可接受噪声比

    返回:
        (保留的 chunk 列表, 被过滤的 chunk 列表)
    """
    kept = []
    filtered = []

    for chunk in chunks:
        content = chunk.get("content", "")
        score = score_chunk(content)

        # 附加评分到 chunk
        chunk_with_score = dict(chunk)
        chunk_with_score["quality_score"] = score.overall
        chunk_with_score["quality"] = {
            "readability": score.readability,
            "coherence": score.coherence,
            "info_density": score.info_density,
            "noise_ratio": score.noise_ratio,
        }

        if score.overall >= min_score and score.noise_ratio <= max_noise_ratio:
            kept.append(chunk_with_score)
        else:
            filtered.append(chunk_with_score)

    return kept, filtered


def save_filtered_report(filtered: list[dict], output_path: str):
    """
    将被过滤的 chunk 写入报告文件，供人工审查。

    参数:
        filtered: 被过滤的 chunk 列表
        output_path: 输出文件路径
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report = []
        for c in filtered:
            report.append({
                "chunk_index": c.get("chunk_index"),
                "source_file": c.get("source_file"),
                "section_title": c.get("section_title"),
                "quality_score": c.get("quality_score"),
                "quality": c.get("quality"),
                "char_count": c.get("char_count"),
                "content_preview": c.get("content", "")[:100] + "...",
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"质量过滤报告已保存: {output_path} ({len(filtered)} 个 chunk 被过滤)")
    except Exception as e:
        logger.error(f"保存质量过滤报告失败: {e}")


# ---- 各维度评分实现 ----

def _score_readability(content: str) -> float:
    """
    可读性评分。

    - 平均句长在 30-80 字符区间得满分
    - 标点多样性：使用多种标点符号（，。！？；）得分更高
    """
    sentences = _split_sentences(content)
    if not sentences:
        return 0.0

    # 平均句长评分
    lengths = [len(s) for s in sentences]
    avg_len = sum(lengths) / len(lengths)

    # 30-80 字为最优区间，使用钟形曲线
    if 30 <= avg_len <= 80:
        len_score = 1.0
    elif avg_len < 10:
        len_score = 0.2  # 太短
    elif avg_len > 200:
        len_score = 0.3  # 太长
    else:
        # 线性衰减
        if avg_len < 30:
            len_score = 0.2 + 0.8 * (avg_len - 10) / 20
        else:
            len_score = 1.0 - 0.7 * min((avg_len - 80) / 120, 1.0)

    # 标点多样性
    punct_types = {"，", "。", "！", "？", "；", "：", ",", ".", "!", "?", ";"}
    found_puncts = sum(1 for p in punct_types if p in content)
    punct_score = min(found_puncts / 5, 1.0)  # 出现 5 种标点即满分

    return 0.6 * len_score + 0.4 * punct_score


def _score_coherence(content: str) -> float:
    """
    连贯性评分：相邻句子间的实词重叠率。

    如果相邻两句共享 ≥2 个实词（非停用词），认为连贯。
    """
    sentences = _split_sentences(content)
    if len(sentences) < 2:
        return 0.7  # 单句给中等偏上评分

    coherent_pairs = 0
    total_pairs = 0

    for i in range(len(sentences) - 1):
        words_a = _extract_content_words(sentences[i])
        words_b = _extract_content_words(sentences[i + 1])

        if not words_a or not words_b:
            continue

        total_pairs += 1
        overlap = len(words_a & words_b)
        if overlap >= 2:
            coherent_pairs += 1

    if total_pairs == 0:
        return 0.5

    return coherent_pairs / total_pairs


def _score_info_density(content: str) -> float:
    """
    信息密度评分：非停用词字符数占总字符数的比例。

    中文按字符计算，英文按单词计算。
    """
    if not content.strip():
        return 0.0

    total_chars = len(content)
    if total_chars == 0:
        return 0.0

    # 提取非停用词字符
    content_words = _extract_content_words(content)

    # 计算实词总字符数
    content_char_count = 0
    for word in content_words:
        content_char_count += len(word)

    # 密度 = 实词字符 / 总字符
    density = content_char_count / total_chars

    # 映射到 0-1：0.3-0.8 为正常区间
    if density >= 0.8:
        return 1.0
    elif density <= 0.3:
        return 0.2
    else:
        return 0.2 + 0.8 * (density - 0.3) / 0.5


def _score_noise(content: str) -> float:
    """
    噪声评分：乱码/URL/邮箱/特殊符号占比（越低越好）。

    返回 0-1，0 = 无噪声，1 = 全是噪声。
    """
    if not content.strip():
        return 1.0

    total_chars = len(content)
    noise_chars = 0

    # URL
    url_matches = re.findall(r'https?://\S+', content)
    noise_chars += sum(len(m) for m in url_matches)

    # 邮箱
    email_matches = re.findall(r'\S+@\S+\.\S+', content)
    noise_chars += sum(len(m) for m in email_matches)

    # 乱码字符（非中英文、非标点、非数字的非常用字符）
    garble_pattern = re.compile(r'[^\w\s一-鿿　-〿＀-￯，。！？；：、""''（）【】《》\[\]{}()\-+*/=<>]')
    garble_chars = garble_pattern.findall(content)
    noise_chars += len(garble_chars)

    # 连续重复字符（如 "aaaaa"、"哈哈哈哈"）
    repeat_pattern = re.compile(r'(.)\1{4,}')
    for m in repeat_pattern.finditer(content):
        noise_chars += len(m.group())

    return min(noise_chars / total_chars, 1.0)


# ---- 辅助函数 ----

def _split_sentences(text: str) -> list[str]:
    """按句子边界拆分。"""
    pattern = r'(?<=[。！？.!?;；])\s*'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_content_words(text: str) -> set[str]:
    """
    提取文本中的实词（去除停用词）。
    中文按单字符处理，英文按单词处理。
    """
    words = set()

    # 英文单词
    en_words = re.findall(r'[a-zA-Z]+', text.lower())
    for w in en_words:
        if w not in EN_STOPWORDS and len(w) > 1:
            words.add(w)

    # 中文字符（排除停用词和标点）
    cn_chars = re.findall(r'[一-鿿]', text)
    for c in cn_chars:
        if c not in CN_STOPWORDS:
            words.add(c)

    return words
