"""
规则去重模块

对文本文件进行精确字符串去重（逐行比较）。
"""

from src.logger import get_logger

logger = get_logger()


def exact_deduplicate(
    input_file: str,
    output_file: str,
    case_sensitive: bool = False,
) -> tuple[int, int]:
    """
    对文本文件进行精确字符串去重。

    参数:
        input_file: 输入文本文件路径（每行一条文本）
        output_file: 输出去重后的文本文件路径
        case_sensitive: 是否区分大小写

    返回:
        (输入行数, 去重后行数)
    """
    seen = set()
    retained = []
    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").rstrip("\r")
            if not text:
                continue
            total += 1

            key = text if case_sensitive else text.lower()
            if key not in seen:
                seen.add(key)
                retained.append(text)

    with open(output_file, "w", encoding="utf-8") as f:
        for text in retained:
            f.write(text + "\n")

    logger.info(f"精确去重: {total} 行 → {len(retained)} 行")
    return total, len(retained)
