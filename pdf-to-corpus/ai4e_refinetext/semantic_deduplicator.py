import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def semantic_deduplicate(input_file, output_file, model_name='sentence-transformers/all-MiniLM-L6-v2',
                         similarity_threshold=0.9, batch_size=1024, encoding='utf-8'):
    """
    使用Sentence-BERT模型对文本进行语义去重的基础示例。

    参数说明：
    - input_file: 输入文本文件，每行一条文本记录
    - output_file: 输出去重后的文本文件
    - model_name: SentenceTransformer可加载的模型名称
    - similarity_threshold: 相似度阈值（0~1之间的余弦相似度）
    - batch_size: 批处理大小，每次计算多少行的嵌入
    - encoding: 文件编码
    """
    try:
        # 加载Sentence-BERT模型
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"加载模型时出错: {e}", file=sys.stderr)
        return

    # 用于保存已保留行的向量
    retained_embeddings = []
    retained_texts = []

    with open(input_file, 'r', encoding=encoding) as fin, \
            open(output_file, 'w', encoding=encoding) as fout:

        buffer_lines = []
        line_count = 0
        unique_count = 0

        for line in tqdm(fin, desc="Processing", unit="line"):
            text = line.strip()
            if not text:
                continue
            buffer_lines.append(text)

            # 批处理，当积累到一定数量时才计算嵌入
            if len(buffer_lines) >= batch_size:
                unique_lines = process_batch(buffer_lines, model, retained_embeddings, retained_texts,
                                             similarity_threshold)
                # 将去重后保留的行写入文件，并更新retained_embeddings
                for t, emb in unique_lines:
                    fout.write(t + "\n")
                    retained_embeddings.append(emb)
                    retained_texts.append(t)
                unique_count += len(unique_lines)
                line_count += len(buffer_lines)
                buffer_lines = []

        # 处理剩余行
        if buffer_lines:
            unique_lines = process_batch(buffer_lines, model, retained_embeddings, retained_texts, similarity_threshold)
            for t, emb in unique_lines:
                fout.write(t + "\n")
                retained_embeddings.append(emb)
                retained_texts.append(t)
            unique_count += len(unique_lines)
            line_count += len(buffer_lines)

        print(f"处理完成！共处理 {line_count} 行，最终输出 {unique_count} 行。", file=sys.stderr)


def process_batch(lines, model, retained_embeddings, retained_texts, similarity_threshold):
    """
    对一批文本行进行嵌入计算，然后对每行查找与已保留文本的最高相似度，决定是否保留。
    返回 (text, embedding) 的列表。
    """
    if len(lines) == 0:
        return []

    embs = model.encode(lines, convert_to_numpy=True)
    # 归一化，使向量长度为1
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    results = []
    # 若尚无已保留文本，全部保留
    if len(retained_embeddings) == 0:
        for i, line in enumerate(lines):
            results.append((line, embs[i]))
        return results

    # 将已保留的嵌入合并为矩阵
    retained_matrix = np.vstack(retained_embeddings) if len(retained_embeddings) > 0 else None
    for i, line in enumerate(lines):
        emb = embs[i]
        # 计算与已保留嵌入的相似度（点积即为余弦相似度，因为已归一化）
        scores = np.dot(retained_matrix, emb)
        max_score = np.max(scores) if len(scores) > 0 else 0
        if max_score < similarity_threshold:
            # 没有相似度超过阈值，保留该文本
            results.append((line, emb))
        # 否则不保留

    return results


if __name__ == '__main__':
    input_path = r"G:\yuliaoku\1-quan\未去重.txt"
    output_path = r"G:\yuliaoku\1-quan\去重.txt"
    semantic_deduplicate(input_path, output_path, similarity_threshold=0.9)
