import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

def is_file_processed(pdf_file_name, output_dir):
    """
    检查 PDF 文件是否已经被处理。
    Args:
        pdf_file_name (str): PDF 文件路径。
        output_dir (str): 输出文件夹路径。
    Returns:
        bool: 如果已处理则返回 True，否则返回 False。
    """
    name_without_suff = os.path.splitext(os.path.basename(pdf_file_name))[0]
    md_file_path = os.path.join(output_dir, "markdown", f"{name_without_suff}.md")
    content_list_path = os.path.join(output_dir, "markdown", f"{name_without_suff}_content_list.json")

    return os.path.exists(md_file_path) and os.path.exists(content_list_path)

def process_pdf(pdf_file_name, output_dir="output"):
    """
    处理单个 PDF 文件，支持 OCR 模式和文本模式。
    Args:
        pdf_file_name (str): 输入的 PDF 文件路径。
        output_dir (str): 输出文件夹路径。
    """
    try:
        # 检查文件是否已经被处理
        if is_file_processed(pdf_file_name, output_dir):
            print(f"Skipping already processed file: {pdf_file_name}")
            return

        # 获取文件名（无后缀）
        name_without_suff = os.path.splitext(os.path.basename(pdf_file_name))[0]

        # 创建输出目录
        local_image_dir = os.path.join(output_dir, "images", os.path.splitext(os.path.basename(pdf_file_name))[0])
        local_md_dir = os.path.join(output_dir, "markdown")
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)

        # 初始化数据写入器
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # 读取 PDF 文件的字节流
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_file_name)

        # 创建数据集实例
        ds = PymuDocDataset(pdf_bytes)

        # 推断并分类
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # 输出结果文件
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", os.path.basename(local_image_dir))
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", os.path.basename(local_image_dir))

        print(f"Successfully processed: {pdf_file_name}")

    except Exception as e:
        print(f"Error processing {pdf_file_name}: {e}")

def process_folder(input_folder, output_dir="output"):
    """
    批量处理文件夹中的所有 PDF 文件。
    Args:
        input_folder (str): 输入文件夹路径。
        output_dir (str): 输出文件夹路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    skipped_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_file_path = os.path.join(root, file)
                if is_file_processed(pdf_file_path, output_dir):
                    skipped_files.append(pdf_file_path)
                process_pdf(pdf_file_path, output_dir)
    
    if skipped_files:
        print("\nSkipped files:")
        for skipped_file in skipped_files:
            print(skipped_file)

if __name__ == "__main__":
    # 输入文件夹和输出文件夹路径
    input_folder = "/home/yancong/xin-llm/第二次拷贝/books"
    output_dir = "/home/yancong/xin-llm/第二次拷贝/books/output"

    # 批量处理整个文件夹中的 PDF 文件
    process_folder(input_folder, output_dir)
