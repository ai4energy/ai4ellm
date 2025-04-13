import os
import math
import torch.multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm  # 引入 tqdm 用于进度条显示

def is_file_processed(pdf_file_name, output_dir):
    """
    检查 PDF 文件是否已经被处理。
    """
    name_without_suff = os.path.splitext(os.path.basename(pdf_file_name))[0]
    md_file_path = os.path.join(output_dir, "markdown", f"{name_without_suff}.md")
    content_list_path = os.path.join(output_dir, "markdown", f"{name_without_suff}_content_list.json")
    return os.path.exists(md_file_path) and os.path.exists(content_list_path)

def process_pdf(pdf_file_name, output_dir="output"):
    """
    处理单个 PDF 文件，支持 OCR 模式和文本模式。
    """
    import torch
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    try:
        if is_file_processed(pdf_file_name, output_dir):
            return

        name_without_suff = os.path.splitext(os.path.basename(pdf_file_name))[0]

        local_image_dir = os.path.join(output_dir, "images", name_without_suff)
        local_md_dir = os.path.join(output_dir, "markdown")
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_file_name)
        ds = PymuDocDataset(pdf_bytes)

        parse_method = ds.classify()
        if parse_method == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", os.path.basename(local_image_dir))
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", os.path.basename(local_image_dir))

    except Exception:
        pass  # 忽略错误信息

def gather_pdf_files(input_folder):
    """
    扫描文件夹，返回所有 PDF 文件的绝对路径列表。
    """
    pdf_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_file_path = os.path.join(root, file)
                pdf_files.append(pdf_file_path)
    return pdf_files

def process_chunk(pdf_files_chunk, output_dir, gpu_id, pbar):
    """
    子进程函数：处理一块 PDF 文件，并更新进度条。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch

    for pdf_file in pdf_files_chunk:
        process_pdf(pdf_file, output_dir)
        pbar.update(1)  # 更新进度条

def process_folder(input_folder, output_dir="output", num_gpus=1):
    """
    主调度函数：使用多 GPU 并行处理 PDF 文件。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_pdf_files = gather_pdf_files(input_folder)
    total_files = len(all_pdf_files)
    if total_files == 0:
        print("No PDF files found.")
        return

    # 将 PDF 文件列表拆分为 num_gpus 份
    chunk_size = math.ceil(total_files / num_gpus)
    pdf_file_chunks = [
        all_pdf_files[i : i + chunk_size]
        for i in range(0, total_files, chunk_size)
    ]

    with tqdm(total=total_files, desc="PDF 文件处理进度") as pbar:
        # 创建多进程池并开始并行处理
        pool = Pool(processes=num_gpus)
        for gpu_id, chunk in enumerate(pdf_file_chunks):
            pool.apply_async(process_chunk, (chunk, output_dir, gpu_id, pbar))

        pool.close()
        pool.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    input_folder = r"E:\合并测试"
    output_dir = r"E:\合并测试\output"

    process_folder(input_folder, output_dir, num_gpus=1)
