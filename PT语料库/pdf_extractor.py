import os
import math
import torch.multiprocessing as mp
from multiprocessing import Pool

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
    注意：不要在此函数外部 import torch 或 magic_pdf，而是在此函数内部 import，
    这样可以确保子进程已设置好 CUDA_VISIBLE_DEVICES 后再初始化相关库。
    """
    # 只有当我们真正执行到这里时，才 import torch 和 magic_pdf
    import torch
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod

    try:
        if is_file_processed(pdf_file_name, output_dir):
            print(f"Skipping already processed file: {pdf_file_name}")
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

        # 如果 doc_analyze 内部会用到 GPU（例如 model.to("cuda:0")），
        # 那么此时进程只看到一块卡，它的 index 就是 0。
        # 所以 "cuda:0" 会对应系统的物理 GPU #gpu_id。
        # 这里无需改动 doc_analyze 的代码，但如果 doc_analyze 里硬编码了 "cuda:0"
        # 作为全局物理卡，就需要改成对子进程而言的 "cuda:0"。

        # 推断并分类
        parse_method = ds.classify()
        if parse_method == SupportedPdfParseMethod.OCR:
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

def process_chunk(pdf_files_chunk, output_dir, gpu_id):
    """
    子进程函数：
    - 先设置子进程可见的 GPU
    - 再 import torch (以及 doc_analyze 相关的库) 或在 process_pdf 里再 import
    - 然后遍历要处理的 PDF 文件
    """

    # 第一步：告诉当前子进程，只能使用物理 GPU #gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 第二步：现在才可以 import torch（如需在此函数用 GPU）
    import torch

    print(f"\n[Process {os.getpid()}] I will use physical GPU={gpu_id}.")
    print(f"  Before any CUDA ops: torch.cuda.device_count() = {torch.cuda.device_count()}")

    # 如果 torch.cuda 还未初始化，这时 device_count() 可能就是 1，或者仍然是4（取决于实际情况）。
    # 要进一步确认可以尝试显式初始化，比如分配一个张量：

    if torch.cuda.is_available():
        try:
            test_x = torch.randn(1).cuda()
            print("  GPU test allocation successful.")
        except Exception as e:
            print(f"  GPU test allocation failed: {e}")

    # 第三步：处理该 chunk 下的所有 PDF 文件
    for pdf_file in pdf_files_chunk:
        process_pdf(pdf_file, output_dir)

    print(f"[Process {os.getpid()}] Done with GPU {gpu_id}.\n")

def process_folder(input_folder, output_dir="output", num_gpus=4):
    """
    主调度函数：
    1. 收集所有 PDF 文件
    2. 按 num_gpus 分成若干个块
    3. 多进程并行，每个进程分配到一个 GPU
    """
    os.makedirs(output_dir, exist_ok=True)

    all_pdf_files = gather_pdf_files(input_folder)
    total_files = len(all_pdf_files)
    if total_files == 0:
        print("No PDF files found.")
        return

    print(f"Found {total_files} PDF files in '{input_folder}'. Starting processing...")

    # 将 PDF 文件列表拆分为 num_gpus 份
    chunk_size = math.ceil(total_files / num_gpus)
    pdf_file_chunks = [
        all_pdf_files[i : i + chunk_size]
        for i in range(0, total_files, chunk_size)
    ]

    # 创建多进程池并开始并行处理
    pool = Pool(processes=num_gpus)
    for gpu_id, chunk in enumerate(pdf_file_chunks):
        pool.apply_async(process_chunk, (chunk, output_dir, gpu_id))

    pool.close()
    pool.join()

    print("All processes are done.")

if __name__ == "__main__":
    # 第一步：使用 'spawn' 模式启动子进程，以避免在 Linux 下使用 'fork' 导致的 CUDA 上下文继承问题
    mp.set_start_method('spawn', force=True)

    # 你要处理的 PDF 所在文件夹
    input_folder = "/home/yancong/第二次语料库/books/"
    output_dir   = "/home/yancong/第二次语料库/books/output"

    # 同时使用 4 张 GPU
    process_folder(input_folder, output_dir, num_gpus=4)
