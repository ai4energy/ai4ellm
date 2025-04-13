import os
import logging
from PyPDF2 import PdfReader, PdfWriter
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe
import json
from tqdm import tqdm


# ── ① 让 Detectron2 / fvcore 只输出 WARNING ─────────────────────────────
os.environ["DETECTRON2_LOG_LEVEL"] = "WARN"          # 对 detectron2 生效
logging.getLogger("fvcore").setLevel(logging.WARNING)
logging.getLogger("d2").setLevel(logging.WARNING)

# ── ② 让 loguru（magic_pdf 用的日志库）静音到文件 ───────────────────────
from loguru import logger

logger.remove()                                      # 移除默认的控制台 handler
logger.add("process.log", level="INFO", enqueue=True)  # 只写文件，不打到终端
# 如果还想把你自己的关键信息打印到控制台，可再加一条级别更高的 handler：
# logger.add(sys.stderr, level="WARNING")

# ── ③ 你自己的 logging 依旧写文件即可 ───────────────────────────────────
logging.basicConfig(
    filename="process.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)

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

def ocr_folder_to_markdown(input_folder):
    log_file = "process.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Script started.")
    logging.getLogger("magic_pdf").setLevel(logging.WARNING)
    pdf_files = gather_pdf_files(input_folder)

    input_folder  = input_folder
    output_folder = os.path.join(input_folder, "markdown")
    os.makedirs(output_folder, exist_ok=True)
    for pdf_file_name in tqdm(pdf_files, desc="OCR→Markdown", unit="pdf"):
        try:
            logging.info(f"Processing file: {pdf_file_name}")

            # Create a temporary PDF containing only the first 2000 pages
            temp_pdf_name = os.path.join(output_folder, "temp_first_2000_pages.pdf")
            with open(pdf_file_name, "rb") as infile:
                reader = PdfReader(infile)
                writer = PdfWriter()

                # Extract first 2000 pages (or fewer if total pages < 2000)
                for page in reader.pages[:2000]:
                    writer.add_page(page)

                # Write to a temporary file
                with open(temp_pdf_name, "wb") as outfile:
                    writer.write(outfile)

            ## Prepare environment for OCR
            local_image_dir = os.path.join(
                output_folder, "images", os.path.splitext(os.path.basename(pdf_file_name))[0]
            )
            local_md_dir = os.path.join(output_folder, "markdown")
            os.makedirs(local_image_dir, exist_ok=True)
            os.makedirs(local_md_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)
            image_dir = str(os.path.basename(local_image_dir))

            reader1 = FileBasedDataReader("")
            pdf_bytes = reader1.read(temp_pdf_name)  # Read the first 2000 pages from the temp file

            pipe = OCRPipe(pdf_bytes, [], image_writer)

            logging.info(f"Running OCR pipeline for {pdf_file_name}")
            pipe.pipe_classify()
            pipe.pipe_analyze()
            pipe.pipe_parse()

            pdf_info = pipe.pdf_mid_data["pdf_info"]
            logging.info(f"PDF info for {pdf_file_name}: {pdf_info}")

            md_content = pipe.pipe_mk_markdown(
                image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
            )

            # 写入markdown
            output_md_file = os.path.join(local_md_dir, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}.md")
            if isinstance(md_content, list):
                md_writer.write_string(output_md_file, "\n".join(md_content))
            else:
                md_writer.write_string(output_md_file, md_content)

            logging.info(f"Markdown file written: {output_md_file}")
            # 写入 JSON 内容
            output_json_file = os.path.join(local_md_dir,
                                            f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}.json")
            try:
                if isinstance(md_content, list):
                    json_content = {"content": md_content}
                else:
                    json_content = {"content": [md_content]}

                with open(output_json_file, "w", encoding="utf-8") as json_file:
                    json.dump(json_content, json_file, ensure_ascii=False, indent=4)

                logging.info(f"JSON 文件已写入：{output_json_file}")
            except Exception as json_error:
                logging.error(f"保存 JSON 文件时出错：{json_error}")

            # Clean up the temporary PDF file
            os.remove(temp_pdf_name)
            logging.info(f"Temporary file deleted: {temp_pdf_name}")

        except Exception as e:
            logging.error(f"Error processing file {pdf_file_name}: {e}")

    logging.info("All files processed.")
    print(f"Processing complete. Logs written to {log_file}")
