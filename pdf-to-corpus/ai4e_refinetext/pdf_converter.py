import os
from comtypes.client import CreateObject
from tqdm import tqdm  # 引入 tqdm 用于进度条显示


def convert_to_pdf(input_path, output_path, file_type):
    try:
        if file_type in ['.docx', '.doc']:
            word = CreateObject('Word.Application')
            doc = word.Documents.Open(input_path)
            doc.SaveAs(output_path, FileFormat=17)  # FileFormat=17 corresponds to PDF
            doc.Close()
            word.Quit()
        elif file_type in ['.pptx', '.ppt']:
            powerpoint = CreateObject('PowerPoint.Application')
            presentation = powerpoint.Presentations.Open(input_path)
            presentation.SaveAs(output_path, 32)  # 32 corresponds to PDF
            presentation.Close()
            powerpoint.Quit()
    except Exception:
        pass  # 忽略异常，避免输出错误信息


def batch_convert(folder_path):
    files_to_convert = []  # 用于存储需要转换的文件路径

    # 遍历文件夹，收集需要转换的文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.docx', '.doc', '.pptx', '.ppt']:
                files_to_convert.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for input_file in tqdm(files_to_convert, desc="PDF 转换进度"):
        output_file = os.path.splitext(input_file)[0] + '.pdf'
        ext = os.path.splitext(input_file)[1].lower()
        convert_to_pdf(input_file, output_file, ext)