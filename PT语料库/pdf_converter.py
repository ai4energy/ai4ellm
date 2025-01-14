import os
from comtypes.client import CreateObject

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
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def batch_convert(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.docx', '.doc', '.pptx', '.ppt']:
                input_file = os.path.join(root, file)
                output_file = os.path.splitext(input_file)[0] + '.pdf'
                print(f"Converting: {input_file} -> {output_file}")
                convert_to_pdf(input_file, output_file, ext)

# 替换为你的文件夹路径
folder_path = r"G:\yuliaoku\新能源减排服务"
batch_convert(folder_path)