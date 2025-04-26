def replace_spaces_with_newlines(input_file_path, output_file_path):
    try:
        # 打开输入文件进行读取
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # 读取文件内容
            content = input_file.read()
            # 将空格替换为换行符
            new_content = content.replace(' ', '\n')

        # 打开输出文件进行写入
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # 将替换后的内容写入输出文件
            output_file.write(new_content)
        print(f"已成功将 {input_file_path} 中的空格替换为换行符，并保存到 {output_file_path}。")
    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    # 输入文件路径，可根据实际情况修改
    input_file_path = 'repos1.txt'
    # 输出文件路径，可根据实际情况修改
    output_file_path = 'repos.txt'
    replace_spaces_with_newlines(input_file_path, output_file_path)