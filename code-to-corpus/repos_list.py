import os
import json


folder = r"E:\daima\code\github-meta"


jsonfile = [file for file in os.listdir(folder) if file.endswith(".jsonl")]

file_name = 'repos_list.txt'
with open(file_name, 'w',encoding='utf-8') as out_file:
    for file in jsonfile:
        with open(os.path.join(folder, file), "r",encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    raw = json.loads(line)
                    data = json.loads(raw)
                    repo_id = data.get("id")
                    repo_url = data.get("html_url")
                    if repo_id and repo_url:
                        out_file.write(f"{repo_id},{repo_url}.git\n")
                except Exception as e:
                    print(f"跳过异常行：{e}")
                    continue