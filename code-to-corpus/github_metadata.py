import requests
import time

# ====== 配置区域 ======
GITHUB_TOKEN = "********"
INPUT_FILE = "repos.txt"          # 每行一个仓库名，如 AnyMOD.jl.git
OUTPUT_FILE = "repos_list.txt"    # 输出：id, url
# ======================

headers = {
    "Accept": "application/vnd.github+json"
}
if GITHUB_TOKEN:
    headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

def search_repo(repo_name):
    query = repo_name.replace(".git", "")
    url = f"https://api.github.com/search/repositories?q={query}&per_page=1"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            if items:
                repo = items[0]
                return repo["id"], repo["clone_url"]
            else:
                print(f"[未找到] {query}")
        else:
            print(f"[错误] 查询失败 {query}: {resp.status_code}")
    except Exception as e:
        print(f"[异常] {query}: {e}")
    return None, None

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    repos = [line.strip() for line in f if line.strip()]

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for repo in repos:
        print(f"正在搜索：{repo}")
        repo_id, clone_url = search_repo(repo)
        if repo_id and clone_url:
            out.write(f"{repo_id}, {clone_url}\n")
        time.sleep(2)  # 避免触发 rate limit