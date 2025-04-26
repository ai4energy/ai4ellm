# ai4ellm
## 代码语料爬取流程，主要参考MNBVC超大规模中文语料集中的GitHub爬取，项目地址如下：https://github.com/esbatmop/MNBVC


## 📄 1. meta信息获取 

### 1.1 自动获取（auto-metedata.py）

自动爬取meta信息，需要配置自己的github_tokens,除此外正常运行，运行结束后需要提取关键信息,直接运行repos_list.py即可

```python
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
```

### 1.2 爬取指定库的代码（replace_spaces_with_newlines.py、github_metadata.py）

1.准备好指定仓库的id使用如下所示，运行replace_spaces_with_newlines.py即可
```txt
AnyMOD.jl.git apm_python.git apopt.git arduino.git CasADi.jl.git CellMLToolkit.jl.git cvxopt.git cyres.git dymos.git EnergyBasedModels.jl.git EnergyModels.jl.git EnergyPlus.git EnergySystemModeling.jl.git EPANET.git FOQUS.git fortran8.git GasModels.jl.git GasNetworkOptimization.git GEKKO.git GlobalEnergyGIS.git
```
2.爬取指定仓库的代码meta信息，运行github_metadata.py即可，返回如下
```
207887099, https://github.com/leonardgoeke/AnyMOD.jl.git
61572326, https://github.com/DataDog/dd-trace-py.git
```

## 📄 2. 仓库代码下载

运行github_downloader.py即可
