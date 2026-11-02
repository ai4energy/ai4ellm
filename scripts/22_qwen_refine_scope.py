#!/usr/bin/env python3
"""细化 scope 重判：全部书名 → KEEP/DROP（能源/化工/电气/PSE 保留）。"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = """你是能源、化工、电气领域的资料管理员。严格按以下规则判断书名：
KEEP：
- 能源：发电、电网、储能、新能源、氢能、电池、动力、燃烧、锅炉、汽轮机、热力；
- 化工：过程、反应、分离、催化、石化、材料化工、冶金、传热传质、流体/泵/阀；
- 电气：电机、电力电子、自动化、控制、仪器、电缆/开关柜/继电保护；
- PSE 过程系统工程：过程建模/仿真/优化/控制/安全，含制药过程工程
  （制药工程、冻干、结晶、制粒、连续制造、PAT、QbD、工艺放大）。
DROP：
- 历史、考试/题库、小说/文学、经管/金融/营销/管理（非技术系统的管理）；
- 纯医学/临床/护理/药理/解剖/制药临床；
- 纯计算机：算法、编程、数据库、网络、安全、纯机器学习理论、软件工程；
- 其他无关领域。
计算机只有与能源/化工/电气/PSE 紧密结合（电力系统计算、CFD、过程仿真、
控制优化、数字孪生、故障诊断、SCADA/EMS）才 KEEP。
只回答一个词：KEEP 或 DROP。"""
URL = "http://127.0.0.1:18112/v1/chat/completions"


def classify(name: str) -> tuple[str, str]:
    body = json.dumps({
        "model": "qwen-scope",
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"书名：{name}"}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"].strip()
    up = text.upper()
    return ("KEEP" if "KEEP" in up else ("DROP" if "DROP" in up else "KEEP")), text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_final.jsonl")
    ap.add_argument("--out", default="/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_refined.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.decisions, encoding="utf-8")]
    print(f"classify {len(rows)}", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify, r["name"]): r for r in rows}
        for i, fut in enumerate(futures, 1):
            r = futures[fut]
            try:
                verdict, raw = fut.result()
            except Exception as exc:  # noqa: BLE001
                verdict, raw = "KEEP", f"ERROR {type(exc).__name__}: {exc}"
            out.append({"name": r["name"], "path": r["path"],
                        "current": r["final"], "refined": verdict, "raw": raw})
            if i % 200 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    keep = sum(1 for r in out if r["refined"] == "KEEP")
    print(json.dumps({"total": len(out), "keep": keep, "drop": len(out) - keep,
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
