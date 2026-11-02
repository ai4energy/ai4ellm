#!/usr/bin/env python3
"""用临时 vLLM (127.0.0.1:18112) 对 review 书名做 KEEP/DROP 分类。"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SYSTEM = (
    "你是能源、化工、电气领域的资料管理员。判断书名是否属于能源、化工、电气或"
    "紧密相关的工程领域（热力、流体、过程、电力电子、材料化工、冶金等）。"
    "只回答一个词：KEEP 或 DROP。"
)
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
    verdict = "KEEP" if "KEEP" in up else ("DROP" if "DROP" in up else "KEEP")
    return verdict, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_decision.jsonl")
    ap.add_argument("--out", default="/home/ai4e-hermes/ai4ellm/data/corpus-index/scope_qwen.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.decisions, encoding="utf-8")]
    todo = [r for r in rows if r["decision"] == "review"]
    print(f"review={len(todo)}", flush=True)
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(classify, r["name"]): r for r in todo}
        for i, fut in enumerate(futures, 1):
            r = futures[fut]
            try:
                verdict, raw = fut.result()
            except Exception as exc:  # noqa: BLE001
                verdict, raw = "KEEP", f"ERROR {type(exc).__name__}: {exc}"
            out.append({"name": r["name"], "path": r["path"],
                        "heuristic": "review", "qwen": verdict, "raw": raw})
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    keep = sum(1 for r in out if r["qwen"] == "KEEP")
    print(json.dumps({"total": len(out), "keep": keep, "drop": len(out) - keep,
                      "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
