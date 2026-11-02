#!/usr/bin/env python3
"""用本地 Qwen3.8-27B 对 scope review 书名做二次判定（KEEP/DROP）。"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

MODEL = "/models/llm/Qwen3.8-27B-bf16"
SYSTEM = (
    "你是能源、化工、电气领域的资料管理员。判断书名是否属于能源、化工、电气或"
    "紧密相关的工程领域（热力、流体、过程、电力电子、材料化工、冶金等）。"
    "只回答一个词：KEEP 或 DROP。"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="/work/data/corpus-index/scope_decision.jsonl")
    ap.add_argument("--out", default="/work/data/corpus-index/scope_qwen.jsonl")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.decisions, encoding="utf-8")]
    todo = [r for r in rows if r["decision"] == "review"]
    if args.max_items:
        todo = todo[:args.max_items]
    print(f"review items: {len(todo)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=False)
    model.eval()
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    out = []
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start:start + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"书名：{r['name']}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in chunk
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for r, text in zip(chunk, texts):
            verdict = "KEEP" if "KEEP" in text.upper() else ("DROP" if "DROP" in text.upper() else "KEEP")
            out.append({"name": r["name"], "path": r["path"],
                        "heuristic": r["decision"], "qwen": verdict,
                        "raw": text.strip()})
        if (start // args.batch_size + 1) % 10 == 0:
            print(f"{start + len(chunk)}/{len(todo)}", flush=True)

    Path(args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")
    keep = sum(1 for r in out if r["qwen"] == "KEEP")
    print(json.dumps({"total": len(out), "keep": keep, "drop": len(out) - keep,
                      "out": args.out}, indent=2), flush=True)


if __name__ == "__main__":
    main()
