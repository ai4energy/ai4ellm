#!/usr/bin/env python3
"""纯文本提取质量抽查：乱码率、语言分布、每页文本量、样本摘录。"""
import argparse
import json
import random
import re
import statistics
from pathlib import Path

MOJI = re.compile(r"[ÃÂÐÑâ€�\u0080-ÿ]{2,}")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def metrics(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chars = len(text)
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    nonprint = len(CTRL.findall(text))
    moji = len(MOJI.findall(text))
    dup = 0
    if lines:
        seen = set()
        for ln in lines:
            if ln in seen:
                dup += 1
            seen.add(ln)
    return {
        "chars": chars,
        "lines": len(lines),
        "cjk_ratio": round(cjk / chars, 4) if chars else 0,
        "latin_ratio": round(latin / chars, 4) if chars else 0,
        "nonprint": nonprint,
        "moji_hits": moji,
        "dup_line_ratio": round(dup / len(lines), 4) if lines else 0,
        "max_line": max((len(x) for x in lines), default=0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/ai4e-hermes/corpus/full-20261102-renamed")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--report", default="/home/ai4e-hermes/ai4ellm/runs/2026-11-02-corpus-qa/report.md")
    args = ap.parse_args()
    out = Path(args.out_dir)
    texts = sorted((out / "extracted_text").glob("*.txt"))
    pages = {}
    for m in (out / "metadata").glob("*_meta.json"):
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            pages[m.name[:-len("_meta.json")]] = int(d.get("page_count") or 0)
        except Exception:
            pass

    agg = []
    for p in texts:
        t = p.read_text(encoding="utf-8", errors="replace")
        m = metrics(t)
        m["name"] = p.name
        m["chars_per_page"] = round(m["chars"] / pages[p.stem], 1) if pages.get(p.stem) else None
        agg.append(m)

    total_chars = sum(x["chars"] for x in agg)
    cpp = [x["chars_per_page"] for x in agg if x["chars_per_page"]]
    low = sorted([x for x in agg if (x["chars_per_page"] or 0) < 200],
                 key=lambda x: x["chars_per_page"] or 0)
    noisy = sorted(agg, key=lambda x: (-x["nonprint"], -x["moji_hits"]))[:10]
    rng = random.Random(42)
    sample = rng.sample(agg, min(args.sample, len(agg)))

    report = {
        "snapshot_files": len(texts),
        "total_chars": total_chars,
        "chars_per_file_median": statistics.median(x["chars"] for x in agg) if agg else 0,
        "chars_per_page_median": statistics.median(cpp) if cpp else None,
        "cjk_ratio_median": statistics.median(x["cjk_ratio"] for x in agg) if agg else None,
        "latin_ratio_median": statistics.median(x["latin_ratio"] for x in agg) if agg else None,
        "low_text_files": len(low),
        "low_text_samples": [{"name": x["name"], "cpp": x["chars_per_page"]} for x in low[:15]],
        "noisy": [{"name": x["name"], "nonprint": x["nonprint"],
                   "moji_hits": x["moji_hits"], "chars": x["chars"]} for x in noisy],
        "sample": [],
    }
    for x in sample:
        text = (out / "extracted_text" / x["name"]).read_text(encoding="utf-8", errors="replace")
        report["sample"].append({**x, "head": text[:400].replace("\n", " ")})

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# 纯文本抽取质量抽查（快照）\n\n```json\n" +
        json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
