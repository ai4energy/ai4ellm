#!/usr/bin/env python3
"""PP-StructureV3 单页探针：版面 + 表格(HTML) + 公式(LaTeX) + 图区。

用法（.venv-paddle）:
  CUDA_VISIBLE_DEVICES=2 .venv-paddle/bin/python scripts/10_structure_probe.py \
      --image <png> --out <dir> --device gpu
"""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from paddleocr import PPStructureV3

    pipeline = PPStructureV3(
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_formula_recognition=True,
        use_chart_recognition=False,
        use_seal_recognition=False,
    )
    t0 = time.perf_counter()
    results = pipeline.predict(args.image)
    elapsed = time.perf_counter() - t0

    summary = {"elapsed_s": round(elapsed, 2), "results": []}
    for i, r in enumerate(results):
        info = {"index": i, "type": type(r).__name__}
        d = getattr(r, "json", None)
        if callable(d):
            d = d()
        if isinstance(d, dict):
            payload = d.get("res", d)
            info["keys"] = sorted(payload.keys())
            info["layout_labels"] = payload.get("layout_det_res", {}).get("label", []) if isinstance(payload.get("layout_det_res"), dict) else None
            info["markdown_chars"] = len(payload.get("markdown", "") or "")
            info["table_html_count"] = len(payload.get("table_res_list", []) or [])
            info["formula_res_count"] = len(payload.get("formula_res_list", []) or [])
            info["parsing_res_count"] = len(payload.get("parsing_res_list", []) or [])
        for method in ("save_to_json", "save_to_markdown"):
            fn = getattr(r, method, None)
            if callable(fn):
                try:
                    fn(str(out / f"page{i}"))
                    info[method] = "ok"
                except Exception as exc:  # noqa: BLE001
                    info[method] = f"{type(exc).__name__}: {exc}"
        summary["results"].append(info)
    print("STRUCTURE_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
