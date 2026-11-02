#!/usr/bin/env python3
"""PaddleOCR 单页探针：测设备/耗时/识别文本量。

用法（在 .venv-paddle 中）:
  CUDA_VISIBLE_DEVICES=2 .venv-paddle/bin/python scripts/09_ocr_probe.py \
      --image runs/.../pages/xxx.png --device gpu
"""
import argparse
import json
import time


def collect_text(result) -> list[str]:
    texts: list[str] = []
    for r in result:
        d = getattr(r, "json", None)
        if callable(d):
            d = d()
        if d is None and isinstance(r, dict):
            d = r
        if not isinstance(d, dict):
            continue
        payload = d.get("res", d)
        texts.extend(payload.get("rec_texts") or [])
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--lang", default="ch")
    args = ap.parse_args()

    import paddle
    from paddleocr import PaddleOCR

    print(json.dumps({
        "paddle": paddle.__version__,
        "compiled_with_cuda": paddle.device.is_compiled_with_cuda(),
        "device_count": paddle.device.cuda.device_count(),
    }), flush=True)

    kwargs = dict(
        lang=args.lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    try:
        ocr = PaddleOCR(device=args.device, **kwargs)
    except TypeError:
        ocr = PaddleOCR(**kwargs)

    t0 = time.perf_counter()
    result = ocr.predict(args.image)
    elapsed = time.perf_counter() - t0
    texts = collect_text(result)
    out = {
        "device": args.device,
        "elapsed_s": round(elapsed, 2),
        "lines": len(texts),
        "chars": sum(len(t) for t in texts),
        "text_head": "\n".join(texts)[:800],
    }
    print("OCR_RESULT " + json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
