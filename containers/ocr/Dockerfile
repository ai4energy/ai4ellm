# AI4ELLM OCR 镜像：PaddleOCR 3.7 + PaddlePaddle GPU 3.4 (CUDA 13.0)
# 独立 /venv-paddle，不污染 base 的 /venv（torch/transformers）。
# Build:
#   docker build --network=host -t ai4ellm-ocr:paddleocr3.7 ~/ai4ellm/containers/ocr
# Run:
#   docker run --rm --gpus '"device=2"' -v /data/models:/models:ro \
#     -v /data:/data ai4ellm-ocr:paddleocr3.7
FROM aigc-base:torch213-cu134

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PADDLE_PDX_CACHE_HOME=/models/paddleocr
ENV PADDLE_PDX_MODEL_SOURCE=bos
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv-paddle && \
    /venv-paddle/bin/pip install --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip \
    /venv-paddle/bin/pip install \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
        paddlepaddle-gpu==3.4.0 \
        paddleocr==3.7.0 \
        "paddlex[ocr]==3.7.2" \
        pymupdf==1.28.2

WORKDIR /work
CMD ["/venv-paddle/bin/python", "-c", "import paddle; print('paddle', paddle.__version__, 'gpus', paddle.device.cuda.device_count())"]
