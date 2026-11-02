# P0 训练镜像: aigc-base:torch213-cu134 + QLoRA 依赖
# Build (root docker):
#   docker build --network=host -t aigc-train:torch213-cu134 ~/ai4ellm/containers/train
FROM aigc-base:torch213-cu134

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# 与 base 中 torch 2.13.0 / transformers 5.12.1 / accelerate 1.15.0 配套钉版
RUN /venv/bin/pip install --no-cache-dir \
        bitsandbytes==0.50.2 \
        peft==0.21.2 \
        trl==1.17.1 \
        datasets==5.1.0 \
        accelerate==1.15.0 \
    && /venv/bin/pip freeze > /opt/train-pip-freeze.txt

LABEL org.aigc.train="qlora" \
      org.aigc.torch="2.13.0" \
      org.aigc.transformers="5.12.1" \
      org.aigc.bitsandbytes="0.50.2" \
      org.aigc.peft="0.21.2"
