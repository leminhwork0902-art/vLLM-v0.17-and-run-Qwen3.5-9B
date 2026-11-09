# Base CUDA 12.8 (yeu cau cua BTC: CUDA 12.8+)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# LUU Y: dau ngoac kep bat buoc, neu khong shell hieu ">=" la chuyen huong output.
# sqlite3 la module co san cua Python, khong cai qua pip.
# Neu vllm >=0.17 khong ho tro Python 3.10, doi base sang image co Python 3.12.
RUN pip3 install --no-cache-dir \
        "vllm>=0.17.0" \
        openai \
        pandas \
        openpyxl \
        numpy \
        tqdm

# Weights Qwen3.5-9B phai nam trong build context ./models/Qwen3.5-9B
COPY ./models /workspace/models
COPY ./src /workspace/src
COPY predict.sh /workspace/predict.sh
RUN chmod +x /workspace/predict.sh

# Khong dat ENTRYPOINT de BTC goi `bash predict.sh` duoc; mac dinh chay predict.sh.
CMD ["/bin/bash", "/workspace/predict.sh"]
