FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        bzip2 \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

RUN mkdir -p /models/sherpa-onnx \
    && cd /models/sherpa-onnx \
    && curl -fSL -o sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2 \
        https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2 \
    && tar xjf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2 \
    && rm sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2

COPY main.py .

CMD ["python3", "main.py"]
