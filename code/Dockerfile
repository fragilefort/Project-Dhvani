FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir --upgrade pip

RUN python -m pip install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install --no-cache-dir \
    transformers==4.41.2 \
    datasets==2.19.2 \
    evaluate==0.4.2 \
    wandb==0.17.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    audiomentations==0.39.0

WORKDIR /app

COPY train_model.py .

CMD ["python", "train_model.py"]

