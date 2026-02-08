FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /home/neuronet_team270/project-dhvani
