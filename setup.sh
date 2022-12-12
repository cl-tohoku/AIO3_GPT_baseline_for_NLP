#!/bin/bash

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    vim \
    curl \
    gcc

mkdir -p data work models

pip install --upgrade pip
pip install --user -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git                                   
pip install accelerate
pip install bitsandbytes

# Download data
wget -nc -O data/dev.jsonl https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl
wget -nc -O data/test.jsonl https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_dev_unlabeled_v1.0.jsonl
