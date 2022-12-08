#!/bin/bash

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    vim \
    curl \
    gcc

mkdir -p data work models

pip install --upgrade pip
pip install --user -r requirements.txt

# Download data
wget -nc -P data https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl
wget -nc -P data https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_dev_unlabeled_v1.0.jsonl

head data/aio_02_train.jsonl -n 1000 >> data/aio_02_train_1000.jsonl
