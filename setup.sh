#!/bin/bash

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    python3-dev\
    vim \
    curl \
    gcc


mkdir -p data work models

#### TODO: add ####
pip install --upgrade pip
pip install --user -r requirements.txt

#wget -P data https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_02/aio_02_train.jsonl
#wget -P data https://jaqket.s3.ap-northeast-1.amazonaws.com/data/aio_03/aio_03_test_unlabeled.jsonl
