#!/bin/bash -x

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    vim \
    curl \
    gcc

mkdir -p outputs

pip install --upgrade pip
pip install -r requirements.txt

# Download data
wget -nc https://storage.googleapis.com/lecnlp/models.tar.gz
wget -nc https://storage.googleapis.com/lecnlp/data.tar.gz
tar xvzf models.tar.gz
tar xvzf data.tar.gz
