#!/bin/bash -x

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    vim \
    curl \
    python3-venv \
    gcc

python3 -m venv .venv
. .venv/bin/activate

mkdir -p outputs

pip install --upgrade pip
pip install -r requirements.txt

# if you need load_in_8bit
#pip install git+https://github.com/huggingface/transformers.git                                   
#pip install accelerate
#pip install bitsandbytes

# Download data
wget -nc https://storage.googleapis.com/lecnlp/models.tar.gz
wget -nc https://storage.googleapis.com/lecnlp/data.tar.gz
tar xvzf models.tar.gz
tar xvzf data.tar.gz

