FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    jq \
    wget \
    python3-dev\
    vim \
    curl \
    gcc \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /code/AIO3_GPT_baseline
COPY requirements.txt /code/AIO3_GPT_baseline
WORKDIR /code/AIO3_GPT_baseline

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code/AIO3_GPT_baseline

