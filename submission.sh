#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

python eval_model_jsonl_unlabel.py ${INPUT_FILE} --output_file ${OUTPUT_FILE}