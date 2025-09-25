#!/bin/bash
set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
QUESTION_FILE=$2
IMAGE_FOLDER=$3
ANSWER_FILE_PREFIX=$4
MAX_TOKENS=$5

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_chair \
    --model-path ${CKPT} \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${ANSWER_FILE_PREFIX}_${IDX}.json \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --max_new_tokens ${MAX_TOKENS} \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &
done

wait

output_file=${ANSWER_FILE_PREFIX}.json

> "$output_file"

python llava/prepare_data/merge_chair_json.py --prefix ${ANSWER_FILE_PREFIX} --num-chunks ${CHUNKS}

rm -f ${ANSWER_FILE_PREFIX}_*.json
