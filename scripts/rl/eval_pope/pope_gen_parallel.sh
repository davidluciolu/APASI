#!/bin/bash
set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
QUESTION_FILE=$2
IMAGE_FOLDER=$3
ANSWER_FILE_PREFIX=$4

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ${ANSWER_FILE_PREFIX}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${ANSWER_FILE_PREFIX}.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${ANSWER_FILE_PREFIX}_${IDX}.jsonl >> "$output_file"
done

rm -f ${ANSWER_FILE_PREFIX}_*.jsonl
