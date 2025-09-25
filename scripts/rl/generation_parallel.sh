#!/bin/bash
set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
QUESTION_FILE=$2
IMAGE_FOLDER=$3
ANSWER_FILE_PREFIX=$4
BATCH_SIZE=$5

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/prepare_data/model_generation.py \
        --model-path ${CKPT} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ${ANSWER_FILE_PREFIX}_${IDX}.json \
        --batch_size ${BATCH_SIZE} \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${ANSWER_FILE_PREFIX}.json

# Clear out the output file if it exists.
> "$output_file"

python llava/prepare_data/merge_json.py --prefix ${ANSWER_FILE_PREFIX} --num-chunks ${CHUNKS}

rm -f ${ANSWER_FILE_PREFIX}_*.json
