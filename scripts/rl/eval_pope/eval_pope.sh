#!/bin/bash
set -e

LOG_FILE='./playground/data/eval/pope/scores0.csv'
QUESTION_FILE='./playground/data/eval/pope/llava_pope_test.jsonl'
IMAGE_FOLDER='./playground/data/coco/val2014'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_FAMILY=llava-v1.5-7b

#SOURCE=detail_23k_llava_gen
#DATA=${SOURCE}_llava-v1.5-7b_lvis_guide_replace_0.4_1_skip1_num1
#LR=4e-8
#BETA=0.1
#WEIGHT=1.0
#MODEL_NAME=new_fp16_ep5_mmlr2e-5_lr${LR}-beta${BETA}-weight${WEIGHT}-llava-v1.5-7b-dpo-${DATA}
#

#STEP=362
#MODEL_PATH="./checkpoints/${MODEL_NAME}/checkpoint-${STEP}"
#ANSWER_NAME="./playground/data/eval/amber/answers/${MODEL_NAME}-${STEP}"
#
#bash ./scripts/rl/eval_pope/pope_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MODEL_BASE}
#
#echo ${ANSWER_NAME}
#python llava/eval/eval_pope.py \
#    --annotation-dir ./playground/data/eval/pope/coco \
#    --question-file ${QUESTION_FILE} \
#    --result-file ${ANSWER_NAME}.jsonl \
#    --log-file ${LOG_FILE}

MODEL_NAMES=("$@")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="./checkpoints/${MODEL_NAME}"
    ANSWER_NAME="./playground/data/eval/pope/answers/${MODEL_NAME}"
    bash ./scripts/rl/eval_pope/pope_gen_parallel.sh \
    ${MODEL_PATH} \
    ${QUESTION_FILE} \
    ${IMAGE_FOLDER} \
    ${ANSWER_NAME}

    echo ${ANSWER_NAME}
    python llava/eval/eval_pope.py \
        --annotation-dir ./playground/data/eval/pope/coco \
        --question-file ${QUESTION_FILE} \
        --result-file ${ANSWER_NAME}.jsonl \
        --log-file ${LOG_FILE}

done

