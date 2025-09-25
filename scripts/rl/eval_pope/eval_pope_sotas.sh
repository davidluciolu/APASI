#!/bin/bash
set -e

LOG_FILE='./playground/data/eval/pope/scores_sotas.csv'
QUESTION_FILE='./playground/data/eval/pope/llava_pope_test.jsonl'
IMAGE_FOLDER='./playground/data/coco/val2014'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_FAMILY=llava-v1.5-7b

MODEL_NAME=RLAIF-V-llava_7B
MODEL_PATH="./ckpt/${MODEL_NAME}"
ANSWER_NAME="./playground/data/eval/pope/answers/${MODEL_NAME}"

#bash ./scripts/rl/eval_pope/pope_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME}
#
#echo ${ANSWER_NAME}
#python llava/eval/eval_pope.py \
#    --annotation-dir ./playground/data/eval/pope/coco \
#    --question-file ${QUESTION_FILE} \
#    --result-file ${ANSWER_NAME}.jsonl \
#    --log-file ${LOG_FILE}

#MODEL_NAME=LLaVA-1.5-7B-SIMA
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#ANSWER_NAME="./playground/data/eval/pope/answers/${MODEL_NAME}"
#
#bash ./scripts/rl/eval_pope/pope_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME}
#
#echo ${ANSWER_NAME}
#python llava/eval/eval_pope.py \
#    --annotation-dir ./playground/data/eval/pope/coco \
#    --question-file ${QUESTION_FILE} \
#    --result-file ${ANSWER_NAME}.jsonl \
#    --log-file ${LOG_FILE}

MODEL_NAME=llava-v1.6-mistral-7b-STIC
MODEL_PATH="./ckpt/${MODEL_NAME}"
ANSWER_NAME="./playground/data/eval/amber/answers/${MODEL_NAME}"

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
