#!/bin/bash
set -e

LOG_FILE='./playground/data/eval/pope/scores0.csv'
QUESTION_FILE='./playground/data/eval/pope/llava_pope_test.jsonl'
IMAGE_FOLDER='./playground/data/coco/val2014'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MODEL_FAMILY=llava-v1.5-7b

#PROMPT="Describe the image and answer: "  # 1
#PROMPT="Describe the image and answer yes or no: "  # 2
#PROMPT="Respond with both a description of the image and an answer of yes or no for the question: "  # 3
#PROMPT="Describe the image"   # 4 dar
#PROMPT="Describe the image simply"   # 5 dar

#PROMPT=""   # 6

PROMPT=" Describe the image and answer the question. "   # 7

MODEL_NAMES=("$@")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="./checkpoints/${MODEL_NAME}"
    ANSWER_NAME="./playground/data/eval/pope/answers/p7_${MODEL_NAME}"
    bash ./scripts/rl/eval_pope/pope_gen_parallel_prompt.sh \
    ${MODEL_PATH} \
    ${QUESTION_FILE} \
    ${IMAGE_FOLDER} \
    ${ANSWER_NAME} \
    "$PROMPT"

    echo ${ANSWER_NAME}

    python llava/eval/eval_pope.py \
        --annotation-dir ./playground/data/eval/pope/coco \
        --question-file ${QUESTION_FILE} \
        --result-file ${ANSWER_NAME}.jsonl \
        --log-file ${LOG_FILE}

done

#LORA_MODEL_NAMES=(
#"new_lora_ia0_fp16_ep1_bs4_mmlr2e-5_lr1e-6-beta0.1-weight1.0-llava-v1.5-7b-dpo-d23cr77vg100_mixed_191k_llava-v1.5-7b_gen_llava-v1.5-7b_lvis_guide_replace_0.4_1_skip1_num1")
#
#MODEL_BASE=liuhaotian/llava-v1.5-7b
#
#for MODEL_NAME in "${LORA_MODEL_NAMES[@]}"; do
#    MODEL_PATH="./checkpoints/${MODEL_NAME}"
#    ANSWER_NAME="./playground/data/eval/pope/answers/p7_${MODEL_NAME}"
#
#    bash ./scripts/rl/eval_pope/pope_gen_parallel_prompt_lora.sh \
#    ${MODEL_PATH} \
#    ${QUESTION_FILE} \
#    ${IMAGE_FOLDER} \
#    ${ANSWER_NAME} \
#    ${MODEL_BASE} \
#    "$PROMPT"
#
#    echo ${ANSWER_NAME}
#    python llava/eval/eval_pope.py \
#        --annotation-dir ./playground/data/eval/pope/coco \
#        --question-file ${QUESTION_FILE} \
#        --result-file ${ANSWER_NAME}.jsonl \
#        --log-file ${LOG_FILE}
#
#done


#MODEL_PATH=liuhaotian/llava-v1.5-7b
#ANSWER_NAME="./playground/data/eval/pope/answers/p7_llava-v1.5-7b"
#bash ./scripts/rl/eval_pope/pope_gen_parallel_prompt.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#"$PROMPT"
#
#echo ${ANSWER_NAME}
#
#python llava/eval/eval_pope.py \
#    --annotation-dir ./playground/data/eval/pope/coco \
#    --question-file ${QUESTION_FILE} \
#    --result-file ${ANSWER_NAME}.jsonl \
#    --log-file ${LOG_FILE}
#
#SOTA_MODEL_NAMES=(
#"llava-v1.6-vicuna-7b"
#)
#
#for MODEL_NAME in "${SOTA_MODEL_NAMES[@]}"; do
#    MODEL_PATH="./ckpt/${MODEL_NAME}"
#    ANSWER_NAME="./playground/data/eval/pope/answers/p7_${MODEL_NAME}"
#    bash ./scripts/rl/eval_pope/pope_gen_parallel_prompt.sh \
#    ${MODEL_PATH} \
#    ${QUESTION_FILE} \
#    ${IMAGE_FOLDER} \
#    ${ANSWER_NAME} \
#    "$PROMPT"
#
#    echo ${ANSWER_NAME}
#
#    python llava/eval/eval_pope.py \
#        --annotation-dir ./playground/data/eval/pope/coco \
#        --question-file ${QUESTION_FILE} \
#        --result-file ${ANSWER_NAME}.jsonl \
#        --log-file ${LOG_FILE}
#
#done


