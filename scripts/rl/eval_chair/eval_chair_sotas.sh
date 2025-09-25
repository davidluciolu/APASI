#!/bin/bash
set -e

LOG_FILE='./playground/data/eval/chair/scores_sotas.csv'
QUESTION_FILE=./playground/data/eval/chair/coco_val2014_500_opera.jsonl
IMAGE_FOLDER='./playground/data/coco/val2014'
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_FAMILY=llava-v1.5-7b

#SOURCE=detail_23k_llava_gen
#DATA=${SOURCE}_llava-v1.5-7b_lvis_guide_replace_0.4_1_skip1_num1
#LR=4e-8
#BETA=0.1
#WEIGHT=1.0
#MODEL_NAME=new_fp16_ep5_mmlr2e-5_lr${LR}-beta${BETA}-weight${WEIGHT}-llava-v1.5-7b-dpo-${DATA}
#
#
#MAX_TOKENS=512
#STEP=362
#MODEL_PATH="./checkpoints/${MODEL_NAME}/checkpoint-${STEP}"
#ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${STEP}-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MODEL_BASE} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"


#MAX_TOKENS=512
#MODEL_NAME=RLAIF-V-llava_7B
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"
#
#
#MODEL_NAME=CSR_LLaVA_1.5_7b_1Iteration/llava_merged_dpo_7b_1epoch_1iteration
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/csr_iter1-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"
#
#
#MODEL_NAME=CSR_LLaVA_1.5_7b_2Iteration/llava_merged_dpo_7b_1epoch_2iteration
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/csr_iter2-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"
#
#
#MODEL_NAME=CSR_LLaVA_1.5_7b_3Iteration
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/csr_iter3-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"
#
#
#MODEL_NAME=llava_POVID_stage_one_merged
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"
#
#
#MODEL_NAME=llava_POVID_stage_two_lora
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#MODEL_BASE=./ckpt/llava_POVID_stage_one_merged
#ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel_lora.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MODEL_BASE} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"

#MAX_TOKENS=512
#MODEL_NAME=LLaVA-1.5-7B-SIMA
#MODEL_PATH="./ckpt/${MODEL_NAME}"
#
#ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${MAX_TOKENS}"
#
#bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
#${MODEL_PATH} \
#${QUESTION_FILE} \
#${IMAGE_FOLDER} \
#${ANSWER_NAME} \
#${MAX_TOKENS}
#
#python llava/eval/eval_chair.py \
#    --log-file ${LOG_FILE} \
#    --cap_file "${ANSWER_NAME}.json"

MAX_TOKENS=512
MODEL_NAME=llava-v1.6-mistral-7b-STIC

MODEL_PATH="./ckpt/${MODEL_NAME}"
ANSWER_NAME="./playground/data/eval/chair/answers/${MODEL_NAME}-${MAX_TOKENS}"

bash ./scripts/rl/eval_chair/chair_gen_parallel.sh \
${MODEL_PATH} \
${QUESTION_FILE} \
${IMAGE_FOLDER} \
${ANSWER_NAME} \
${MAX_TOKENS}

python llava/eval/eval_chair.py \
    --log-file ${LOG_FILE} \
    --cap_file "${ANSWER_NAME}.json"
