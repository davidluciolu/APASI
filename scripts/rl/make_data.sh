#!/bin/bash
set -e

BASE_DATASET="./playground/data/instructions/detail_23k.json"
GEN_BATCHSIZE=6
REF_NAME=liuhaotian/llava-v1.5-7b
MODEL_FAMILY=llava-v1.5-7b

ITER=0

SOURCE=detail_23k_${MODEL_FAMILY}_ia${ITER}_ep3_gen
DATA=${SOURCE}_${MODEL_FAMILY}_lvis_guide_replace_0.2_1_skip1_num1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/rl/generation_parallel.sh \
${REF_NAME} \
${BASE_DATASET} \
./playground/data/coco/train2017 \
./playground/data/neg_data/${SOURCE} \
${GEN_BATCHSIZE}

python ./llava/prepare_data/co_graph_construction.py --cap_file ${SOURCE} --obj_vocab_mode lvis

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/rl/llm_guide_replace_parallel.sh \
./playground/data/coco/train2017 \
${REF_NAME} \
${MODEL_FAMILY} \
${SOURCE} \
0 \
0.2 \
1 \
1 \
1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/rl/prepare_logp_parallel.sh \
${DATA} \
${REF_NAME} \
${MODEL_FAMILY} \
${SOURCE} \
0
