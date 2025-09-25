#!/bin/bash
export WANDB_PROJECT="llava-dpo"
export WANDB_MODE="offline"
set -e

BASE_DATASET="./playground/data/instructions/detail_23k.json"
GEN_BATCHSIZE=6
REF_NAME=liuhaotian/llava-v1.5-7b
MODEL_FAMILY=llava-v1.5-7b

LR=4e-7
BETA=0.1
WEIGHT=1.0
ITER=0

SOURCE=detail_23k_${MODEL_FAMILY}_ia${ITER}_ep3_gen
DATA=${SOURCE}_${MODEL_FAMILY}_lvis_guide_replace_0.2_1_skip1_num1

MODEL_NAME=lora_ia${ITER}_lr${LR}-beta${BETA}-weight${WEIGHT}-${MODEL_FAMILY}-dpo-${DATA}
deepspeed \
    llava/train/train_dpo_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${REF_NAME} \
    --version v1 \
    --data_root ./playground/data/neg_data/ \
    --neg_data ${DATA} \
    --source_data ${SOURCE} \
    --iteration 0 \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --bf16 False \
    --output_dir ./checkpoints/${MODEL_NAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8\
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${MODEL_NAME} \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight ${WEIGHT} \
    --dpo_beta ${BETA}


#python scripts/merge_lora_weights.py \
#    --model-path ./checkpoints/${MODEL_NAME} \
#    --model-base ${REF_NAME} \
#    --save-model-path ./checkpoints/merged_ia${ITER}_lr${LR}-beta${BETA}-weight${WEIGHT}-${MODEL_FAMILY}-dpo-${DATA}

rm -rf ./checkpoints/*/checkpoint*/global_step*
export WANDB_MODE=online
wandb sync ./wandb/offline-run-*
mv ./wandb/offline-run-* ./wandb_synced
