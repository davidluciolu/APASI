set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

IMAGE_DIR=$1
REF_NAME=$2
MODEL_FAMILY=$3
SOURCE=$4
ITERATION=$5

REP_RATE=$6
SENT_THRES=$7
SKIP_SENT=$8
NUM_NEG=$9

python llava/prepare_data/data_construct_llm.py \
  --cap_file ${SOURCE}.json \
  --replace_rate ${REP_RATE} \
  --sent_threshold ${SENT_THRES} \
  --skip_sent ${SKIP_SENT} \
  --weighted_sample \
  --model_path ${REF_NAME} \
  --model_family ${MODEL_FAMILY} \
  --caption_obj_file ${SOURCE}_lvis_processed.json \
  --use_obj_guide \
  --num_neg ${NUM_NEG} \
  --iteration ${ITERATION} \
  --image_dir ${IMAGE_DIR} \
  --find_rep_candidate


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/prepare_data/data_construct_llm.py \
        --cap_file ${SOURCE}.json \
        --replace_rate ${REP_RATE} \
        --sent_threshold ${SENT_THRES} \
        --skip_sent ${SKIP_SENT} \
        --weighted_sample \
        --model_path ${REF_NAME} \
        --model_family ${MODEL_FAMILY} \
        --caption_obj_file ${SOURCE}_lvis_processed.json \
        --use_obj_guide \
        --num_neg ${NUM_NEG} \
        --iteration ${ITERATION} \
        --image_dir ${IMAGE_DIR} \
        --num-chunks $CHUNKS \
        --conv-mode vicuna_v1 \
        --chunk-idx $IDX &
done

wait


python llava/prepare_data/data_construct_llm.py \
  --cap_file ${SOURCE}.json \
  --replace_rate ${REP_RATE} \
  --sent_threshold ${SENT_THRES} \
  --skip_sent ${SKIP_SENT} \
  --weighted_sample \
  --model_path ${REF_NAME} \
  --model_family ${MODEL_FAMILY} \
  --caption_obj_file ${SOURCE}_lvis_processed.json \
  --use_obj_guide \
  --num_neg ${NUM_NEG} \
  --iteration ${ITERATION} \
  --image_dir ${IMAGE_DIR} \
  --merge-parquets \
  --num-chunks $CHUNKS

