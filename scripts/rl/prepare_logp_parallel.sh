set -e
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
DATA=$1
REF_NAME=$2
MODEL_FAMILY=$3
SOURCE=$4
ITERATION=$5
ANSWER_FILE_PREFIX=./playground/data/neg_data/${DATA}/${SOURCE}_${MODEL_FAMILY}_iter_${ITERATION}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llava/prepare_data/prepare_dpo_logp.py \
        --neg_data ${DATA} \
        --model_path ${REF_NAME} \
        --model_family ${MODEL_FAMILY} \
        --source_data ${SOURCE}  \
        --iteration ${ITERATION} \
        --overwrite_logps \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait

output_file=${ANSWER_FILE_PREFIX}.parquet

> "$output_file"

python llava/prepare_data/merge_parquet.py --prefix ${ANSWER_FILE_PREFIX} --num-chunks ${CHUNKS}

rm -f ${ANSWER_FILE_PREFIX}_*.parquet
