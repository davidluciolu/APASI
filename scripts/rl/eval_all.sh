#!/bin/bash
set -e
MODEL_NAMES=(
'YOUR_MODEL_NAME'
)

bash scripts/rl/eval_chair/eval_chair.sh "${MODEL_NAMES[@]}"
bash scripts/rl/eval_pope/eval_pope.sh "${MODEL_NAMES[@]}"

