#!/bin/bash

# REASONING_MODEL_PATH="pretrained_models/VisionReasoner-3B"
# REASONING_MODEL_PATH="/data/cr7/visionreasoner_workdir/run_visionreasoner_3b/global_step_250/actor/huggingface"
# sft version
# REASONING_MODEL_PATH="pretrained_models/VisionReasoner-3B-1k"
REASONING_MODEL_PATH="/data/cr7/vlreasoner/qwen2_5vl-3b/full/sft"

SEGMENTATION_MODEL_PATH="facebook/sam2-hiera-large"

export NCCL_P2P_DISABLE=1
MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*pretrained_models\/(.*)\/actor\/.*/\1/')
# TEST_DATA_PATH="Ricky06662/ReasonSeg_test"
TEST_DATA_PATH="Ricky06662/ReasonSeg_val"


TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./reasonseg_eval_results/${MODEL_DIR}/${TEST_NAME}"

NUM_PARTS=8
# Create output directory
mkdir -p $OUTPUT_PATH

# Run 8 processes in parallel
for idx in {0..7}; do
    export CUDA_VISIBLE_DEVICES=$idx
    python evaluation_scripts/evaluation_visionreasoner_sft.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --segmentation_model_path $SEGMENTATION_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --batch_size 50 &
done

# Wait for all processes to complete
wait

python evaluation_scripts/calculate_iou.py --output_dir $OUTPUT_PATH