#!/bin/bash


ARCH='vit_small'
NORM_LAST_LAYER='false'
DATA_PATH='/home/workspace/NLST//data/NLST/split'
PRETRAINED_WEIGHTS='./saving_dir/checkpoint-paco.pth'
CHECKPOINT_KEY='teacher'
OUTPUT_DIR='./saving_dir/three_classes_paco/linear_eval'
PATCH_SIZE='8'
NUM_LABELS='2'

# python \
#     eval_linear.py \
#     --pretrained_weights $PRETRAINED_WEIGHTS \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR

torchrun \
    --nproc_per_node=3 \
    eval_linear.py \
    --pretrained_weights $PRETRAINED_WEIGHTS \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR 