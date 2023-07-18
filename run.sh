#!/bin/bash

ARCH='vit_small'
NORM_LAST_LAYER='false'
DATA_PATH='/home/reem/ncai/lung/data/NLST/split/df_train_cons_norm_lw.csv'
OUTPUT_DIR='./saving_dir/2'
PRETRAINED_WEIGHTS='./saving_dir/2/checkpoint.pth'
CHECKPOINT_KEY='teacher'


# python \
#     -m \
#     torch.distributed.launch \
#     --nproc_per_node=3 \
#     main_dino.py \
#     --arch $ARCH \
#     --norm_last_layer false \
#     --data_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR

torchrun \
    --nproc_per_node=3 \
    main_dino.py \
    --arch $ARCH \
    --norm_last_layer false \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR