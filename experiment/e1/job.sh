#!/bin/bash

# VARIABLE
PROJECT_ROOT="/media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch"
EXP_NAME="e1"
CONFIG_FILE="experiment/${EXP_NAME}/deeplabv3plus_mobilenet_v3_large_cityscapes.yaml"
MODEL_CHECKPOINT=""

cd $PROJECT_ROOT
pwd

# Any additional command-line arguments will override the config file
python main.py \
    --config $CONFIG_FILE \
    --exp_name $EXP_NAME \

    # --continue-training \
    # ckpt $MODEL_CHECKPOINT \

    # --test_only \
    # --save_val_results \