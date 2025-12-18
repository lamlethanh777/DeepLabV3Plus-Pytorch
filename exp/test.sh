#!/bin/bash

cd /media02/tphung/workspace-Lam/cv_project/DeepLabV3Plus-Pytorch
pwd

python main.py \
    --model deeplabv3plus_resnet101 \
    --gpu_id 0 \
    --dataset cityscapes \
    --data_root datasets/data/cityscapes \
    --crop_val \
    --lr 0.01 \
    --crop_size 513 \
    --batch_size 16 \
    --output_stride 16 \
    --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth \
    --test_only \
    --save_val_results