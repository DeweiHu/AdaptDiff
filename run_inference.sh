#!/bin/bash


MASK_PATH="/home/dewei/Desktop/ConditionalDiffusion"
CKPT_PATH="/home/dewei/Desktop/ConditionalDiffusion/ckpts"
SAVE_PATH="/home/dewei/Desktop/ConditionalDiffusion"

MASK_NAME="binary_mask.png"
CKPT_NAME="diffusion.octa500.pt"
SAVE_NAME="test_result_2"


python /home/dewei/Desktop/ConditionalDiffusion/adapter_inference.py \
--mask_path $MASK_PATH \
--mask_name $MASK_NAME \
--ckpt_path $CKPT_PATH \
--ckpt_name $CKPT_NAME \
--save_path $SAVE_PATH \
--save_name $SAVE_NAME

