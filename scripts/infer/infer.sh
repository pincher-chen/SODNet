#!/bin/bash


export PYTHONNOUSERSITE=True    # prevent using packages from base

CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --model  'best_models/' \
    --data_path 'datasets/SuperCon' \

