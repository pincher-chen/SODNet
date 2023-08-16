#!/bin/bash

#conda activate equiformer2

export PYTHONNOUSERSITE=True    # prevent using packages from base

CUDA_VISIBLE_DEVICES=0 python -u main_sc.py \
    --output-dir 'models/SuperCon/' \
    --model-name 'graph_attention_transformer_nonlinear_l2_e3_noNorm' \
    --input-irreps '100x0e' \
    --data-path 'datasets/SuperCon' \
    --order-type 'all' \
    --run-fold 6 \
    --feature-type 'crystalnet' \
    --batch-size 64 \
    --epochs 150 \
    --radius 8.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-5 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp
