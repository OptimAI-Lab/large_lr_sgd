#!/bin/bash


cd /XXX/large_lr_sgd/ || exit
wandb login XXX

python train_llama.py \
    --wandb_entity XXX \
    --wandb_project_name sgd-works \
    --model_name ray_1b_sgd-ll \
    --model_config $(pwd)/configs/llama_1b.json \
    --ray_use_gpu \
    --ray_num_workers 4 \
    --workers 0 \
    --optimizer sgd_ll \
    --lr 300 \
    --momentum 0.9 \
    --batch_size 64 \
    --total_batch_size 4096 \
    --weight_decay 0.0 \
    --num_training_steps 19125 \
    --warmup_steps 1912 \
    --eval_every 1000 \
    --dataset_path=/XXX/c4/en \
    --amp \
    --scheduler cosine \
    --eval_in_fp32 \
    --compile_model \
    --update_limit 2e-4 \
    --col_thresh 1e-3

exit
