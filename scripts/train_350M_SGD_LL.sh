#!/bin/bash


cd /XXX/large_lr_sgd/ || exit
wandb login XXX

python train_llama.py \
    --wandb_entity XXX \
    --wandb_project_name sgd-works \
    --model_name ray_350m_sgd-ll \
    --model_config $(pwd)/configs/llama_350m.json \
    --ray_use_gpu \
    --ray_num_workers 4 \
    --workers 0 \
    --optimizer sgd_ll \
    --lr 200 \
    --momentum 0.9 \
    --batch_size 128 \
    --total_batch_size 2048 \
    --weight_decay 0.0 \
    --num_training_steps 15000 \
    --warmup_steps 1500 \
    --eval_every 1000 \
    --dataset_path=/XXX/c4/en \
    --amp \
    --scheduler cosine \
    --eval_in_fp32 \
    --compile_model \
    --update_limit 5e-4 \
    --col_thresh 1e-3

exit
