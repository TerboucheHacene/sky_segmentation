#!/bin/bash

#SBATCH --partition=low
#SBATCH --gres=gpu:1      # Request GPU "generic resources"
#SBATCH --cpus-per-gpu=8  # Cores proportional to GPUs
#SBATCH --mem-per-gpu=32000M       # Memory proportional to GPUs
#SBATCH --output=artifacts/out/%N-%j.out

poetry run python scripts/train.py \
    --learning_rate 1e-4 \
    --batch_size 80 \
    --max_epochs 30 \
    --gpus 1 \
