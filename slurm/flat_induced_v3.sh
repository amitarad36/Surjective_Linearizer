#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=flat_ind_v3

set -euo pipefail
mkdir -p logs

cd /rg/shocher_prj/amit.arad/Surjective_Linearizer
source venv/bin/activate

python -u train.py \
    --epochs 201 \
    --batch_size 64 \
    --steps 100 \
    --eval_epoch 10 \
    --sampling_method rk \
    --latent_size 128 \
    --lora_rank 8 \
    --noise_level 0.1 \
    --var_match_lambda 8 \
    --exp_name flat_induced_v3
