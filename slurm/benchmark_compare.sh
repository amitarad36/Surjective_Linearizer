#!/bin/bash
#SBATCH -A shocher_prj
#SBATCH -p rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --job-name=bench_compare

set -euo pipefail
mkdir -p logs

cd /home/amit.arad/work/Surjective_Linearizer
source venv/bin/activate

python scripts/benchmark_compare.py \
    --checkpoint outputs/flat_induced_v3/checkpoints/linearizer_epoch_100.pt
