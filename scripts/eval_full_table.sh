#!/bin/bash
#SBATCH --job-name=eval_full_table
#SBATCH --output=slurm_logs/eval_table_%j.out
#SBATCH --error=slurm_logs/eval_table_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6k-shocher
#SBATCH --qos=contrib
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

source /rg/shocher_prj/amit.arad/Surjective_Linearizer/venv/bin/activate
cd /rg/shocher_prj/amit.arad/Surjective_Linearizer
export PYTHONPATH=/rg/shocher_prj/amit.arad/Surjective_Linearizer

CHECKPOINT=outputs/flat_induced_v3/checkpoints/linearizer_epoch_100.pt
REAL_DIR=img_align_celeba
BASE=outputs/flat_induced_v3/fid_eval

mkdir -p $BASE

echo "========== 1 step =========="
python scripts/compute_fid.py --checkpoint $CHECKPOINT --real_dir $REAL_DIR \
    --output_dir $BASE/steps_1 --steps 1 --method rk \
    --exp_name flat_induced_v3_fid_1step

echo "========== 10 steps =========="
python scripts/compute_fid.py --checkpoint $CHECKPOINT --real_dir $REAL_DIR \
    --output_dir $BASE/steps_10 --steps 10 --method rk \
    --exp_name flat_induced_v3_fid_10steps

echo "========== 100 steps (reuse existing samples) =========="
python scripts/compute_fid_only.py \
    --generated_dir outputs/flat_induced_v3/fid_samples_ep100 \
    --real_dir $REAL_DIR

echo "========== one-step 100->1 =========="
python scripts/compute_fid.py --checkpoint $CHECKPOINT --real_dir $REAL_DIR \
    --output_dir $BASE/one_step_100 --method one_step --T 100 \
    --exp_name flat_induced_v3_fid_onestep100

echo "========== one-step 1000->1 =========="
python scripts/compute_fid.py --checkpoint $CHECKPOINT --real_dir $REAL_DIR \
    --output_dir $BASE/one_step_1000 --method one_step --T 1000 \
    --exp_name flat_induced_v3_fid_onestep1000
