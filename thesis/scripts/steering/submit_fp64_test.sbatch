#!/bin/bash
#SBATCH --job-name=aurora-fp64-test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --constraint=scratch-node
#SBATCH --output=slurm-fp64-test-%j.out

echo "========================================"
echo "  AURORA FP64 PRECISION TEST"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Date:         $(date)"
echo "Hostname:     $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "========================================"

PYTHON=/home/ekasteleyn/aurora_thesis/aurora_env/bin/python

echo "Python: $PYTHON"
$PYTHON --version

# Run fp64 test: predictions only (no latent extraction to save RAM)
# Use --latent-init-hour 99 to skip latent hooks entirely
$PYTHON -u /home/ekasteleyn/aurora_thesis/thesis/scripts/steering/extract_latents_hres.py \
    --dates 2022-05-11 \
    --num-steps 2 \
    --init-hours 12 \
    --latent-init-hour 99 \
    --save-predictions \
    --fp64

echo ""
echo "========================================"
echo "  COMPLETE: $(date)"
echo "========================================"
