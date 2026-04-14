#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=scratch-node
#SBATCH --output=slurm-extract-latents-%j.out

echo "========================================"
echo "  AURORA LATENT EXTRACTION - HRES T0"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Date:         $(date)"
echo "Hostname:     $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "========================================"

PYTHON=/home/ekasteleyn/aurora_thesis/aurora_env/bin/python

echo "Python: $PYTHON"
$PYTHON --version

# Run the latent extraction script
# --num-steps 40: 40 × 6h = 10 days (WB2 evaluation range)
# --init-hours 0 12: Both init times for WB2 submission
# --latent-init-hour 12: Only extract latents for 12:00 UTC (thesis)
# --compile: Use torch.compile for faster inference
$PYTHON /home/ekasteleyn/aurora_thesis/thesis/scripts/steering/extract_latents_hres.py \
    --num-steps 40 \
    --init-hours 0 12 \
    --latent-init-hour 12 \
    --save-predictions \
    --compile

echo ""
echo "========================================"
echo "  COMPLETE: $(date)"
echo "========================================"
