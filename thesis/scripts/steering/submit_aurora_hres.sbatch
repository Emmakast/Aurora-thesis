#!/bin/bash
#SBATCH --job-name=aurora_hres
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-aurora-hres-%j.out

echo "========================================"
echo "  AURORA 0.25 FINE-TUNED - IFS HRES T0"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Date:         $(date)"
echo "Hostname:     $(hostname)"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "========================================"

# Use venv Python explicitly
PYTHON=/home/ekasteleyn/aurora_thesis/aurora_env/bin/python
PIP=/home/ekasteleyn/aurora_thesis/aurora_env/bin/pip

echo "Python: $PYTHON"
$PYTHON --version

# Ensure dependencies are installed
$PIP install --quiet fsspec gcsfs zarr

# Run the validation script
$PYTHON /home/ekasteleyn/aurora_thesis/thesis/scripts/run_aurora_hres_official.py \
    --dates 2022-01-15 2022-05-15 2022-09-15 \
    --num-steps 3

echo ""
echo "========================================"
echo "  COMPLETE: $(date)"
echo "========================================"
