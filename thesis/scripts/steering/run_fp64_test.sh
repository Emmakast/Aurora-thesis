#!/bin/bash
#SBATCH --job-name=aurora-fp64-test
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-fp64-test-%j.out

echo "Starting FP64 precision test..."
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Run with the correct Python from the venv
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python /home/ekasteleyn/aurora_thesis/thesis/scripts/steering/test_fp64_precision.py

echo "Done!"
