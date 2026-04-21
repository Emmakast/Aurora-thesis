#!/bin/bash
#SBATCH --job-name=lapse_rate_pangu
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_pangu_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_pangu_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# =============================================================================
# IMPORTANT: Replace the path below with the output of 'which python' 
# from your terminal when your environment is active!
# =============================================================================
PYTHON_EXEC="/gpfs/home5/ekasteleyn/aurora_thesis/aurora_env/bin/python"

cd /home/ekasteleyn/aurora_thesis/thesis/benchmark/scripts/

echo "Starting Lapse Rate evaluation for Pangu..."
echo "Using Python: $PYTHON_EXEC"

$PYTHON_EXEC run_lapse_rate.py \
    --model pangu \
    --year 2020 \
    --prediction-zarr gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
    --lead-hours 12,24,48,72,120,168,240

echo "Done!"
