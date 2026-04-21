#!/bin/bash
#SBATCH --job-name=lapse_rate_hres
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_hres_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_hres_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# =============================================================================
# IMPORTANT: Replace the path below with the output of 'which python' 
# from your terminal when your environment is active!
# Example: "/gpfs/home5/ekasteleyn/miniconda3/envs/climate/bin/python"
# =============================================================================
PYTHON_EXEC="/gpfs/home5/ekasteleyn/aurora_thesis/aurora_env/bin/python"

cd /home/ekasteleyn/aurora_thesis/thesis/benchmark/scripts/

echo "Starting Lapse Rate evaluation for HRES..."
echo "Using Python: $PYTHON_EXEC"

$PYTHON_EXEC run_lapse_rate.py \
    --model hres \
    --year 2020 \
    --prediction-zarr gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr \
    --lead-hours 12,24,48,72,120,168,240

echo "Done!"
