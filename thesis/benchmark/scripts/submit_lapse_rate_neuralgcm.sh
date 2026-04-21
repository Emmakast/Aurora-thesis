#!/bin/bash
#SBATCH --job-name=lapse_rate_neuralgcm
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_neuralgcm_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_neuralgcm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# =============================================================================
# IMPORTANT: Replace the path below with the output of 'which python' 
# from your terminal when your environment is active!
# =============================================================================
PYTHON_EXEC="/gpfs/home5/ekasteleyn/aurora_thesis/aurora_env/bin/python"

cd /home/ekasteleyn/aurora_thesis/thesis/benchmark/scripts/

echo "Starting Lapse Rate evaluation for NeuralGCM..."
echo "Using Python: $PYTHON_EXEC"

$PYTHON_EXEC run_lapse_rate.py \
    --model neuralgcm \
    --year 2020 \
    --prediction-zarr gs://weatherbench2/datasets/neuralgcm_deterministic/2020-512x256.zarr \
    --era5-zarr gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr \
    --lead-hours 12,24,48,72,120,168,240

echo "Done!"