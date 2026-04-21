#!/bin/bash
#SBATCH --job-name=lapse_rate_graphcast
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_graphcast_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/benchmark/results/logs/lapse_rate_graphcast_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# =============================================================================
# IMPORTANT: Replace the path below with the output of 'which python' 
# from your terminal when your environment is active!
# =============================================================================
PYTHON_EXEC="/gpfs/home5/ekasteleyn/aurora_thesis/aurora_env/bin/python"

cd /home/ekasteleyn/aurora_thesis/thesis/benchmark/scripts/

echo "Starting Lapse Rate evaluation for GraphCast..."
echo "Using Python: $PYTHON_EXEC"

$PYTHON_EXEC run_lapse_rate.py \
    --model graphcast \
    --year 2020 \
    --prediction-zarr gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr \
    --lead-hours 12,24,48,72,120,168,240

echo "Done!"