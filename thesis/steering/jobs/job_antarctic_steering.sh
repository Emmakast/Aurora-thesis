#!/bin/bash
#SBATCH --job-name=aao_antarctic_steering
#SBATCH --output=logs/aao_antarctic_steering_%j.out
#SBATCH --error=logs/aao_antarctic_steering_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Navigate to the data directory and subset the first 54 neutral dates (55 including the header)
cd /home/ekasteleyn/aurora_thesis/thesis/steering/data
head -n 55 target_dates_aao_neutral.csv > target_dates_aao_neutral_54_subset.csv

# Navigate back to the steering code directory
cd /home/ekasteleyn/aurora_thesis/thesis/steering

# Run the steering code with the subsetted neutral dates and Antarctic mask specific arguments
# Adjust the python script name and arguments to match your actual script standard
python run_steering.py \
    --neutral_dates data/target_dates_aao_neutral_54_subset.csv \
    --active_dates data/target_dates_aao_high.csv \
    --region antarctic \
    --apply_mask True
