#!/bin/bash
#SBATCH --job-name=daily_rmse
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-daily-rmse-%j.out

# Activate environment
source ~/aurora_thesis/aurora_env/bin/activate

# Change to scripts directory
cd ~/aurora_thesis/thesis/scripts

# Run with unbuffered output for real-time logging
python -u compute_daily_rmse_wb2.py --model all --year 2020

echo "Job completed at $(date)"
