#!/bin/bash
#SBATCH --job-name=steer_ao
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/steer_ao_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/steer_ao_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G

export ALPHAS="-0.5 -0.2 -0.1 -0.05 0.05 0.1 0.2 0.5"

PYTHONUNBUFFERED=1 /home/ekasteleyn/aurora_thesis/aurora_env/bin/python \
    /home/ekasteleyn/aurora_thesis/thesis/steering/scripts/steer_aurora.py \
    --phenomenon AO \
    --alphas $ALPHAS \
    --csv /home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_232.csv \
    --use-climatology \
    --name-suffix climo
