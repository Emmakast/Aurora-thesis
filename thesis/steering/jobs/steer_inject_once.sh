#!/bin/bash
#SBATCH --job-name=aurora_inject_once
#SBATCH --output=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/inject_once_%j.out
#SBATCH --error=/home/ekasteleyn/aurora_thesis/thesis/steering/logs/inject_once_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=192G
#SBATCH --partition=gpu_a100
#SBATCH --constraint=scratch-node
#SBATCH --chdir=/home/ekasteleyn/aurora_thesis/thesis/steering/scripts

# ── Environment ──────────────────────────────────────────────────────────────
module load 2023
module load PROJ/9.2.0-GCCcore-12.3.0
source /home/ekasteleyn/aurora_thesis/aurora_env/bin/activate

# Create output and logs directories if they don't exist
mkdir -p /home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder\(2\)_inject
mkdir -p /home/ekasteleyn/aurora_thesis/thesis/steering/logs

# ── Run ──────────────────────────────────────────────────────────────────────
ALPHAS="-10.0 -5.0 -2.0 -1.0 1.0 2.0 5.0 10.0"

echo "Starting Aurora Steering (Inject Once)..."
echo "Running with alphas: $ALPHAS"

PYTHONUNBUFFERED=1 /home/ekasteleyn/aurora_thesis/aurora_env/bin/python steering/steer_aurora.py \
    --phenomenon AO \
    --csv /home/ekasteleyn/aurora_thesis/thesis/steering/data/target_dates_ao_81.csv \
    --steps 12 \
    --init-hour 12 \
    --alphas $ALPHAS \
    --output-dir "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_inject" \
    --name-suffix "ao81_inject" \
    --inject-once

echo "COMPLETE: $(date)"
