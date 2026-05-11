#!/bin/bash
#SBATCH --job-name=check_hydro_tv
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

BASE_DIR="$HOME/aurora_thesis/neuripspaper"
SCRIPT="$BASE_DIR/scripts/check_hydrostatic_tv.py"
PY="$HOME/aurora_thesis/aurora_env/bin/python"

echo "Running targeted hydrostatic Tv vs T evaluation..."
$PY $SCRIPT
